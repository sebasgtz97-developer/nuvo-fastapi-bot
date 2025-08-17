import os
import re
import pandas as pd
import asyncio
from starlette.concurrency import run_in_threadpool
from time import time
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from xml.sax.saxutils import escape
from openai_agent import interpret_query
from gsheet import load_data


load_dotenv()  # loads env vars; DO NOT print secrets

app = FastAPI(title="nuvo-fastapi-bot", version="1.0.0")

# ---------- Simple cache for the Google Sheet ----------
_SHEET_CACHE = {"df": None, "ts": 0.0}
_CACHE_TTL = 300  # seconds

def get_df():
    now = time()
    if _SHEET_CACHE["df"] is not None and now - _SHEET_CACHE["ts"] < _CACHE_TTL:
        return _SHEET_CACHE["df"]
    df = load_data()
    _SHEET_CACHE["df"] = df
    _SHEET_CACHE["ts"] = now
    return df

# ---------- Helpers ----------

def extract_code(raw: str) -> str:
    """
    Devuelve una sola expresión segura sobre df/pd.
    Rechaza código que declare variables o use nombres no permitidos.
    """
    if not raw:
        return ""

    # 1) bloques con ```python
    blocks = re.findall(r"```(?:python)?\s*([^`]+)```", raw, flags=re.IGNORECASE)
    candidates = [b.strip() for b in blocks]

    # 2) df[...] o df.algo inline
    candidates += [m.strip() for m in re.findall(r"(df\s*\.[^\n]+|df\s*\[[^\n]+)", raw, flags=re.IGNORECASE)]

    # 3) líneas que mencionan df[
    candidates += [ln.strip() for ln in raw.splitlines() if "df[" in ln or ln.strip().startswith("df.")]

    WHITELIST_PREFIX = ("df[", "df.", "pd.")  # solo expresiones sobre df/pd
    SAFE = []
    for c in candidates:
        c = c.split("#")[0].strip().rstrip(";")
        c = re.sub(r"^[^d]*?(?=df\s*[\[\.]|pd\.)", "", c, flags=re.IGNORECASE)
        if c.count("[") > c.count("]"):
            c += "]"
        if not c.startswith(WHITELIST_PREFIX):
            continue
        # No asignaciones ni estructuras de control
        if re.search(r"\b(import|for|while|lambda|def|class)\b|=", c):
            continue
        SAFE.append(c)

    return SAFE[0] if SAFE else ""



def safe_eval_df_expr(code: str, df: pd.DataFrame):
    import io, contextlib
    from datetime import datetime
    import builtins

    if not code:
        raise RuntimeError("No se pudo interpretar la pregunta en una expresión de pandas sobre df.")

    try:
        # Normalizaciones
        date_cols = ["CREATED_AT_MX", "PICKUP_STARTS_AT", "INVOICED_AT", "ACTUAL_DELIVERED_TO_DESTINATION_"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        money_cols = ["GROSS_PROFIT", "TOTAL_COST", "TOTAL_REVENUE", "FREIGHT_COST", "FREIGHT_REVENUE"]
        for col in money_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        local_vars = {"df": df, "pd": pd, "datetime": datetime}

        if "\n" in code or ";" in code:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, {"__builtins__": builtins.__dict__}, local_vars)
            for var_name in reversed(list(local_vars.keys())):
                if isinstance(local_vars[var_name], (int, float, str, pd.Series, pd.DataFrame)):
                    return local_vars[var_name]
        return eval(code, {"__builtins__": builtins.__dict__}, local_vars)

    except NameError as ne:
        raise RuntimeError(f"No conozco esa variable en la expresión: {ne}. "
                           "Intenta mencionar *cliente, carrier o shipment id* explícitamente.")
    except Exception as e:
        raise RuntimeError(f"Eval error: {e} | code={code}")



def fallback_carrier_lookup(message: str, df: pd.DataFrame):
    match = re.search(r"(SH-)?(\d{3,})", (message or "").upper())
    if not match:
        return None
    ship_id = match.group(2)

    if "SHIPMENT_ID" in df.columns:
        res = df.loc[df["SHIPMENT_ID"].astype(str).str.strip() == ship_id, "CARRIER_NAME"]
        if not res.empty:
            return res

    if "SHIPMENT_NAME" in df.columns:
        alt = f"SH-{ship_id}"
        res = df.loc[df["SHIPMENT_NAME"].astype(str).str.strip().str.upper() == alt, "CARRIER_NAME"]
        if not res.empty:
            return res
    return None


# Deterministic mini-QL for "shipment id 12345"
SHIP_ID_RE = re.compile(r"(?:shipment\s*id|shipment\s*|sh-?)\s*[:#]?\s*(\d{3,})", re.I)

def direct_lookup(message: str, df: pd.DataFrame):
    m = SHIP_ID_RE.search(message or "")
    if not m:
        return None
    sid = m.group(1)

    if "SHIPMENT_ID" in df.columns:
        hit = df.loc[df["SHIPMENT_ID"].astype(str).str.strip() == sid]
        if not hit.empty and "CARRIER_NAME" in hit:
            return hit["CARRIER_NAME"].iloc[0]

    if "SHIPMENT_NAME" in df.columns:
        alt = f"SH-{sid}"
        hit = df.loc[df["SHIPMENT_NAME"].astype(str).str.upper().str.strip() == alt]
        if not hit.empty and "CARRIER_NAME" in hit:
            return hit["CARRIER_NAME"].iloc[0]
    return None

def quick_router(msg: str, df: pd.DataFrame):
    """
    Resuelve rápido y sin LLM:
    - ¿Quién cubrió el shipment id 12345?
    - ¿Cuáles son los shipment id en tránsito?
    - ¿Cuánto revenue dejó el cliente/shipper X en <mes>?
    """
    text = (msg or "").strip()

    # 1) Shipment ID → carrier
    m = re.search(r"(?:shipment\s*id|shipment\s*|sh-?)\s*[:#]?\s*(\d{3,})", text, re.I)
    if m:
        sid = m.group(1)
        direct = direct_lookup(text, df)
        if direct:
            return f"📄 Carrier name: {direct}"

    # 2) En tránsito
    if re.search(r"\ben tránsito\b|\bin transit\b", text, re.I):
        status_col_candidates = ["STATUS", "SHIPMENT_STATUS", "CURRENT_STATUS"]
        st_col = next((c for c in status_col_candidates if c in df.columns), None)
        if st_col:
            hits = df[df[st_col].astype(str).str.contains("TRANSIT", case=False, na=False)]
            cols = [c for c in ["SHIPMENT_ID", "SHIPMENT_NAME", "CARRIER_NAME"] if c in df.columns]
            if not cols:
                cols = df.columns[:3].tolist()
            return hits[cols].head(20)

    # 3) Revenue por cliente + mes
    m2 = re.search(r"revenue.*?(cliente|shipper)\s+([A-Za-z0-9 .,&\-]+?)\s+en\s+([a-záéíóú]+)", text, re.I)
    if m2 and {"TOTAL_REVENUE", "CREATED_AT_MX"}.issubset(set(df.columns)):
        entity_name = m2.group(2).strip()
        mes_txt = m2.group(3).strip().lower()
        month_map = {
            "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
            "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
        }
        mm = month_map.get(mes_txt)
        # nombre del campo de cliente/shipper (ajusta si usas otro)
        shipper_col_candidates = ["SHIPPER_NAME", "CUSTOMER_NAME", "CLIENTE"]
        shipper_col = next((c for c in shipper_col_candidates if c in df.columns), None)

        if mm and shipper_col:
            df["CREATED_AT_MX"] = pd.to_datetime(df["CREATED_AT_MX"], errors="coerce")
            mask = (
                df[shipper_col].astype(str).str.contains(entity_name, case=False, na=False) &
                (df["CREATED_AT_MX"].dt.month == mm)
            )
            total = pd.to_numeric(df["TOTAL_REVENUE"], errors="coerce")[mask].sum()
            return f"💵 Revenue de {entity_name} en {mes_txt}: ${total:,.2f}"

    return None



def format_reply_from_result(result):
    def is_money_column(col_name):
        keywords = ["COST", "REVENUE", "PROFIT", "RATE", "PRICE"]
        return any(k in col_name.upper() for k in keywords)

    def format_money(val):
        try:
            num = float(val)
            return f"${num:,.2f}"
        except:
            return val

    if isinstance(result, pd.Series):
        if result.empty:
            return "No encontré información con ese criterio."
        key = result.name
        val = result.iloc[0]
        if is_money_column(key):
            val = format_money(val)
        key = key.replace("_", " ").capitalize()
        return f"📄 {key}: {val}"

    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No encontré información con ese criterio."
        lines = []
        for row in result.to_dict(orient="records"):
            parts = []
            for k, v in row.items():
                if is_money_column(k):
                    v = format_money(v)
                parts.append(f"{k.replace('_', ' ').capitalize()}: {v}")
            lines.append(" • " + ", ".join(parts))
        return "\n".join(lines)

    return str(result)


def twiml_response(body_text: str) -> PlainTextResponse:
    safe_text = escape(body_text or "")
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe_text}</Message></Response>'
    return PlainTextResponse(content=xml, media_type="application/xml; charset=utf-8")


# ---------- Utility Routes ----------
@app.get("/", summary="Root")
def root():
    return {"service": "nuvo-fastapi-bot", "status": "running"}

@app.get("/healthz", summary="Healthz")
def healthz():
    return {"ok": True}

# Minimal TwiML echo to validate Twilio pipeline quickly
@app.post("/twilio/test")
def twilio_test():
    return twiml_response("Hola 👋 Funciona.")


# ---------- Business Routes ----------
@app.get("/ask", summary="Ask Gsheet")
def ask_gsheet(q: str):
    df = get_df()
    try:
        raw_code = interpret_query(q, df)
        code = extract_code(raw_code)
        print("🧠 Código generado (/ask):", raw_code, "=>", code)
        result = safe_eval_df_expr(code, df)
        return format_reply_from_result(result)
    except Exception as e:
        return {"error": str(e)}


@app.post("/whatsapp", summary="Whatsapp Webhook")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    message_body = (form.get("Body") or "").strip()

    # default rápido
    reply = "Recibido ✅"

    try:
        df = get_df()

        # A) Router rápido sin LLM
        fast = quick_router(message_body, df)
        if fast is not None:
            return twiml_response(format_reply_from_result(fast) if not isinstance(fast, str) else fast)

        # B) Si no resolvió, intentar LLM con timeout duro
        async def llm_flow():
            raw_code = await run_in_threadpool(interpret_query, message_body, df)
            code = extract_code(raw_code)
            print("🧠 Código generado (WhatsApp):", raw_code, "=>", code)
            result = await run_in_threadpool(safe_eval_df_expr, code, df)
            return (format_reply_from_result(result) or "").strip()

        try:
            llm_reply = await asyncio.wait_for(llm_flow(), timeout=7.0)
            if llm_reply:
                reply = llm_reply
        except asyncio.TimeoutError:
            reply = (
                "Estoy consultando datos y puede tardar.\n"
                "Ejemplos:\n"
                "• ¿Quién cubrió el shipment id 63357?\n"
                "• Total revenue de *Porcelana Corona* en *julio*.\n"
                "• Shipments *en tránsito* hoy."
            )
        except Exception as e:
            reply = f"No pude entender la consulta: {e}"

    except Exception as e:
        print("webhook error:", e)

    return twiml_response(reply)









