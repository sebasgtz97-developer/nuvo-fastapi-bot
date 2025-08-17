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

# ⬇️ Nuevo: importamos NLU + formateo humano
from openai_agent import interpret_intent, format_answer

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

# ---------- Helpers de respuesta ----------

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

# ---------- Lookups existentes ----------
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

    # 3) Revenue por cliente + mes (pattern clásico)
    m2 = re.search(r"revenue.*?(cliente|shipper)\s+([A-Za-z0-9 .,&\-]+?)\s+en\s+([a-záéíóú]+)", text, re.I)
    if m2 and {"TOTAL_REVENUE", "CREATED_AT_MX"}.issubset(set(df.columns)):
        entity_name = m2.group(2).strip()
        mes_txt = m2.group(3).strip().lower()
        month_map = {
            "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
            "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
        }
        mm = month_map.get(mes_txt)
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

# ---------- NLU helpers (pandas ejecutor por intención) ----------

MONTHS_ES = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}

def _first_existing(df: pd.DataFrame, candidates):
    return next((c for c in candidates if c in df.columns), None)

def _shipment_id_from_text_or_intent(text: str, intent_obj):
    # del intent
    ent = (intent_obj or {}).get("entity")
    if ent and re.search(r"\d{3,}", ent):
        m = re.search(r"(\d{3,})", ent)
        if m:
            return m.group(1)
    # del texto
    m2 = SHIP_ID_RE.search(text or "")
    return m2.group(1) if m2 else None

async def answer_with_nlu(text: str, df: pd.DataFrame) -> str:
    """
    Usa interpret_intent (LLM ligero) y ejecuta pandas seguro según intención,
    luego formatea con format_answer.
    """
    # 1) NLU (con timeout para Twilio)
    try:
        intent_obj = await asyncio.wait_for(run_in_threadpool(interpret_intent, text, df), timeout=5.0)
    except asyncio.TimeoutError:
        return ("Estoy procesando tu solicitud. "
                "Prueba: “¿Quién cubrió el shipment id 63357?” o "
                "“Revenue de Porcelana Corona en julio”.")
    except Exception:
        intent_obj = {"intent": "generic", "entity": None, "entity_kind": "none", "month": None, "year": None, "metric": "none", "top_n": None}

    intent = (intent_obj or {}).get("intent", "generic")

    # 2) Ejecutar pandas según intención
    try:
        if intent == "shipment_lookup":
            sid = _shipment_id_from_text_or_intent(text, intent_obj)
            if not sid:
                return "Necesito el shipment id (por ejemplo: 63357)."
            carrier = direct_lookup(text, df)  # ya usa regex en el propio texto
            if not carrier:
                # intenta con SHIPMENT_ID directo
                if "SHIPMENT_ID" in df.columns:
                    hit = df.loc[df["SHIPMENT_ID"].astype(str).str.strip() == str(sid)]
                    carrier = hit["CARRIER_NAME"].iloc[0] if not hit.empty and "CARRIER_NAME" in hit else None
            return format_answer("shipment_lookup", {"shipment_id": sid, "carrier": carrier})

        if intent == "pickup_date":
            sid = _shipment_id_from_text_or_intent(text, intent_obj)
            if not sid:
                return "Necesito el shipment id para el pickup date."
            # columnas candidatas
            pickup_cols = ["PICKUP_STARTS_AT", "PICKUP_AT_ORIGIN_", "PICKUP_DATE", "SCHEDULED_PICKUP_AT_ORIGIN_"]
            pcol = _first_existing(df, pickup_cols)
            date_val = None
            if pcol:
                hit = df.loc[df["SHIPMENT_ID"].astype(str).str.strip() == str(sid), pcol] if "SHIPMENT_ID" in df.columns else pd.Series([])
                if not hit.empty:
                    date_val = pd.to_datetime(hit.iloc[0], errors="coerce")
                    date_val = date_val.strftime("%Y-%m-%d %H:%M") if pd.notnull(date_val) else None
            return format_answer("pickup_date", {"shipment_id": sid, "date": date_val})

        if intent == "in_transit":
            status_col = _first_existing(df, ["STATUS", "SHIPMENT_STATUS", "CURRENT_STATUS"])
            cols = [c for c in ["SHIPMENT_ID", "SHIPMENT_NAME", "CARRIER_NAME"] if c in df.columns]
            if not status_col:
                return "No encuentro una columna de estatus para identificar 'en tránsito'."
            hits = df[df[status_col].astype(str).str.contains("TRANSIT", case=False, na=False)]
            rows = []
            if not cols:
                cols = df.columns[:3].tolist()
            for r in hits[cols].head(20).to_dict(orient="records"):
                parts = []
                for k in cols:
                    if k in r:
                        parts.append(str(r[k]))
                rows.append(" · ".join(parts))
            return format_answer("in_transit", {"total": int(len(hits)), "rows": rows})

        if intent == "revenue_lookup":
            shipper_col = _first_existing(df, ["SHIPPER_NAME", "CUSTOMER_NAME", "CLIENTE"])
            if not shipper_col or "TOTAL_REVENUE" not in df.columns:
                return "No encuentro columnas para calcular revenue por cliente."
            ent = (intent_obj or {}).get("entity")
            mes_txt = (intent_obj or {}).get("month")
            mm = MONTHS_ES.get((mes_txt or "").lower()) if mes_txt else None
            if "CREATED_AT_MX" in df.columns:
                df["CREATED_AT_MX"] = pd.to_datetime(df["CREATED_AT_MX"], errors="coerce")
            mask = df[shipper_col].astype(str).str.contains(ent or "", case=False, na=False)
            if mm and "CREATED_AT_MX" in df.columns:
                mask = mask & (df["CREATED_AT_MX"].dt.month == mm)
            total = pd.to_numeric(df["TOTAL_REVENUE"], errors="coerce")[mask].sum()
            return format_answer("revenue_lookup", {"entity": ent or "ese cliente", "period": mes_txt or "el periodo indicado", "value": float(total)})

        if intent == "carrier_stats":
            carrier_col = _first_existing(df, ["CARRIER_NAME", "CARRIER"])
            ent = (intent_obj or {}).get("entity")
            if not carrier_col or not ent:
                return "Necesito el nombre del carrier para dar estadísticas."
            subset = df[df[carrier_col].astype(str).str.contains(ent, case=False, na=False)]
            n_ship = len(subset)
            total_rev = float(pd.to_numeric(subset.get("TOTAL_REVENUE", pd.Series(dtype=float)), errors="coerce").sum()) if "TOTAL_REVENUE" in subset else 0.0
            gp = 0.0
            if "GROSS_PROFIT" in subset:
                gp = float(pd.to_numeric(subset["GROSS_PROFIT"], errors="coerce").sum())
            elif "TOTAL_REVENUE" in subset and "TOTAL_COST" in subset:
                gp = float(pd.to_numeric(subset["TOTAL_REVENUE"], errors="coerce").sum() - pd.to_numeric(subset["TOTAL_COST"], errors="coerce").sum())
            stats_txt = f"shipments: {n_ship}, revenue: ${total_rev:,.2f}, GP: ${gp:,.2f}"
            return format_answer("carrier_stats", {"carrier": ent, "stats": stats_txt})

        # Fallback genérico
        return "No estoy seguro de lo que pides. Prueba con: “¿Quién cubrió el shipment id 63357?”, “Revenue de Porcelana Corona en julio” o “¿Cuáles van en tránsito?”."

    except Exception as e:
        return f"No pude procesar la consulta: {e}"

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

# Conservamos /ask como sanity check: usa el router rápido y luego NLU
@app.get("/ask", summary="Ask Gsheet")
async def ask_gsheet(q: str):
    df = get_df()
    fast = quick_router(q, df)
    if fast is not None:
        return format_reply_from_result(fast) if not isinstance(fast, str) else fast
    return await answer_with_nlu(q, df)

@app.post("/whatsapp", summary="Whatsapp Webhook")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    message_body = (form.get("Body") or "").strip()

    # default rápido
    reply = "Recibido ✅"

    try:
        df = get_df()

        # A) Router rápido sin LLM (instantáneo)
        fast = quick_router(message_body, df)
        if fast is not None:
            return twiml_response(format_reply_from_result(fast) if not isinstance(fast, str) else fast)

        # B) NLU + pandas + formato humano con timeout duro
        try:
            nlu_reply = await asyncio.wait_for(answer_with_nlu(message_body, df), timeout=7.0)
            if nlu_reply:
                reply = nlu_reply
        except asyncio.TimeoutError:
            reply = (
                "Estoy consultando datos y puede tardar.\n"
                "Ejemplos:\n"
                "• ¿Quién cubrió el shipment id 63357?\n"
                "• Revenue de *Porcelana Corona* en *julio*.\n"
                "• Shipments *en tránsito* hoy."
            )
        except Exception as e:
            reply = f"No pude entender la consulta: {e}"

    except Exception as e:
        print("webhook error:", e)

    return twiml_response(reply)










