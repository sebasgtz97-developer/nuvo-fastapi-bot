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
from openai_agent import build_column_resolver, col
from openai_agent import interpret_intent, format_answer
from gsheet import load_data
import unicodedata


load_dotenv()  # loads env vars; DO NOT print secrets

app = FastAPI(title="nuvo-fastapi-bot", version="1.0.0")

# ---------- Simple cache for the Google Sheet ----------
_SHEET_CACHE = {"df": None, "ts": 0.0}
_CACHE_TTL = 300  # seconds

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def pick_pickup_date_col(df):
    """
    Elige la mejor columna de 'pickup date' disponible, en este orden:
    1) ACTUAL_PICKED_UP_FROM_ORIGIN_  (real)
    2) PICKUP_STARTS_AT               (programado/ready)
    3) APPOINTMENT_TIME_AT_ORIGIN_LOCAL_TIME (cita)
    4) SCHEDULED_PICKUP_AT_ORIGIN_
    5) PICKUP_DATE
    """
    R = getattr(df, "__colresolver__", {}) or {}
    candidates = [
        "ACTUAL_PICKED_UP_FROM_ORIGIN_",
        "PICKUP_STARTS_AT",
        "APPOINTMENT_TIME_AT_ORIGIN_LOCAL_TIME",
        "SCHEDULED_PICKUP_AT_ORIGIN_",
        "PICKUP_DATE",
    ]
    for key in candidates:
        colname = (col(df, R, key) or key)
        if colname in df.columns:
            return colname
    return None

def in_transit_mask(df):
    """
    Devuelve una máscara booleana 'en tránsito' robusta:
    - acepta IN TRANSIT, IN-TRANSIT, EN TRANSITO, EN TRÁNSITO, etc.
    """
    R = getattr(df, "__colresolver__", {}) or {}
    status_col = (col(df, R, "SHIPMENT_STATUS") or col(df, R, "CURRENT_STATUS") or "SHIPMENT_STATUS")
    if status_col not in df.columns:
        return None, None
    # normalizamos acentos y espacios
    series = df[status_col].astype(str).map(_strip_accents).str.upper()
    mask = series.str.contains(r"IN.?TRANSIT|EN.?TRANSITO", regex=True, na=False)
    return status_col, mask

def get_df():
    now = time()
    if _SHEET_CACHE["df"] is not None and now - _SHEET_CACHE["ts"] < _CACHE_TTL:
        return _SHEET_CACHE["df"]
    df = load_data()
    # Construir resolver de sinónimos una vez por carga
    try:
        df.__colresolver__ = build_column_resolver(df)
    except Exception:
        df.__colresolver__ = {}
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

    id_col = col(df, getattr(df, "__colresolver__", {}), "SHIPMENT_ID") or "SHIPMENT_ID"
    name_col = col(df, getattr(df, "__colresolver__", {}), "SHIPMENT_NAME") or "SHIPMENT_NAME"
    carrier_col = col(df, getattr(df, "__colresolver__", {}), "CARRIER_NAME") or "CARRIER_NAME"

    if id_col in df.columns:
        res = df.loc[df[id_col].astype(str).str.strip() == ship_id, carrier_col] if carrier_col in df.columns else pd.Series([])
        if not res.empty:
            return res

    if name_col in df.columns:
        alt = f"SH-{ship_id}"
        res = df.loc[df[name_col].astype(str).str.strip().str.upper() == alt, carrier_col] if carrier_col in df.columns else pd.Series([])
        if not res.empty:
            return res
    return None

# Deterministic mini-QL for "shipment id 12345"
SHIP_ID_RE = re.compile(r"(?:shipment\s*id|shipment\s*|sh-?)\s*[:#]?\s*(\d{3,})", re.I)

def _prefer_name(df, guess: str, name_col: str, id_col: str) -> str:
    """
    Si el resolver nos dio un *_ID, pero existe el *_NAME, usa el *_NAME.
    """
    if guess and "ID" in guess.upper() and name_col in df.columns:
        return name_col
    if guess:
        return guess
    return name_col if name_col in df.columns else id_col


def _prefer_name(df, guess: str, name_col: str, id_col: str) -> str:
    if guess and "ID" in guess.upper() and name_col in df.columns:
        return name_col
    if guess:
        return guess
    return name_col if name_col in df.columns else id_col

def direct_lookup(message: str, df: pd.DataFrame):
    m = SHIP_ID_RE.search(message or "")
    if not m:
        return None
    sid = m.group(1)

    R = getattr(df, "__colresolver__", {}) or {}
    id_col = col(df, R, "SHIPMENT_ID") or "SHIPMENT_ID"
    name_col = col(df, R, "SHIPMENT_NAME") or "SHIPMENT_NAME"
    carrier_guess = col(df, R, "CARRIER_NAME") or "CARRIER_NAME"
    carrier_col = _prefer_name(df, carrier_guess, "CARRIER_NAME", "CARRIER_ID_")

    if id_col in df.columns and carrier_col in df.columns:
        hit = df.loc[df[id_col].astype(str).str.strip() == sid, carrier_col]
        if not hit.empty:
            return hit.iloc[0]

    if name_col in df.columns and carrier_col in df.columns:
        alt = f"SH-{sid}"
        hit = df.loc[df[name_col].astype(str).str.upper().str.strip() == alt, carrier_col]
        if not hit.empty:
            return hit.iloc[0]
    return None



def quick_router(msg: str, df: pd.DataFrame):
    """
    Resuelve rápido y sin LLM:
    - ¿Quién cubrió el shipment id 12345?
    - ¿Cuáles están en tránsito?
    - Revenue del cliente X en <mes> (bajado al pickup date)
    - ¿Cuál es la ruta/lane del shipment 12345?
    """
    text = (msg or "").strip()
    R = getattr(df, "__colresolver__", {}) or {}

    # 0) Ruta / lane / route del shipment
    if re.search(r"\b(ruta|lane|route)\b", text, re.I):
        m = re.search(r"(?:shipment\s*id|shipment\s*|sh-?)\s*[:#]?\s*(\d{3,})", text, re.I)
        if m:
            sid = m.group(1)
            id_col = col(df, R, "SHIPMENT_ID") or "SHIPMENT_ID"
            lane_col = col(df, R, "CITY_TO_CITY_ROUTE")
            if lane_col and id_col in df.columns and lane_col in df.columns:
                val = df.loc[df[id_col].astype(str).str.strip() == sid, lane_col]
                if not val.empty:
                    return format_answer("generic", {"text": f"🧭 La ruta del shipment {sid} es *{val.iloc[0]}*."})
            o_city = col(df, R, "ORIGIN_CITY")
            d_city = col(df, R, "DESTINATION_CITY")
            if id_col in df.columns and o_city in df.columns and d_city in df.columns:
                row = df.loc[df[id_col].astype(str).str.strip() == sid, [o_city, d_city]]
                if not row.empty:
                    a, b = row.iloc[0].tolist()
                    return format_answer("generic", {"text": f"🧭 La ruta del shipment {sid} es *{a} → {b}*."})

    # 1) Shipment ID → carrier
    m = re.search(r"(?:shipment\s*id|shipment\s*|sh-?)\s*[:#]?\s*(\d{3,})", text, re.I)
    if m:
        sid = m.group(1)
        direct = direct_lookup(text, df)
        if direct:
            return format_answer("shipment_lookup", {"shipment_id": sid, "carrier": direct})

    # 2) En tránsito (robusto: IN-TRANSIT / EN TRÁNSITO / EN TRANSITO)
    status_col, mask = in_transit_mask(df)
    if re.search(r"\ben tránsito\b|\bin transit\b|\btransito\b|\btr[aá]nsito\b", text, re.I):
        if status_col is not None:
            hits = df[mask]
            cols = [
                x for x in [
                    col(df, R, "SHIPMENT_ID") or "SHIPMENT_ID",
                    col(df, R, "SHIPMENT_NAME") or "SHIPMENT_NAME",
                    col(df, R, "CARRIER_NAME") or "CARRIER_NAME",
                ] if x in df.columns
            ]
            if not cols:
                cols = df.columns[:3].tolist()
            rows = []
            for r in hits[cols].head(20).to_dict(orient="records"):
                rows.append(" · ".join([str(r.get(k, "")) for k in cols]))
            return format_answer("in_transit", {"total": int(len(hits)), "rows": rows})

    # 3) Revenue por cliente + mes (bajado al PICKUP DATE)
    m2 = re.search(r"revenue.*?(cliente|shipper|customer|company)\s+([A-Za-z0-9 .,&\-]+?)\s+en\s+([a-záéíóú]+)", text, re.I)
    if m2:
        entity_name = m2.group(2).strip()
        mes_txt = m2.group(3).strip().lower()
        month_map = {
            "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
            "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
        }
        mm = month_map.get(mes_txt)

        shipper_col = (col(df, R, "COMPANY_NAME") or col(df, R, "SHIPPER_NAME") or col(df, R, "CUSTOMER_NAME"))
        revenue_col = (col(df, R, "TOTAL_REVENUE") or col(df, R, "FREIGHT_REVENUE"))
        date_col = pick_pickup_date_col(df)  # 👈 bajamos al pickup date

        if shipper_col and revenue_col and shipper_col in df.columns and revenue_col in df.columns:
            mask_shipper = df[shipper_col].astype(str).str.contains(entity_name, case=False, na=False)
            if mm and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                mask_shipper = mask_shipper & (df[date_col].dt.month == mm)
            total = pd.to_numeric(df[revenue_col], errors="coerce")[mask_shipper].sum()
            return format_answer("revenue_lookup", {"entity": entity_name, "period": mes_txt, "value": float(total)})

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
    R = getattr(df, "__colresolver__", {}) or {}

    # 2) Ejecutar pandas según intención
    try:
        if intent == "shipment_lookup":
            sid = _shipment_id_from_text_or_intent(text, intent_obj)
            if not sid:
                return "Necesito que me digas el shipment id 😉 (ejemplo: 63357)."
            carrier = direct_lookup(text, df)  # intenta por regex en el mensaje
            if not carrier:
                id_col = col(df, R, "SHIPMENT_ID") or "SHIPMENT_ID"
                carrier_col = col(df, R, "CARRIER_NAME") or "CARRIER_NAME"
                if id_col in df.columns and carrier_col in df.columns:
                    hit = df.loc[df[id_col].astype(str).str.strip() == str(sid)]
                    carrier = hit[carrier_col].iloc[0] if not hit.empty else None
            return format_answer("shipment_lookup", {"shipment_id": sid, "carrier": carrier})

        if intent == "pickup_date":
            sid = _shipment_id_from_text_or_intent(text, intent_obj)
            if not sid:
                return "Dime el shipment id y te digo la fecha de pickup 😎."
            id_col = col(df, getattr(df, "__colresolver__", {}) or {}, "SHIPMENT_ID") or "SHIPMENT_ID"
            pickup_col = pick_pickup_date_col(df)
            date_val = None
            if pickup_col and id_col in df.columns and pickup_col in df.columns:
                hit = df.loc[df[id_col].astype(str).str.strip() == str(sid), pickup_col]
                if not hit.empty:
                    date_val = pd.to_datetime(hit.iloc[0], errors="coerce")
                    date_val = date_val.strftime("%Y-%m-%d %H:%M") if pd.notnull(date_val) else None
            return format_answer("pickup_date", {"shipment_id": sid, "date": date_val})


        if intent == "in_transit":
            status_col, mask = in_transit_mask(df)
            if status_col is None:
                return "No encuentro una columna de estatus para identificar 'en tránsito'."
            hits = df[mask]
            if not cols:
                cols = df.columns[:3].tolist()
            rows = []
            for r in hits[cols].head(20).to_dict(orient="records"):
                rows.append(" · ".join([str(r.get(k, "")) for k in cols]))
            return format_answer("in_transit", {"total": int(len(hits)), "rows": rows})

        if intent == "revenue_lookup":
            R = getattr(df, "__colresolver__", {}) or {}
            shipper_col = (col(df, R, "COMPANY_NAME")
                           or col(df, R, "SHIPPER_NAME")
                           or col(df, R, "CUSTOMER_NAME"))
            revenue_col = (col(df, R, "TOTAL_REVENUE") or col(df, R, "FREIGHT_REVENUE") or "TOTAL_REVENUE")
            date_col = pick_pickup_date_col(df)  # 👈 PICKUP
            if not shipper_col or revenue_col not in df.columns:
                return "No encuentro columnas de revenue por cliente 😅. Revisa que exista TOTAL_REVENUE/FREIGHT_REVENUE y COMPANY_NAME/SHIPPER_NAME."
            ent = (intent_obj or {}).get("entity")
            mes_txt = (intent_obj or {}).get("month")
            mm = MONTHS_ES.get((mes_txt or "").lower()) if mes_txt else None
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            mask = df[shipper_col].astype(str).str.contains(ent or "", case=False, na=False)
            if mm and date_col in df.columns:
                mask = mask & (df[date_col].dt.month == mm)
            total = pd.to_numeric(df[revenue_col], errors="coerce")[mask].sum()
            return format_answer("revenue_lookup", {"entity": ent or "ese cliente", "period": mes_txt or "el periodo indicado", "value": float(total)})

        if intent == "carrier_stats":
            carrier_col = col(df, R, "CARRIER_NAME") or "CARRIER_NAME"
            ent = (intent_obj or {}).get("entity")
            if not ent or carrier_col not in df.columns:
                return "Necesito el nombre del carrier para darte sus stats 🚛."
            subset = df[df[carrier_col].astype(str).str.contains(ent, case=False, na=False)]
            n_ship = len(subset)
            revenue_col = col(df, R, "TOTAL_REVENUE")
            cost_col = col(df, R, "TOTAL_COST")
            gp_col = col(df, R, "GROSS_PROFIT")
            total_rev = float(pd.to_numeric(subset.get(revenue_col, pd.Series(dtype=float)), errors="coerce").sum()) if revenue_col else 0.0
            if gp_col and gp_col in subset:
                gp = float(pd.to_numeric(subset[gp_col], errors="coerce").sum())
            elif revenue_col and cost_col and revenue_col in subset and cost_col in subset:
                gp = float(pd.to_numeric(subset[revenue_col], errors="coerce").sum() - pd.to_numeric(subset[cost_col], errors="coerce").sum())
            else:
                gp = 0.0
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
        return fast  # quick_router ya devuelve texto humano con format_answer
    return await answer_with_nlu(q, df)

@app.post("/whatsapp", summary="Whatsapp Webhook")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    message_body = (form.get("Body") or "").strip()

    reply = "Recibido ✅"

    try:
        df = get_df()

        # A) Router rápido
        fast = quick_router(message_body, df)
        if fast is not None:
            return twiml_response(fast)   # quick_router ya devuelve texto bonito

        # B) NLU
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











