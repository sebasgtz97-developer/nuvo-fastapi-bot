import os
import re
import pandas as pd
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
    Pull a pandas expression on df from an LLM response.
    Tries fenced blocks, inline df[...] snippets, and fixes common glitches.
    """
    if not raw:
        return ""

    # 1) fenced code blocks
    blocks = re.findall(r"```(?:python)?\s*([^`]+)```", raw, flags=re.IGNORECASE)
    candidates = [b.strip() for b in blocks]

    # 2) inline df[...] snippets
    candidates += [m.strip() for m in re.findall(r"(df\s*\[[^\n]+)", raw, flags=re.IGNORECASE)]

    # 3) any line mentioning df[
    for line in raw.splitlines():
        if "df[" in line:
            candidates.append(line.strip())

    for c in candidates:
        # remove chatter before df[
        c = re.sub(r"^[^d]*?(?=df\s*\[)", "", c, flags=re.IGNORECASE)
        # strip comments and trailing semicolons
        c = c.split("#")[0].strip().rstrip(";")
        # quick repair: unmatched brackets
        if c.count("[") > c.count("]"):
            c += "]"
        # if the model returned "x = <expr>", keep only the rhs
        if "=" in c and not c.strip().startswith("df["):
            parts = c.split("=", 1)
            if "df[" in parts[1]:
                c = parts[1].strip()
        return c

    return raw.strip()


def safe_eval_df_expr(code: str, df: pd.DataFrame):
    import io
    import contextlib
    from datetime import datetime
    import builtins

    try:
        # Normalize dates if present
        date_cols = ["CREATED_AT_MX", "PICKUP_STARTS_AT", "INVOICED_AT", "ACTUAL_DELIVERED_TO_DESTINATION_"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Normalize numeric money columns if present
        money_cols = ["GROSS_PROFIT", "TOTAL_COST", "TOTAL_REVENUE", "FREIGHT_COST", "FREIGHT_REVENUE"]
        for col in money_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        local_vars = {"df": df, "pd": pd, "datetime": datetime}

        if "\n" in code or ";" in code:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, {"__builtins__": builtins.__dict__}, local_vars)
            # return last "result-like" variable
            for var_name in reversed(list(local_vars.keys())):
                if isinstance(local_vars[var_name], (int, float, str, pd.Series, pd.DataFrame)):
                    return local_vars[var_name]
        return eval(code, {"__builtins__": builtins.__dict__}, local_vars)
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
    return PlainTextResponse(content=xml, media_type="text/xml; charset=utf-8")


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

    # quick default reply so Twilio always gets something fast
    reply = "Recibido ✅"

    try:
        df = get_df()

        # 1) deterministic lookup first
        direct = direct_lookup(message_body, df)
        if direct:
            reply = f"📄 Carrier name: {direct}"
        else:
            # 2) LLM route
            raw_code = interpret_query(message_body, df)
            code = extract_code(raw_code)
            print("🧠 Código generado (WhatsApp):", raw_code, "=>", code)
            result = safe_eval_df_expr(code, df)
            reply = (format_reply_from_result(result) or "").strip() or reply

            # 3) final fallback heuristic
            if reply.startswith("No encontré"):
                alt = fallback_carrier_lookup(message_body, df)
                if alt is not None:
                    reply = (format_reply_from_result(alt) or "").strip() or reply

    except Exception as e:
        print("webhook error:", e)

    return twiml_response(reply)









