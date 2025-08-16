import os
import re
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import Response
from dotenv import load_dotenv
from xml.sax.saxutils import escape

from openai_agent import interpret_query
from gsheet import load_data

load_dotenv()  # loads env vars; do NOT print secrets

app = FastAPI(title="nuvo-fastapi-bot", version="1.0.0")

# ---------- Helpers ----------
def extract_code(raw: str) -> str:
    if not raw:
        return ""
    code_blocks = re.findall(r"```(?:python)?\s*([^`]+)```", raw, flags=re.IGNORECASE)
    if code_blocks:
        candidate = code_blocks[0].strip()
        lines = [l.strip() for l in candidate.splitlines()]
        for line in reversed(lines):
            if 'df[' in line or '.sum()' in line or '.mean()' in line or '.count()' in line:
                if "=" in line:
                    return line.split("=", 1)[1].strip()
                return line.strip()
        return candidate
    for line in raw.splitlines():
        if 'df[' in line:
            if '=' in line:
                return line.split('=', 1)[1].strip()
            return line.strip()
    return raw.strip()


def safe_eval_df_expr(code: str, df: pd.DataFrame):
    import io
    import contextlib
    from datetime import datetime
    import builtins

    try:
        # Convertir fechas si existen
        date_cols = ["CREATED_AT_MX", "PICKUP_STARTS_AT", "INVOICED_AT", "ACTUAL_DELIVERED_TO_DESTINATION_"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convertir números si existen
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
    except Exception as e:
        raise RuntimeError(f"Eval error: {e} | code={code}")


def fallback_carrier_lookup(message: str, df: pd.DataFrame):
    match = re.search(r"(SH-)?(\d{3,})", message.upper())
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


def twiml_response(body_text: str) -> Response:
    safe_text = escape(body_text or "")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{safe_text}</Message>
</Response>"""
    return Response(content=xml, media_type="application/xml")


# ---------- Utility Routes ----------
@app.get("/")
def root():
    return {"service": "nuvo-fastapi-bot", "status": "running"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/send-updates")
def manual_send_updates():
    # calls the same logic your cron used to run
    from send_updates import main as run_updates
    run_updates()
    return {"status": "updates sent"}


# ---------- Business Routes ----------
@app.get("/ask", summary="Ask Gsheet")
def ask_gsheet(q: str):
    df = load_data()
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
    message_body = form.get("Body", "")
    print("📩 Mensaje recibido (WhatsApp):", message_body)

    df = load_data()
    print("📄 Columnas del DataFrame:", df.columns.tolist())

    reply = "No encontré información con ese criterio."
    try:
        raw_code = interpret_query(message_body, df)
        code = extract_code(raw_code)
        print("🧠 Código generado (WhatsApp):", raw_code, "=>", code)

        result = safe_eval_df_expr(code, df)
        print("📦 Resultado eval (WhatsApp):", result)

        reply = format_reply_from_result(result)
        reply = (reply or "").strip()

        if reply == "" or reply.startswith("No encontré"):
            print("🔁 Fallback manual activado...")
            alt_result = fallback_carrier_lookup(message_body, df)
            if alt_result is not None:
                reply = (format_reply_from_result(alt_result) or "").strip()

    except Exception as e:
        print("❌ Error en procesamiento primario:", e)
        alt_result = fallback_carrier_lookup(message_body, df)
        if alt_result is not None:
            reply = (format_reply_from_result(alt_result) or "").strip()
        else:
            reply = f"Error: {str(e)}"

    print("📤 Twilio response preview:\n", twiml_response(reply).body.decode())
    return twiml_response(reply)







