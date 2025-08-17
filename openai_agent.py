import os
import json
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # lee OPENAI_API_KEY del entorno
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Utilidades
# ---------------------------

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # ```json ... ```
    m = re.match(r"```(?:json)?\s*(.+?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s

def _json_or_default(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(_strip_code_fences(text))
    except Exception:
        # intento extra: busca el primer {...} válido
        try:
            brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
            return json.loads(brace.group(0)) if brace else default
        except Exception:
            return default

# Meses ES → número
_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

def month_number_es(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    return _MONTHS_ES.get(name.strip().lower())

# ---------------------------
# 1) NLU: interpretar intención
# ---------------------------

def interpret_intent(message: str, df) -> Dict[str, Any]:
    """
    Devuelve un dict con:
      - intent: shipment_lookup | revenue_lookup | carrier_stats | pickup_date | in_transit | generic
      - entity: texto (shipment id, cliente/shipper o carrier) o null
      - entity_kind: shipment_id | shipper | carrier | none
      - month: nombre del mes en español si aplica (ej. 'julio') o null
      - year: entero o null
      - metric: revenue | gross_profit | cost | count | status | date | none
      - top_n: entero si piden 'top N' (opcional)
    """
    cols = list(df.columns)
    sys = (
        "Eres un analista NLU para logística. "
        "Tu tarea es extraer la intención y entidades de preguntas en español sobre embarques. "
        "Devuelve SOLO un JSON válido, sin texto adicional."
    )
    user = f"""
Pregunta: "{message}"

Estas son columnas reales del DataFrame (para darte contexto): {cols}

Enumera así:
- intent: una de [shipment_lookup, revenue_lookup, carrier_stats, pickup_date, in_transit, generic]
- entity: si hay shipment id (número o 'SH-####'), o un nombre de shipper/cliente/carrier
- entity_kind: una de [shipment_id, shipper, carrier, none]
- month: nombre del mes en español si se menciona (ej. "julio"), si no null
- year: año numérico si se menciona, si no null
- metric: una de [revenue, gross_profit, cost, count, status, date, none]
- top_n: entero si se pide top N, si no null

Responde SOLO con JSON.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )

    default = {
        "intent": "generic",
        "entity": None,
        "entity_kind": "none",
        "month": None,
        "year": None,
        "metric": "none",
        "top_n": None,
    }
    return _json_or_default(resp.choices[0].message.content, default)

# ---------------------------
# 2) Formateo humano de la respuesta
# ---------------------------

def format_answer(intent: str, payload: Dict[str, Any]) -> str:
    """
    Convierte un resultado crudo (de pandas) en texto humano y amable.
    Espera llaves en payload según el intent.
    """
    try:
        if intent == "shipment_lookup":
            sid = payload.get("shipment_id")
            carrier = payload.get("carrier")
            if carrier:
                return f"🚚 El shipment {sid} fue cubierto por *{carrier}*."
            return f"No encontré el carrier para el shipment {sid}."

        if intent == "pickup_date":
            sid = payload.get("shipment_id")
            dt = payload.get("date")
            return f"📅 El pickup date del shipment {sid} es {dt}." if dt else f"No encontré el pickup date del shipment {sid}."

        if intent == "in_transit":
            total = payload.get("total", 0)
            rows = payload.get("rows")
            if isinstance(rows, list) and rows:
                head = "\n".join([f"• {r}" for r in rows[:5]])
                more = "" if len(rows) <= 5 else f"\n… y {len(rows)-5} más."
                return f"🚛 Hay {total} shipments en tránsito. Aquí algunos:\n{head}{more}"
            return f"🚛 Hay {total} shipments en tránsito."

        if intent == "revenue_lookup":
            ent = payload.get("entity")
            mes = payload.get("period")  # texto “julio”, etc.
            val = payload.get("value")
            if val is None:
                return f"No pude calcular el revenue de {ent} en {mes}."
            return f"💵 {ent} generó un revenue de *${val:,.2f}* en {mes}."

        if intent == "carrier_stats":
            carrier = payload.get("carrier")
            stats = payload.get("stats")
            return f"📊 Estadísticas de *{carrier}*: {stats}."

        # genérico / fallback
        text = payload.get("text")
        if text:
            return text
        return "No estoy seguro de lo que pides. Puedes preguntar, por ejemplo: “¿Quién cubrió el shipment id 63357?” o “Revenue de Porcelana Corona en julio”."
    except Exception:
        return "Hubo un problema al armar la respuesta. Intenta reformular o pide un ejemplo."

# ---------------------------
# 3) (Legacy) interpret_query
#     — mantenido por compatibilidad, pero ya NO genera pandas.
#     — si tu main.py aún lo llama, devolverá una cadena vacía.
# ---------------------------

def interpret_query(message: str, df) -> str:
    """
    Obsoleto en el nuevo flujo (NLU + pandas + format_answer).
    Se deja por compatibilidad para que no rompa imports existentes.
    """
    return ""







