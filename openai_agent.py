import os
import openai
from dotenv import load_dotenv

load_dotenv()  # 👈 Necesario para cargar el .env aquí también

print("🔐 API KEY:", os.getenv("OPENAI_API_KEY"))  # solo debug temporal (quítalo en prod)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _build_prompt(message: str, df):
    cols = list(df.columns)

    return f"""
Eres un **analista senior de operaciones y data** que responde en **español**, con tono **amable, conciso y útil**.
Tienes un DataFrame de Pandas llamado `df` con **estas columnas reales**:
{cols}

Tu tarea:
1) **Entender la intención** de la pregunta (estatus, fechas, costos, revenue, margen/GP, on-time, riesgos, carrier, cliente, lane, conteos, filtros por rango de fechas, top N, últimos eventos, comparativos, etc.).
2) **Responder primero** con una **explicación corta y amigable** (1–3 oraciones).
3) **Entregar después** **un ÚNICO bloque de código** (```python ... ```) con **una sola expresión de Pandas** que pueda ejecutarse directamente y que:
   - Use exclusivamente el DataFrame `df`.
   - Devuelva un **resultado final** (Series, DataFrame o escalar) coherente con la pregunta.
   - Sea **una sola línea** o una **única expresión** (sin múltiples pasos, sin variables temporales, sin funciones auxiliares).
   - No imprima ni explique nada: solo la **expresión**.
4) Si la consulta menciona un **Shipment ID / SHIPMENT_NAME** (p. ej. “63357” o “SH-63357”):
   - **Normaliza**: acepta ambas formas; si el DataFrame tiene `SHIPMENT_ID` numérico y viene con “SH-”, quítale el prefijo; si el DataFrame usa `SHIPMENT_NAME` tipo “SH-####”, agrégalo si el usuario mandó solo números.
   - Si no hay coincidencias, dilo amablemente y **la expresión** debe devolver un filtro vacío válido (p. ej., `df[df['SHIPMENT_ID'] == -1]` o condición imposible).

Reglas de oro:
- **Primero** intenta usar columnas exactas presentes en {cols}.
- Si una columna no existe pero hay otra **muy similar** (ej. “CARRIER” vs “CARRIER_NAME”), usa la que exista y **menciónalo** en la explicación.
- Para **últimos eventos/estatus**: si hay columnas como `LAST_EVENT`, `LATEST_STATUS`, `STATUS`, `TRACKING_LAST_EVENT`, úsala(s). Si no existen, aproxima con orden por fecha del evento disponible (ej. `EVENT_DATE`, `ACTUAL_ARRIVED_TO_DESTINATION_`, etc.) y toma el más reciente.
- Para **on-time / performance**: busca columnas como `ON_TIME`, `IS_ONTIME`, `OTD`, `DELIVERY_ON_TIME`, o compara `ACTUAL_ARRIVED_TO_DESTINATION_` vs `APPOINTMENT_AT_DESTINATION_` (si existen).
- Para **costos/revenue/margen**: busca nombres como `FREIGHT_COST`, `TOTAL_COST`, `COST`, `REVENUE`, `BILL_AMOUNT`, `GP`, `GROSS_PROFIT`, `MARGIN`, etc. Si no hay `GP`, calcula `REVENUE - COST` si ambas existen.
- Para **fechas clave** (recogida/origen/cruce/destino): revisa columnas como `ACTUAL_ARRIVED_TO_ORIGIN_`, `ACTUAL_ARRIVED_TO_DESTINATION_`, `PICKUP_AT_ORIGIN_`, `CROSSING_DATE`, `APPOINTMENT_*`.
- Para **carrier/cliente/lane**: usa columnas como `CARRIER_NAME`, `CUSTOMER`, `CLIENT`, `ORIGIN STATE`, `ORIGIN`, `DESTINATION`, `LANE`, `ORIGIN_CITY`, `DESTINATION_CITY`, etc. Ajusta al nombre real existente.
- **Top N**: si piden “top 5 carriers por revenue”, agrupa, ordena desc y limita con `.head(5)`.
- **Rangos de fecha**: si el usuario menciona un mes/rango (ej. “julio 2025”), filtra por la columna de fecha más apropiada existente (idealmente la de “evento relevante” al contexto).
- **Nunca** generes múltiples bloques de código ni comentarios dentro del bloque. Solo **una expresión** ejecutable.
- **No inventes** columnas: si faltan, explica brevemente la aproximación en la respuesta amistosa.
- Mantén **temperature 0** (ya configurado afuera).

Ejemplos de intención (debes cubrir variantes similares):
• Estatus:
  - “¿Cuál es el estatus del shipment 63357 / SH-63357?”
  - “¿Cuál fue el último evento del SH-63357?”
• Fechas y eventos:
  - “¿Cuándo llegó a origen / se entregó / cruzó frontera el 63357?”
• Costos / revenue / margen:
  - “¿Cuánto fue el freight cost / revenue / gross profit del 63357?”
• On-time:
  - “¿Se entregó a tiempo el 63357?”
• Carrier:
  - “¿Quién cubrió el SH-63357?”
• Agregados:
  - “Top 5 carriers por revenue este mes”
  - “Conteo de embarques por cliente en julio”
  - “Revenue total y margen promedio por cliente”
• Últimos estatus de varios:
  - “Dame el último status de los embarques de {{"{cliente}"}} hoy”
  - “Último evento de los shipments cubiertos por {{'Carrier X'}}”
• Filtros de negocio:
  - “Embarques con riesgo abierto”, “entregas tardías de la semana pasada”, “por lane MTY → LAREDO”
• Conteos y auditorías:
  - “¿Cuántos embarques cancelados ayer?”, “¿qué % estuvo on-time?”

**FORMATO DE SALIDA OBLIGATORIO**:
1) Una breve respuesta **amigable** (máx 3 oraciones).
2) Un **único** bloque de código con una **sola expresión** de Pandas sobre `df` que compute la respuesta.

Consulta del usuario:
\"\"\"{message}\"\"\"
""".strip()

def interpret_query(message, df):
    prompt = _build_prompt(message, df)

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",  # rápido y barato; cambia a gpt-5 si quieres más robustez
        messages=[
            {"role": "system", "content": "Eres un analista de datos y operaciones experto. Respondes en español, breve y amable. Después de responder, siempre incluyes un único bloque de código con UNA SOLA expresión de pandas sobre df que produce la respuesta solicitada."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return chat_completion.choices[0].message.content.strip()







