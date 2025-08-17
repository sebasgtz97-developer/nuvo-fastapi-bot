import os
import json
import re
import unicodedata
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
    m = re.match(r"```(?:json)?\s*(.+?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s

def _json_or_default(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(_strip_code_fences(text))
    except Exception:
        try:
            brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
            return json.loads(brace.group(0)) if brace else default
        except Exception:
            return default

def _norm_token(s: str) -> str:
    """lowercase + sin acentos + solo [a-z0-9]"""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())

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
# Resolver de columnas por sinónimos (CATÁLOGO COMPLETO)
# ---------------------------

def build_column_resolver(df) -> Dict[str, str]:
    """
    Mapea 'claves lógicas' → nombre real de columna del df, usando sinónimos
    en español/inglés como los pediría un humano por WhatsApp.
    """
    aliases = {
        # Identificadores
        "SHIPMENT_ID": ["shipment id", "id", "embarque", "embarque id", "folio", "guia", "shipmentid"],
        "SHIPMENT_NAME": ["shipment name", "nombre embarque", "nombre carga", "sh-", "sh_"],
        "LANE_ID": ["lane id", "id de ruta", "ruta id", "laneid"],
        "COMPANY_ID": ["company id", "id cliente", "id compañía", "id shipper", "account id"],
        "COMPANY_NAME": ["company", "cliente", "customer", "shipper", "compañía", "account", "consignor", "nombre cliente"],
        "CUSTOMER_REFERENCE": ["customer reference", "referencia cliente", "ref cliente", "customer ref"],

        # Rutas / ciudades
        "CITY_TO_CITY_ROUTE": ["ruta ciudad", "origen-destino", "city to city", "ruta"],
        "ORIGIN_CITY": ["ciudad origen", "origen ciudad"],
        "ORIGIN_STATE": ["estado origen"],
        "ORIGIN_COUNTRY": ["país origen"],
        "DESTINATION_CITY": ["ciudad destino", "destino ciudad"],
        "DESTINATION_STATE": ["estado destino"],
        "DESTINATION_COUNTRY": ["país destino"],

        # Estatus
        "SHIPMENT_STATUS": ["estatus", "status", "estado embarque", "shipment status"],
        "CANEL_OR_REJECT_ACCOUNTABLE": ["cancelado", "rechazado", "cancel or reject", "responsable cancelación"],

        # Fechas generales
        "CREATED_AT_MX": ["fecha creación", "created at", "creado"],
        "PICKUP_STARTS_AT": ["pickup", "fecha de recolección", "ready for pickup", "pickup starts", "rfp", "pickup at"],
        "FIRST_LEG_BOOKED_AT": ["first leg booked", "primera pierna agendada"],

        # Carrier / transporte
        "CARRIER_ID_": ["carrier id", "id transportista", "id proveedor"],
        "CARRIER_NAME": ["carrier", "transportista", "proveedor", "compañía carrier", "nombre carrier"],
        "TRAILER_NUMBER_": ["trailer", "remolque", "número de caja", "trailer number"],

        # Citas
        "APPOINTMENT_TYPE_ORIGIN": ["tipo cita origen", "appointment type origin"],
        "APPOINTMENT_TIME_AT_ORIGIN_LOCAL_TIME": ["cita origen", "hora cita origen", "appointment origin"],
        "APPOINTMENT_TYPE_DESTINATION": ["tipo cita destino", "appointment type destination"],
        "APPOINTMENT_TIME_AT_DESTINATION_LOCAL_TIME": ["cita destino", "hora cita destino", "appointment destination"],

        # Eventos reales (timestamps)
        "LATEST_ACTUAL_EVENT": ["último evento", "last event"],
        "ACTUAL_ARRIVED_TO_ORIGIN_": ["arribo origen", "arrived origin"],
        "ACTUAL_PICKED_UP_FROM_ORIGIN_": ["pickup real", "salida origen", "picked up origin"],
        "ACTUAL_ARRIVED_TO_BORDER_": ["arribo frontera", "arrived border"],
        "ACTUAL_RECEIVED_PAPERWORK_": ["recibió papeles", "paperwork"],
        "ACTUAL_CROSSED_BORDER_": ["cruzó frontera", "crossed border"],
        "ACTUAL_PICKED_UP_FROM_BORDER_": ["pickup en frontera", "picked up border"],
        "ACTUAL_ARRIVED_TO_DESTINATION_": ["arribo destino", "arrived destination"],
        "ACTUAL_DELIVERED_TO_DESTINATION_": ["entrega destino", "delivered", "pod"],

        # Performance y tiempos
        "ROLLED_OVER_BY": ["rollover por", "cambio por", "modificado por"],
        "ON_TIME_AT_ORIGIN_TABLE": ["on time origen (tabla)", "puntual origen tabla"],
        "ON_TIME_AT_DESTINATION_TABLE": ["on time destino (tabla)", "puntual destino tabla"],
        "LOADING_TIME": ["loading time", "tiempo carga"],
        "TRANSIT_TIME_CAL": ["transit time", "tiempo tránsito", "duración viaje"],
        "TIME_TO_BORDER_CROSSING": ["tiempo a cruce de frontera", "border crossing time"],
        "DAYS_TO_INVOICE": ["días a factura", "days to invoice"],

        # Costos / Revenue
        "FREIGHT_COST": ["costo flete", "freight cost", "costo transporte"],
        "CUSTOMS_SERVICE_COST": ["costo aduana", "costo despacho", "customs cost"],
        "ADDITIONAL_COST": ["costo adicional", "extra cost"],
        "INSURANCE_COST": ["costo seguro", "insurance cost"],
        "FREIGHT_REVENUE": ["ingreso flete", "freight revenue", "revenue transporte"],
        "CUSTOMS_SERVICE_REVENUE": ["ingreso aduana", "customs revenue"],
        "ADDITIONAL_REVENUE": ["ingreso adicional", "extra revenue"],
        "INSURANCE_REVENUE": ["ingreso seguro", "insurance revenue"],
        "TOTAL_COST": ["costo total", "total cost"],
        "TOTAL_REVENUE": ["revenue total", "total revenue", "ingresos"],
        "GROSS_PROFIT": ["utilidad", "margen", "gp", "ganancia bruta", "gross profit"],
        "CLIENT_RATE": ["tarifa cliente", "rate", "precio cliente"],

        # KPIs, estados y desempeño
        "CREATED_VS_PICKUP": ["creación vs pickup", "tiempo a pickup"],
        "TRANSIT_TIME_PERFORMANCE": ["desempeño tránsito", "transit performance"],
        "CANCELLED_AT": ["cancelado en", "cancelled at"],
        "INVOICED_AT": ["facturado en", "fecha factura", "invoiced"],
        "CURRENT_TRANSIT_TIME": ["tiempo tránsito actual", "current transit"],
        "HOURS_TO_PICKUP": ["horas a pickup", "hours to pickup"],
        "HORS_TO_DELAPPT": ["horas a cita de entrega", "hours to delivery appt"],
        "LATE_FOR_PICKUP": ["tarde en pickup", "late pickup"],
        "LATE_TRANSIT": ["late transit", "tránsito tardío"],
        "LATE_FOR_DELIVERY": ["tarde en entrega", "late delivery"],
        "AT_RISK_FOR_PICKUP": ["en riesgo pickup", "risk pickup"],
        "AT_RISK_AT_DESTINATION": ["en riesgo destino", "risk destination"],
        "ON_TIME_AT_ORIGIN": ["on time origen", "puntual origen"],
        "ON_TIME_TO_DESTINATION": ["on time destino", "puntual destino"],

        # Cambios de citas y razones
        "PICKUP_APPOINTMENT_CHANGES": ["cambios cita pickup", "pickup changes"],
        "PICKUP_APPOINTMENT_CHANGED_BY": ["pickup cambiado por", "pickup changed by"],
        "PICKUP_APPOINTMENT_CHANGE_REASON": ["motivo cambio pickup"],

        "DELIVERY_APPOINTMENT_CHANGES": ["cambios cita entrega", "delivery changes"],
        "DELIVERY_APPOINTMENT_CHANGED_BY": ["delivery cambiado por", "delivery changed by"],
        "DELIVERY_APPOINTMENT_CHANGE_REASON": ["motivo cambio entrega"],

        # Datos del camión
        "TRUCK_PLATES": ["placas", "matrícula", "truck plates"],
        "TRUCK_NUMBER": ["camión", "unidad", "truck number"],
        "TRUCK_STATE": ["estado camión", "truck state"],

        # Variantes de arribo a destino (CST / TZ)
        "ACTUAL_ARRIVED_TO_DESTINATION_CST": ["arribo destino cst", "arrived destination cst"],
        "ACTUAL_ARRIVED_TO_DESTINATION_TIMEZONE_FROM_APPT": ["arribo destino tz de cita", "arrived destination timezone from appt"],

        # Semanas (agrupaciones)
        "WEEK_OF_READY_FOR_PICKUP": ["semana ready for pickup", "semana rfp"],
        "WEEK_OF_ARRIVED_TO_ORIGIN": ["semana arribo origen"],
        "WEEK_OF_PICKUP_FROM_ORIGIN": ["semana pickup origen"],
        "WEEK_OF_ARRIVED_TO_DESTINATION": ["semana arribo destino"],
        "WEEK_OF_DELIVERED_TO_DESTINATION": ["semana entregado destino"],

        # Trailer / yard
        "TRAILER_LOADED_AT_MX": ["trailer cargado mx", "trailer loaded mx"],
        "LAODING_TIMES_DROP": ["loading times drop", "tiempos carga drop"],  # (typo en fuente)
        "WEEK_OF_TRAILER_LOADED_AT": ["semana trailer loaded"],

        # Stops / citas intermedias
        "DROPOFF_STOP_ONE": ["dropoff stop 1", "bajada 1", "entrega 1"],
        "DROPOFF_STOP_TWO": ["dropoff stop 2", "bajada 2", "entrega 2"],
        "STOP_1_APPOINTMENT": ["cita stop 1"],
        "ARRIVED_TO_STOP_1": ["arribo stop 1"],
        "STOP_2_APPOINTMENT": ["cita stop 2"],
        "ARRIVED_TO_STOP_2": ["arribo stop 2"],
        "STOP_1_DEPARTED_AT": ["salida stop 1", "departed stop 1"],
        "STOP_2_DEPARTED_AT": ["salida stop 2", "departed stop 2"],

        # Direcciones y facilities
        "ADDRESS_LINE_1": ["dirección 1", "address line 1", "domicilio"],
        "PICKUP_FACILITY_NAME_": ["facility pickup", "nombre bodega pickup", "pickup facility"],
        "DROPOFF_FACILITY_NAME": ["facility destino", "nombre bodega destino", "dropoff facility"],
        "DROPOFF_FACILITY_ADDRESS": ["dirección facility destino"],

        # POD / validaciones
        "LAST_POD_RECEIVED": ["último pod recibido", "pod recibido"],
        "LAST_POD_RECEIVED_AT": ["último pod recibido en", "pod recibido fecha"],
        "LAST_POD_VALIDATED": ["último pod validado", "pod validado"],
        "LAST_POD_VALIDATED_AT": ["último pod validado en", "pod validado fecha"],

        # Incidentes / rollovers
        "INCIDENTS": ["incidentes", "incidents"],
        "INCIDENT_DETAILS": ["detalle incidente", "incident details"],
        "DELIVERY_ROLLOVER_CHANGE_REASON": ["motivo rollover entrega"],
        "DELIVERY_ROLLOVER_COUNT": ["conteo rollover entrega"],
        "DELIVERY_ROLLOVER_STATUS": ["estatus rollover entrega"],

        # GPS / enlaces
        "ORIGIN_GPS_LINK": ["gps origen", "link gps origen"],
        "ORIGIN_GPS_LINK_ADDED_A": ["gps origen agregado en"],

        # Yard / tiempos de trailer
        "TRAILER_UNLOADED_ACTUAL_DATETIME": ["descarga trailer real"],
        "TRAILER_LOADED_ACTUAL_DATETIME": ["carga trailer real"],
        "EMPTY_TRAILER_PICKED_UP_ACTUAL_DATETIME": ["recolección caja vacía real"],
        "TRAILER_DROPPED_ACTUAL_DATETIME": ["trailer drop real"],

        # Códigos postales
        "ORIGIN_ZIP": ["zip origen", "código postal origen"],
        "ZIP_DROPOFF_STOP_ONE": ["zip dropoff 1", "cp dropoff 1"],
        "ZIP_DROPOFF_STOP_TWO": ["zip dropoff 2", "cp dropoff 2"],
        "ZIP_DROPOFF_STOP_THREE": ["zip dropoff 3", "cp dropoff 3"],
        "DESTINATION_ZIP": ["zip destino", "código postal destino"],

        # On-time por stops/tolerancias
        "ON_TIME_TO_STOP_1": ["on time stop 1", "puntual stop 1"],
        "ON_TIME_TO_STOP_2": ["on time stop 2", "puntual stop 2"],
        "ON_TIME_AT_DESTINATION_15_TOLERANCE": ["on time 15 destino", "tolerancia 15 destino"],
        "ON_TIME_AT_DESTINATION_60_TOLERANCE": ["on time 60 destino", "tolerancia 60 destino"],
        "ON_TIME_AT_DESTINATION_DAY": ["on time destino día", "entrega mismo día"],
        "ON_TIME_AT_ORIGIN_DAY": ["on time origen día"],

        # Llegadas / salidas por stop
        "ARRIVED_AT_STOP_1": ["arribo stop 1"],
        "DEPARTED_AT_STOP_1": ["salida stop 1"],
        "ARRIVED_AT_STOP_2": ["arribo stop 2"],
        "DEPARTED_AT_STOP_2": ["salida stop 2"],
        "ARRIVED_AT_STOP_3": ["arribo stop 3"],
        "DEPARTED_AT_STOP_3": ["salida stop 3"],

        # Pickup / drop extra
        "PICKUP_STOP_ONE": ["pickup stop 1"],
        "PICKUP_STOP_TWO": ["pickup stop 2"],
        "PICKUP_STOP_THREE": ["pickup stop 3"],
        "DROPOFF_STOP_THREE": ["dropoff stop 3"],

        # Ciudades/estados por stop
        "CITY_DROPOFF_STOP_ONE": ["ciudad dropoff 1"],
        "STATE_DROPOFF_STOP_ONE": ["estado dropoff 1"],
        "CITY_DROPOFF_STOP_TWO": ["ciudad dropoff 2"],
        "STATE_DROPOFF_STOP_TWO": ["estado dropoff 2"],
        "CITY_DROPOFF_STOP_THREE": ["ciudad dropoff 3"],
        "STATE_DROPOFF_STOP_THREE": ["estado dropoff 3"],

        # Tarifas publicadas y vencimiento
        "RATE_PUBLISHED_PRICE_CENTS": ["precio publicado centavos", "published price"],
        "RATE_EXPIRES_AT": ["tarifa expira", "rate expires at"],

        # Documentación / facturas
        "INVOICE_UPLOADED_AT_": ["subida factura", "invoice uploaded"],

        # Tipo de carga / merge
        "TRUCKLOAD_TYPE": ["truckload type", "tipo carga", "tipo embarque"],
        "IS_MERGE_SHIPMENT": ["embarque combinado", "merge shipment"],
    }

    # normaliza columnas reales
    norm_map = {_norm_token(c): c for c in list(df.columns)}

    resolved = {}
    for canonical, syns in aliases.items():
        best = None
        for syn in syns:
            n = _norm_token(syn)
            if n in norm_map:
                best = norm_map[n]; break
            for nk, real in norm_map.items():
                if n in nk:
                    best = real; break
            if best:
                break
        if best:
            resolved[canonical] = best
    return resolved

def col(df, resolver: Dict[str, str], canonical: str, fallback=None) -> Optional[str]:
    """Devuelve el nombre real de columna para una clave canónica."""
    if resolver and canonical in resolver:
        return resolver[canonical]
    if fallback:
        for c in fallback:
            if c in df.columns:
                return c
    return None

# ---------------------------
# 1) NLU: interpretar intención
# ---------------------------

def interpret_intent(message: str, df) -> Dict[str, Any]:
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
            mes = payload.get("period")
            val = payload.get("value")
            if val is None:
                return f"No pude calcular el revenue de {ent} en {mes}."
            return f"💵 {ent} generó un revenue de *${val:,.2f}* en {mes}."

        if intent == "carrier_stats":
            carrier = payload.get("carrier")
            stats = payload.get("stats")
            return f"📊 Estadísticas de *{carrier}*: {stats}."

        text = payload.get("text")
        if text:
            return text
        return "No estoy seguro de lo que pides. Puedes preguntar, por ejemplo: “¿Quién cubrió el shipment id 63357?” o “Revenue de Porcelana Corona en julio”."
    except Exception:
        return "Hubo un problema al armar la respuesta. Intenta reformular o pide un ejemplo."

# ---------------------------
# 3) (Legacy) interpret_query — compat
# ---------------------------

def interpret_query(message: str, df) -> str:
    return ""








