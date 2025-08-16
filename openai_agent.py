import os
import openai
from dotenv import load_dotenv

load_dotenv()  # 👈 Necesario para cargar el .env aquí también

print("🔐 API KEY:", os.getenv("OPENAI_API_KEY"))  # solo debug temporal

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def interpret_query(message, df):
    prompt = f'''
Eres un experto en análisis de datos. Tienes un DataFrame de Pandas con estas columnas:

{list(df.columns)}

El usuario puede hacer preguntas en lenguaje natural sobre un shipment como:

🟢 Estatus del embarque:
- ¿Cuál es el estatus del shipment 63357?
- ¿Ya se entregó el SH-63357?
- ¿En qué etapa va el embarque SH-63357?
- ¿Se canceló el shipment 63357?
- ¿Cuál fue el último evento del SH-63357?

🟩 Fechas y eventos:
- ¿Cuándo llegó a origen el shipment 63357?
- ¿Cuándo se entregó el SH-63357?
- ¿Cuándo cruzó frontera el embarque 63357?
- ¿En qué fecha se recogió de origen el SH-63357?

💸 Costos y revenue:
- ¿Cuál fue el freight cost del SH-63357?
- ¿Cuánto revenue generó el shipment 63357?
- ¿Cuál fue el gross profit de ese embarque?
- ¿Cuál fue el costo total del SH-63357?

⏱️ Performance / On-Time:
- ¿El shipment 63357 fue on time?
- ¿Fue entregado a tiempo el SH-63357?
- ¿Llegó tarde a destino el 63357?
- ¿Tuvo algún riesgo ese embarque?

🚛 Carrier / Transporte:
- ¿Quién cubrió el SH-63357?
- Qué carrier transportó el shipment 63357?
- ¿Quién fue el carrier de ese embarque?

Siempre responde con una expresión de Pandas que devuelva los datos correctos. Usa SHIPMENT_ID o SHIPMENT_NAME según corresponda.

Ejemplo esperado:
df[df['SHIPMENT_ID'] == '63357']['CARRIER_NAME']

Ahora genera la línea de código para esta consulta:
"{message}"
'''.strip()

    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return chat_completion.choices[0].message.content.strip()






