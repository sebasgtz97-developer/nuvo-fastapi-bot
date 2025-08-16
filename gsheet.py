import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def load_data():
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name("carrier_ops_service_account.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("14Ijn2MJj9a2lyNEaqpeSIlGSpDoiP013TL0fJNTfAKo").sheet1
    data = pd.DataFrame(sheet.get_all_records())

    # 🔧 Limpieza robusta del SHIPMENT_ID
    data['SHIPMENT_ID'] = data['SHIPMENT_ID'].astype(str).str.extract(r'(\d+)')
    data['SHIPMENT_ID'] = data['SHIPMENT_ID'].str.strip()

    # ➕ Columna auxiliar para SH-XXXXX
    data['SHIPMENT_NAME'] = 'SH-' + data['SHIPMENT_ID']

    return data


