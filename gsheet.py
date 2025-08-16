import os
import json
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

def _authorize():
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        info = json.loads(sa_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        # Fallback to a file path if you really want to ship a file
        key_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
        creds = Credentials.from_service_account_file(key_path, scopes=SCOPES)
    return gspread.authorize(creds)

def load_data():
    """Loads your sheet into a pandas DataFrame."""
    sheet_ref = os.getenv("SHEET_URL", "").strip()
    if not sheet_ref:
        raise RuntimeError("SHEET_URL env var is empty.")
    gc = _authorize()
    sh = gc.open_by_url(sheet_ref) if sheet_ref.startswith("http") else gc.open_by_key(sheet_ref)
    ws = sh.sheet1  # or specify a worksheet name/idx if you prefer
    rows = ws.get_all_records()  # list[dict]
    return pd.DataFrame(rows)


