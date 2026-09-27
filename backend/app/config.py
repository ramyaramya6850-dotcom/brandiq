from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
DATABASE_DIR = ROOT / 'database'
for p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, DATABASE_DIR): p.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv('SECRET_KEY', 'CHANGE-ME-IN-PRODUCTION')
DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{DATABASE_DIR / "brandiq.db"}')
CORS_ORIGINS = [x.strip() for x in os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',') if x.strip()]
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '100'))
