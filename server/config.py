import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
CACHE_DIR  = os.path.join(ROOT_DIR, "cache")
PLOTS_DIR  = os.path.join(UPLOAD_DIR, "plots")

for d in (UPLOAD_DIR, CACHE_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)
