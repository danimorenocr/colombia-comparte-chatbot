from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import create_client

try:
    from langdetect import LangDetectException, detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect no instalado. Usando campo 'language' del body como fallback.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_JSON = DATA_DIR / "chunks.json"
INDEX_FAISS = DATA_DIR / "index.faiss"
MODELO_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
MIN_SCORE = 0.20
GROQ_MODEL = "llama-3.1-8b-instant"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("colombia_comparte_api")

FALLBACK = {
    "es": "No tengo suficiente información para responder esa pregunta con los datos disponibles.",
    "en": "I don't have enough information to answer that question with the available data.",
}

PALABRAS_NEGOCIO = [
    "emprendimiento", "emprender", "negocio", "empresa", "startup", "pyme",
    "producto", "servicio", "vender", "venta", "cliente", "mercado",
    "exportar", "importar", "financiamiento", "inversión", "capital",
    "crecer", "escalar", "socio", "alianza", "proyecto",
    "business", "startup", "venture", "entrepreneur", "product", "service",
    "sell", "sales", "market", "funding", "investment", "grow", "scale",
    "partner", "project", "customer",
]

CTA = {
    "es": (
        "\n\n---\n"
        "🚀 **¿Te gustaría que Colombia Comparte te acompañe en este proceso?**\n"
        "Puedes registrarte o contactarnos directamente para conocer cómo podemos apoyar tu emprendimiento.\n"
        "👉 [Contáctanos aquí](https://colombiacomparte.com/contacto)"
    ),
    "en": (
        "\n\n---\n"
        "🚀 **Would you like Colombia Comparte to support you in this process?**\n"
        "You can register or contact us directly to learn how we can help your business.\n"
        "👉 [Contact us here](https://colombiacomparte.com/contacto)"
    ),
}

load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supa = create_client(SUPABASE_URL, SUPABASE_KEY)

print("⏳ Iniciando API RAG Colombia Comparte v3 (Groq + Country)...")

embed_model = SentenceTransformer(MODELO_EMBED)
index = faiss.read_index(str(INDEX_FAISS))

with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("No se encontró GROQ_API_KEY en .env")

groq_client = Groq(api_key=GROQ_API_KEY)

print("✅ API lista\n")
