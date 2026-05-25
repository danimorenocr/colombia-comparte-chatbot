from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

try:
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None

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

OUT_OF_CONTEXT = {
    "es": (
        "Lo siento, esa pregunta está fuera del alcance de Colombia Comparte. "
        "Soy un asistente especializado en emprendimiento, negocios y oportunidades en Latinoamérica. "
        "¿Hay algo sobre tu negocio o emprendimiento que pueda ayudarte? 🚀"
    ),
    "en": (
        "Sorry, that question is outside the scope of Colombia Comparte. "
        "I'm a specialized assistant for entrepreneurship, business, and opportunities in Latin America. "
        "Is there anything about your business or entrepreneurship that I can help you with? 🚀"
    ),
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


@dataclass
class _InMemoryExecuteResult:
    data: list[dict[str, Any]]


class _InMemoryTableQuery:
    def __init__(self, client: "_InMemorySupabaseClient", table_name: str):
        self.client = client
        self.table_name = table_name
        self._operation = "select"
        self._payload: dict[str, Any] | None = None
        self._selected_columns: str | None = None
        self._filters: list[tuple[str, Any]] = []
        self._order_field: str | None = None
        self._order_desc = False
        self._limit: int | None = None

    def insert(self, payload: dict[str, Any]):
        self._operation = "insert"
        self._payload = payload
        return self

    def select(self, columns: str):
        self._operation = "select"
        self._selected_columns = columns
        return self

    def update(self, payload: dict[str, Any]):
        self._operation = "update"
        self._payload = payload
        return self

    def eq(self, field: str, value: Any):
        self._filters.append((field, value))
        return self

    def order(self, field: str, desc: bool = False):
        self._order_field = field
        self._order_desc = desc
        return self

    def limit(self, amount: int):
        self._limit = amount
        return self

    def execute(self):
        rows = self.client._tables.setdefault(self.table_name, [])

        if self._operation == "insert":
            row = dict(self._payload or {})
            row.setdefault("id", str(uuid4()))
            row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            rows.append(row)
            return _InMemoryExecuteResult(data=[row])

        matched = [row for row in rows if all(row.get(field) == value for field, value in self._filters)]

        if self._operation == "update":
            payload = self._payload or {}
            updated_rows = []
            for row in rows:
                if all(row.get(field) == value for field, value in self._filters):
                    row.update(payload)
                    updated_rows.append(row)
            return _InMemoryExecuteResult(data=updated_rows)

        if self._order_field:
            matched = sorted(matched, key=lambda row: row.get(self._order_field), reverse=self._order_desc)

        if self._limit is not None:
            matched = matched[: self._limit]

        if self._selected_columns and self._selected_columns != "*":
            columns = [column.strip() for column in self._selected_columns.split(",")]
            matched = [{column: row.get(column) for column in columns} for row in matched]

        return _InMemoryExecuteResult(data=matched)


class _InMemorySupabaseClient:
    def __init__(self):
        self._tables: dict[str, list[dict[str, Any]]] = {}

    def table(self, table_name: str):
        return _InMemoryTableQuery(self, table_name)

load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if create_client and SUPABASE_URL and SUPABASE_KEY:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase no disponible; usando almacenamiento en memoria temporal.")
    supa = _InMemorySupabaseClient()

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
