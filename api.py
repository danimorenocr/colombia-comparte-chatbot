# ============================================================
# API RAG - Colombia Comparte
# FastAPI — consumible desde Postman o cualquier cliente HTTP
# pip install fastapi uvicorn transformers accelerate sentence-transformers faiss-cpu
#
# Correr: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Docs:   http://localhost:8000/docs
# ============================================================

import json
import torch
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── CONFIG ──────────────────────────────────────────────────
MODELO_EMBED   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_LLM     = "Qwen/Qwen2.5-0.5B-Instruct"
CHUNKS_JSON    = "data/chunks.json"
INDEX_FAISS    = "data/index.faiss"
TOP_K          = 3
MIN_SCORE      = 0.20
MAX_NEW_TOKENS = 250
TEMPERATURE    = 0.2
TOP_P          = 0.85

FALLBACK = "No tengo suficiente información para responder esa pregunta con los datos disponibles."
PALABRAS_RIESGO = ["$", "usd", "cop", "costo", "precio", "vale", "gratis", "gratuito",
                   "@gmail", "@hotmail", "@yahoo", "medellín", "cali", "barranquilla"]
# ────────────────────────────────────────────────────────────


# ── STARTUP: cargar modelos una sola vez ─────────────────────
print("⏳ Iniciando API RAG Colombia Comparte...")

print("  → Cargando embeddings...")
embed_model = SentenceTransformer(MODELO_EMBED)

print("  → Cargando índice FAISS...")
index = faiss.read_index(INDEX_FAISS)

print("  → Cargando chunks...")
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  → Cargando LLM en {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_LLM)
llm = AutoModelForCausalLM.from_pretrained(
    MODELO_LLM,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
if device == "cpu":
    llm = llm.to(device)
llm.eval()

print("✅ API lista\n")
# ─────────────────────────────────────────────────────────────


# ── APP ──────────────────────────────────────────────────────
app = FastAPI(
    title="Chatbot RAG — Colombia Comparte",
    description="API para hacer preguntas sobre Colombia Comparte / Latinoamérica Comparte.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ─────────────────────────────────────────────────────────────


# ── SCHEMAS ──────────────────────────────────────────────────
class PreguntaRequest(BaseModel):
    pregunta: str

    class Config:
        json_schema_extra = {
            "example": {"pregunta": "¿Qué es Colombia Comparte?"}
        }

class FuenteItem(BaseModel):
    seccion: str
    score: float

class RespuestaResponse(BaseModel):
    pregunta: str
    respuesta: str
    fuentes: list[FuenteItem]
    chunks_encontrados: int
# ─────────────────────────────────────────────────────────────


# ── LÓGICA RAG ───────────────────────────────────────────────
def retrieve(query: str):
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_vec, TOP_K)
    resultados = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1 or float(score) < MIN_SCORE:
            continue
        resultados.append({
            "seccion": chunks[idx]["seccion"],
            "texto":   chunks[idx]["texto"],
            "score":   round(float(score), 4),
        })
    return resultados


def formatear_contexto(resultados):
    return "\n\n---\n\n".join(f"[{r['seccion']}]\n{r['texto']}" for r in resultados)


def verificar_alucinacion(respuesta: str, contexto: str) -> bool:
    ctx = contexto.lower()
    for palabra in PALABRAS_RIESGO:
        if palabra in respuesta.lower() and palabra not in ctx:
            return True
    return False


def generar(query: str, contexto: str) -> str:
    system = (
        "Eres el asistente virtual oficial de Colombia Comparte / Latinoamérica Comparte. "
        "Tu única fuente de información es el CONTEXTO proporcionado. "
        "REGLAS ABSOLUTAS:\n"
        "1. PROHIBIDO inventar datos que no estén en el contexto (precios, costos, fechas, ubicaciones).\n"
        "2. Si la información no está en el contexto responde EXACTAMENTE: "
        f"'{FALLBACK}'\n"
        "3. Responde en español, de forma clara y amable.\n"
        "4. No menciones 'contexto', 'documento' ni 'sección'."
    )
    user = (
        f"CONTEXTO:\n{contexto}\n\n"
        f"PREGUNTA: {query}\n\n"
        "Responde basándote SOLO en el CONTEXTO."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    respuesta = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # Limpiar artefactos
    lineas = [l for l in respuesta.splitlines() if l.strip()]
    respuesta = "\n".join(lineas)

    # Anti-alucinación
    if not respuesta or len(respuesta) < 10 or verificar_alucinacion(respuesta, contexto):
        return FALLBACK

    return respuesta
# ─────────────────────────────────────────────────────────────


# ── ENDPOINTS ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "mensaje": "API RAG Colombia Comparte activa",
        "uso": "POST /preguntar con body { 'pregunta': 'tu pregunta aquí' }",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_cargados": len(chunks),
        "vectores_faiss": int(index.ntotal),
        "dispositivo": device,
    }


@app.post("/preguntar", response_model=RespuestaResponse)
def preguntar(body: PreguntaRequest):
    pregunta = body.pregunta.strip()

    if not pregunta:
        raise HTTPException(status_code=400, detail="El campo 'pregunta' no puede estar vacío.")

    if len(pregunta) > 500:
        raise HTTPException(status_code=400, detail="La pregunta no puede superar 500 caracteres.")

    # Retrieval
    resultados = retrieve(pregunta)

    # Sin contexto → fallback directo (sin gastar tokens del LLM)
    if not resultados:
        return RespuestaResponse(
            pregunta=pregunta,
            respuesta=FALLBACK,
            fuentes=[],
            chunks_encontrados=0,
        )

    contexto  = formatear_contexto(resultados)
    respuesta = generar(pregunta, contexto)

    fuentes = [FuenteItem(seccion=r["seccion"], score=r["score"]) for r in resultados]

    return RespuestaResponse(
        pregunta=pregunta,
        respuesta=respuesta,
        fuentes=fuentes,
        chunks_encontrados=len(resultados),
    )
# ─────────────────────────────────────────────────────────────


# ── MAIN (desarrollo local) ───────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)