# ============================================================
# API RAG - Colombia Comparte
# ============================================================

import json
import torch
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# pip install langdetect
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect no instalado. Usando campo 'language' del body como fallback.")

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

FALLBACK = {
    "es": "No tengo suficiente información para responder esa pregunta con los datos disponibles.",
    "en": "I don't have enough information to answer that question with the available data.",
}

PALABRAS_RIESGO = ["$", "usd", "cop", "costo", "precio", "vale", "gratis", "gratuito",
                   "@gmail", "@hotmail", "@yahoo", "medellín", "cali", "barranquilla"]
# ────────────────────────────────────────────────────────────


# ── STARTUP ─────────────────────────────────────────────────
print("⏳ Iniciando API RAG Colombia Comparte...")

embed_model = SentenceTransformer(MODELO_EMBED)
index = faiss.read_index(INDEX_FAISS)

with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
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
    message: str
    country: str | None = None
    language: str | None = "es"   # "es" | "en" — usado si langdetect no está disponible
    action: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is Colombia Comparte?",
                "country": "Colombia",
                "language": "en",
                "action": "chat"
            }
        }


class FuenteItem(BaseModel):
    seccion: str
    score: float


class RespuestaResponse(BaseModel):
    reply: str
    fuentes: list[FuenteItem] = []
    chunks_encontrados: int = 0
    idioma_detectado: str = "es"   # nuevo campo informativo
# ─────────────────────────────────────────────────────────────


# ── DETECCIÓN DE IDIOMA ──────────────────────────────────────
def detectar_idioma(texto: str, fallback_lang: str = "es") -> str:
    """
    Devuelve 'en' o 'es'.
    Prioridad: langdetect sobre el texto → campo language del body → 'es'.
    Solo se distingue entre inglés y español; cualquier otro idioma cae a 'es'.
    """
    if LANGDETECT_AVAILABLE and texto and len(texto.split()) >= 2:
        try:
            detected = detect(texto)
            if detected == "en":
                return "en"
            return "es"
        except LangDetectException:
            pass

    # Fallback: usar el campo language que viene en el body
    if fallback_lang and fallback_lang.lower().startswith("en"):
        return "en"
    return "es"
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


def generar(query: str, contexto: str, lang: str = "es") -> str:
    fallback_msg = FALLBACK.get(lang, FALLBACK["es"])

    if lang == "en":
        system = (
            "You are the official virtual assistant of Colombia Comparte / Latinoamérica Comparte. "
            "Your ONLY source of information is the provided CONTEXT. "
            "ABSOLUTE RULES:\n"
            "1. FORBIDDEN to invent data not present in the context (prices, costs, dates, locations).\n"
            "2. If the information is not in the context, respond EXACTLY: "
            f"'{fallback_msg}'\n"
            "3. Respond in English, clearly and in a friendly tone.\n"
            "4. Do NOT mention 'context', 'document', or 'section'."
        )
        user = (
            f"CONTEXT:\n{contexto}\n\n"
            f"QUESTION: {query}\n\n"
            "Answer based ONLY on the CONTEXT above."
        )
    else:
        system = (
            "Eres el asistente virtual oficial de Colombia Comparte / Latinoamérica Comparte. "
            "Tu única fuente de información es el CONTEXTO proporcionado. "
            "REGLAS ABSOLUTAS:\n"
            "1. PROHIBIDO inventar datos que no estén en el contexto (precios, costos, fechas, ubicaciones).\n"
            "2. Si la información no está en el contexto responde EXACTAMENTE: "
            f"'{fallback_msg}'\n"
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

    lineas = [l for l in respuesta.splitlines() if l.strip()]
    respuesta = "\n".join(lineas)

    if not respuesta or len(respuesta) < 10 or verificar_alucinacion(respuesta, contexto):
        return fallback_msg

    return respuesta
# ─────────────────────────────────────────────────────────────


# ── ENDPOINTS ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "mensaje": "API RAG Colombia Comparte activa",
        "uso": "POST /preguntar con body { 'message': 'tu pregunta aquí' }",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_cargados": len(chunks),
        "vectores_faiss": int(index.ntotal),
        "dispositivo": device,
        "langdetect": LANGDETECT_AVAILABLE,
    }


@app.post("/preguntar", response_model=RespuestaResponse)
def preguntar(body: PreguntaRequest):
    pregunta = body.message.strip()

    if not pregunta and body.action != "initial":
        raise HTTPException(status_code=400, detail="El campo 'message' no puede estar vacío.")

    # Mensaje inicial — respetar el language del body
    if body.action == "initial":
        lang = detectar_idioma("", fallback_lang=body.language or "es")
        if lang == "en":
            saludo = "Hi 👋 I'm the virtual assistant for Colombia Comparte. How can I help you today?"
        else:
            saludo = "Hola 👋 Soy el asistente virtual de Colombia Comparte. ¿En qué puedo ayudarte hoy?"
        return RespuestaResponse(reply=saludo, fuentes=[], chunks_encontrados=0, idioma_detectado=lang)

    if len(pregunta) > 500:
        raise HTTPException(status_code=400, detail="La pregunta no puede superar 500 caracteres.")

    # Detectar idioma: primero analiza el texto, luego usa el campo language como fallback
    lang = detectar_idioma(pregunta, fallback_lang=body.language or "es")

    # Retrieval
    resultados = retrieve(pregunta)

    if not resultados:
        return RespuestaResponse(
            reply=FALLBACK.get(lang, FALLBACK["es"]),
            fuentes=[],
            chunks_encontrados=0,
            idioma_detectado=lang,
        )

    contexto = formatear_contexto(resultados)
    respuesta = generar(pregunta, contexto, lang=lang)

    fuentes = [FuenteItem(seccion=r["seccion"], score=r["score"]) for r in resultados]

    return RespuestaResponse(
        reply=respuesta,
        fuentes=fuentes,
        chunks_encontrados=len(resultados),
        idioma_detectado=lang,
    )
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)