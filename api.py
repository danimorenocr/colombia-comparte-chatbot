# ============================================================
# API RAG - Colombia Comparte (v3 — Groq + Lead Conversion + Country)
# pip install fastapi uvicorn sentence-transformers faiss-cpu groq langdetect
# ============================================================
from dotenv import load_dotenv
import os
import json
import logging
from datetime import datetime, timezone
import faiss
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from analytics_dashboard import render_analytics_dashboard

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect no instalado. Usando campo 'language' del body como fallback.")

# ── CONFIG ──────────────────────────────────────────────────
load_dotenv()

MODELO_EMBED  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS_JSON   = "data/chunks.json"
INDEX_FAISS   = "data/index.faiss"
TOP_K         = 3
MIN_SCORE     = 0.20
GROQ_MODEL    = "llama-3.1-8b-instant"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s - %(message)s'
)
logger = logging.getLogger("colombia_comparte_api")

FALLBACK = {
    "es": "No tengo suficiente información para responder esa pregunta con los datos disponibles.",
    "en": "I don't have enough information to answer that question with the available data.",
}

PALABRAS_NEGOCIO = [
    # es
    "emprendimiento", "emprender", "negocio", "empresa", "startup", "pyme",
    "producto", "servicio", "vender", "venta", "cliente", "mercado",
    "exportar", "importar", "financiamiento", "inversión", "capital",
    "crecer", "escalar", "socio", "alianza", "proyecto",
    # en
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


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supa = create_client(SUPABASE_URL, SUPABASE_KEY)
# ────────────────────────────────────────────────────────────


# ── STARTUP ─────────────────────────────────────────────────
print("⏳ Iniciando API RAG Colombia Comparte v3 (Groq + Country)...")

embed_model = SentenceTransformer(MODELO_EMBED)
index = faiss.read_index(INDEX_FAISS)

with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("No se encontró GROQ_API_KEY en .env")

groq_client = Groq(api_key=GROQ_API_KEY)

print("✅ API lista\n")
# ─────────────────────────────────────────────────────────────


# ── APP ──────────────────────────────────────────────────────
app = FastAPI(
    title="Chatbot RAG — Colombia Comparte",
    description="API para hacer preguntas sobre Colombia Comparte / Latinoamérica Comparte.",
    version="3.1.0",
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
    session_id: str | None = None
    country: str | None = None
    language: str | None = None
    action: str | None = None
    history: list[dict] | None = []

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Tengo un emprendimiento de café artesanal, ¿pueden ayudarme?",
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": []
            }
        }


class FuenteItem(BaseModel):
    seccion: str
    score: float


class RespuestaResponse(BaseModel):
    reply: str
    fuentes: list[FuenteItem] = []
    chunks_encontrados: int = 0
    idioma_detectado: str = "es"
    es_lead: bool = False
    session_id: str | None = None
# ─────────────────────────────────────────────────────────────


# ── HELPERS ──────────────────────────────────────────────────
def crear_sesion(country: str | None, language: str | None) -> str:
    """Crea una sesión nueva y retorna el session_id."""
    logger.info("Creating session country=%s language=%s", country, language)
    res = supa.table("sessions").insert({
        "country": country,
        "language": language or "es",
    }).execute()
    return res.data[0]["id"]


def guardar_mensaje(session_id: str, role: str, content: str, is_lead: bool = False):
    supa.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "is_lead": is_lead,
    }).execute()
    supa.table("sessions").update({
        "last_active": datetime.now(timezone.utc).isoformat(),
    }).eq("id", session_id).execute()


def cargar_historial(session_id: str, ultimos_n: int = 6) -> list[dict]:
    """Trae los últimos N mensajes como lista {role, content}."""
    res = (
        supa.table("messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(ultimos_n)
        .execute()
    )
    return list(reversed(res.data))


def detectar_idioma(texto: str, fallback_lang: str | None = None) -> str:
    # 1. Autodetectar primero desde el contenido real del mensaje
    if LANGDETECT_AVAILABLE and texto and len(texto.split()) >= 2:
        try:
            detected = detect(texto)
            if detected in ["en", "es"]:
                return detected
        except LangDetectException:
            pass

    # 2. Si la autodetección falla, usar preferencia explícita del usuario si existe
    if fallback_lang:
        lang_lower = fallback_lang.lower().strip()
        if lang_lower.startswith("en"):
            return "en"
        if lang_lower.startswith("es"):
            return "es"

    # 3. Fallback final
    return "es"


def es_intencion_negocio(texto: str) -> bool:
    texto_lower = texto.lower()
    return any(palabra in texto_lower for palabra in PALABRAS_NEGOCIO)


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

def actualizar_resumen(session_id: str, history: list[dict], lang: str):
    """Actualiza el resumen cada 4+ mensajes (cada 2 rondas)."""
    if len(history) < 4:
        print("No se actualiza resumen: menos de 4 mensajes en el historial.")
        return

    texto_conv = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in history[-8:]  # últimas 4 rondas
    )
    prompt = (
        f"Resume en máximo 3 oraciones lo que el usuario quiere o necesita, "
        f"basándote en esta conversación. Idioma: {'español' if lang == 'es' else 'English'}.\n\n"
        f"{texto_conv}"
    )
    try:
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.2,
        )
        resumen = res.choices[0].message.content.strip()
        supa.table("sessions").update(
            {"summary": resumen}
        ).eq("id", session_id).execute()
    except Exception as e:
        print("Error al actualizar resumen:", e)
        logger.warning("No se pudo actualizar resumen: %s", e)

def obtener_resumen(session_id: str) -> str | None:
    res = supa.table("sessions").select("summary").eq("id", session_id).execute()
    if res.data:
        return res.data[0].get("summary")
    return None
# ─────────────────────────────────────────────────────────────


# ── GENERACIÓN CON GROQ ──────────────────────────────────────
def generar(
    query: str,
    contexto: str,
    lang: str = "es",
    es_lead: bool = False,
    history: list[dict] | None = None,
    country: str | None = None,
    resumen: str | None = None,
) -> str:

    # Contexto de país para el prompt
    country_ctx_es = (
        f" El usuario se conecta desde {country}."
        if country else ""
    )
    country_ctx_en = (
        f" The user is connecting from {country}."
        if country else ""
    )

    regla_pais_es = (
        f"5. PAÍS DEL USUARIO:{country_ctx_es} Si existen regulaciones, trámites, "
        f"entidades de apoyo, ejemplos o referencias relevantes para ese país "
        f"(especialmente en el contexto de emprendimiento y comercio en América Latina), "
        f"mencionarlos cuando aplique. Si no hay información específica del país en el contexto, "
        f"responde con la información general disponible.\n"
    ) if country else ""

    regla_pais_en = (
        f"5. USER'S COUNTRY:{country_ctx_en} If there are relevant regulations, procedures, "
        f"support entities, examples or references for that country "
        f"(especially in the context of entrepreneurship and trade in Latin America), "
        f"mention them when applicable. If no country-specific information is in the context, "
        f"respond with the general information available.\n"
    ) if country else ""

    regla_lead_es = (
        "6. El usuario tiene un emprendimiento o necesidad de negocio. "
        "Identifica cómo los servicios de Colombia Comparte pueden ayudarle específicamente. "
        "Sé concreto y guíalo hacia una acción (registrarse, contactar, postularse).\n"
    ) if es_lead else ""

    regla_lead_en = (
        "6. The user has a business or entrepreneurship need. "
        "Identify how Colombia Comparte's services can help them specifically. "
        "Be concrete and guide them toward taking action (registering, contacting, applying).\n"
    ) if es_lead else ""

    resumen_ctx = ""
    if resumen:
        resumen_ctx = (
            f"\n\nCONTEXTO PREVIO DEL USUARIO (resumen de la conversación anterior):\n{resumen}\n"
            if lang == "es" else
            f"\n\nPREVIOUS USER CONTEXT (summary of earlier conversation):\n{resumen}\n"
        )

    if lang == "en":
        system = (
            "You are the official virtual assistant of Colombia Comparte / Latinoamérica Comparte, "
            "a platform that connects entrepreneurs and businesses with opportunities, allies, and resources across Latin America.\n\n"
            "YOUR ONLY information source is the CONTEXT provided below.\n\n"
            "RULES:\n"
            "1. Never invent data (prices, dates, locations, contact info).\n"
            "2. If information is not in the context, say you don't have that data and suggest contacting the team.\n"
            "3. Respond in English, warmly and concisely (max 3 paragraphs).\n"
            "4. Do NOT use the words 'context', 'document', or 'section'.\n"
            f"{regla_pais_en}"
            f"{regla_lead_en}"
            f"{resumen_ctx}\nCONTEXT:\n{contexto}"
        )
    else:
        system = (
            "Eres el asistente virtual oficial de Colombia Comparte / Latinoamérica Comparte, "
            "una plataforma que conecta emprendedores y empresas con oportunidades, aliados y recursos en América Latina.\n\n"
            "TU ÚNICA fuente de información es el CONTEXTO proporcionado.\n\n"
            "REGLAS:\n"
            "1. Nunca inventes datos (precios, fechas, ubicaciones, contactos).\n"
            "2. Si la información no está en el contexto, dilo y sugiere contactar al equipo.\n"
            "3. Responde en español, de forma cálida y concisa (máximo 3 párrafos).\n"
            "4. NO uses las palabras 'contexto', 'documento' ni 'sección'.\n"
            f"{regla_pais_es}"
            f"{regla_lead_es}"
            f"{resumen_ctx}\nCONTEXTO:\n{contexto}"
        )

    # Historial conversacional (últimas 3 rondas)
    messages = [{"role": "system", "content": system}]
    if history:
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0.3,
        top_p=0.85,
    )

    return response.choices[0].message.content.strip()
# ─────────────────────────────────────────────────────────────


# ── ENDPOINTS ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "3.1.0 (Groq + Country)",
        "modelo": GROQ_MODEL,
        "docs": "/docs",
    }

@app.get("/analytics")
def analytics():
    leads   = supa.table("analytics_leads_por_pais").select("*").execute().data
    diaria  = supa.table("analytics_actividad_diaria").select("*").limit(30).execute().data
    faqs    = supa.table("analytics_preguntas_frecuentes").select("*").execute().data
    return {"leads_por_pais": leads, "actividad_diaria": diaria, "faqs": faqs}


@app.get("/analytics/dashboard")
def analytics_dashboard():
    data = analytics()
    return render_analytics_dashboard(
        leads=data["leads_por_pais"] or [],
        diaria=data["actividad_diaria"] or [],
        faqs=data["faqs"] or [],
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_cargados": len(chunks),
        "vectores_faiss": int(index.ntotal),
        "modelo_llm": GROQ_MODEL,
        "langdetect": LANGDETECT_AVAILABLE,
    }


@app.post("/preguntar", response_model=RespuestaResponse)
def preguntar(body: PreguntaRequest):
    pregunta = body.message.strip()

    logger.info(
        "Incoming request action=%s country=%s language=%s session_id=%s history_len=%s message=%r",
        body.action,
        body.country,
        body.language,
        body.session_id,
        len(body.history or []),
        pregunta,
    )

    if not pregunta and body.action != "initial":
        raise HTTPException(status_code=400, detail="El campo 'message' no puede estar vacío.")

    session_id = body.session_id
    if not session_id:
        session_id = crear_sesion(body.country, body.language)

    # ── Saludo inicial personalizado por país ──
    if body.action == "initial":
        lang = detectar_idioma("", fallback_lang=body.language) or "es"
        logger.info("Initial message resolved language=%s", lang)
        country_label = body.country or ("Colombia" if lang == "es" else "your country")

        # Nombre de la plataforma según el país
        colombia_countries = {"colombia"}
        if (body.country or "").strip().lower() in colombia_countries:
            platform_name = "Colombia Comparte"
        else:
            platform_name = f"{country_label} Comparte / Latinoamérica Comparte"

        if lang == "en":
            saludo = (
                f"Hi 👋 I'm the virtual assistant for {platform_name}. "
                f"I see you're connecting from {country_label} — "
                f"tell me about your business or ask me anything!"
            )
        else:
            saludo = (
                f"Hola 👋 Soy el asistente de {platform_name}. "
                f"Veo que te conectas desde {country_label} — "
                f"¡cuéntame sobre tu emprendimiento o hazme cualquier pregunta!"
            )

        guardar_mensaje(session_id, "assistant", saludo)
        return RespuestaResponse(reply=saludo, idioma_detectado=lang, session_id=session_id)

    if len(pregunta) > 500:
        raise HTTPException(status_code=400, detail="La pregunta no puede superar 500 caracteres.")

    lang = detectar_idioma(pregunta, fallback_lang=body.language)
    lead = es_intencion_negocio(pregunta)
    logger.info("Detected language=%s lead=%s", lang, lead)

    resultados = retrieve(pregunta)
    logger.info("Retrieve results count=%s", len(resultados))
    history = cargar_historial(session_id, ultimos_n=6)
    resumen = obtener_resumen(session_id)
    actualizar_resumen(session_id, history, lang)
    

    if not resultados:
        reply = FALLBACK.get(lang, FALLBACK["es"])
        if lead:
            reply += CTA[lang]
        logger.info("Returning fallback response language=%s lead=%s", lang, lead)
        guardar_mensaje(session_id, "user", pregunta, is_lead=lead)
        guardar_mensaje(session_id, "assistant", reply, is_lead=lead)
        return RespuestaResponse(reply=reply, idioma_detectado=lang, es_lead=lead, session_id=session_id)

    contexto = formatear_contexto(resultados)

    try:
        respuesta = generar(
            query=pregunta,
            contexto=contexto,
            lang=lang,
            es_lead=lead,
            history=history,
            country=body.country,
            resumen=resumen,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al llamar a Groq: {str(e)}")

    # CTA si es lead y la respuesta no incluye ya un enlace
    if lead and "http" not in respuesta:
        respuesta += CTA[lang]

    logger.info("Returning response language=%s lead=%s fuentes=%s", lang, lead, len(resultados))

    fuentes = [FuenteItem(seccion=r["seccion"], score=r["score"]) for r in resultados]

    guardar_mensaje(session_id, "user", pregunta, is_lead=lead)
    guardar_mensaje(session_id, "assistant", respuesta, is_lead=lead)

    return RespuestaResponse(
        reply=respuesta,
        fuentes=fuentes,
        chunks_encontrados=len(resultados),
        idioma_detectado=lang,
        es_lead=lead,
        session_id=session_id,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)