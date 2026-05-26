from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from unittest import result

from app.core import (
    CTA,
    FALLBACK,
    GROQ_MODEL,
    LANGDETECT_AVAILABLE,
    MIN_SCORE,
    PALABRAS_NEGOCIO,
    SUPABASE_KEY,
    SUPABASE_URL,
    TOP_K,
    chunks,
    embed_model,
    groq_client,
    index,
    logger,
    supa,
)

try:
    from langdetect import detect
    from langdetect import LangDetectException
except ImportError:  # pragma: no cover
    detect = None
    LangDetectException = Exception


def crear_sesion(country: str | None, language: str | None) -> str:
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
    if LANGDETECT_AVAILABLE and detect and texto and len(texto.split()) >= 2:
        try:
            detected = detect(texto)
            if detected in ["en", "es"]:
                return detected
        except LangDetectException:
            pass

    if fallback_lang:
        lang_lower = fallback_lang.lower().strip()
        if lang_lower.startswith("en"):
            return "en"
        if lang_lower.startswith("es"):
            return "es"

    return "es"


def es_intencion_negocio(texto: str) -> bool:
    texto_lower = texto.lower()
    return any(palabra in texto_lower for palabra in PALABRAS_NEGOCIO)


def es_fuera_de_contexto(resultados: list, es_negocio: bool) -> bool:
    """
    Detecta si una pregunta está fuera de contexto.
    
    Una pregunta está fuera de contexto si:
    - No tiene resultados relevantes (no encuentra nada en la base de datos)
    - Y NO es una pregunta de negocio/emprendimiento
    
    Si es de negocio, se responde igual aunque no haya contexto.
    """
    return len(resultados) == 0 and not es_negocio


def retrieve(query: str):
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_vec, TOP_K)
    resultados = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1 or float(score) < MIN_SCORE:
            continue
        resultados.append({
            "seccion": chunks[idx]["seccion"],
            "texto": chunks[idx]["texto"],
            "score": round(float(score), 4),
        })
    return resultados


def formatear_contexto(resultados):
    return "\n\n---\n\n".join(f"[{r['seccion']}]\n{r['texto']}" for r in resultados)


def actualizar_resumen(session_id: str, history: list[dict], lang: str):
    if len(history) < 2:
        logger.info("No se actualiza resumen: menos de 2 mensajes en el historial.")
        return

    texto_conv = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in history[-8:]
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

        logger.info("Resumen ✅✅✅✅generado para session=%s: %r", session_id, resumen)

        result = supa.table("sessions").update({"summary": resumen}).eq("id", session_id).execute()

        logger.info("Update ✅✅✅✅✅summary result: data=%s", result.data)
    except Exception as exc:  # pragma: no cover
        logger.warning("No se pudo actualizar resumen: %s", exc)


def obtener_resumen(session_id: str) -> str | None:
    res = supa.table("sessions").select("summary").eq("id", session_id).execute()
    if res.data:
        return res.data[0].get("summary")
    return None


def generar(
    query: str,
    contexto: str,
    lang: str = "es",
    es_lead: bool = False,
    history: list[dict] | None = None,
    country: str | None = None,
    resumen: str | None = None,
) -> str:
    country_ctx_es = f" El usuario se conecta desde {country}." if country else ""
    country_ctx_en = f" The user is connecting from {country}." if country else ""

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
            "5. IMPORTANT: If a question is about topics completely unrelated to entrepreneurship, business, or Latin America (like general knowledge, entertainment, politics, etc.), politely decline and redirect to your expertise.\n"
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
            "5. IMPORTANTE: Si una pregunta trata sobre temas completamente ajenos a emprendimiento, negocios u oportunidades en Latinoamérica (como conocimiento general, entretenimiento, política, etc.), recházala educadamente y redirige hacia tu expertise.\n"
            f"{regla_pais_es}"
            f"{regla_lead_es}"
            f"{resumen_ctx}\nCONTEXTO:\n{contexto}"
        )

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


def obtener_analiticas():
    leads = _fetch_supabase_rest("analytics_leads_por_pais")
    diaria = _fetch_supabase_rest("analytics_actividad_diaria", limit=30)
    faqs = _fetch_supabase_rest("analytics_preguntas_frecuentes")
    return {"leads_por_pais": leads, "actividad_diaria": diaria, "faqs": faqs}


def _fetch_supabase_rest(table_name: str, limit: int | None = None) -> list[dict]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    query = {"select": "*"}
    if limit is not None:
        query["limit"] = str(limit)

    url = f"{SUPABASE_URL}/rest/v1/{table_name}?{urlencode(query)}"
    request = Request(
        url,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    return data if isinstance(data, list) else []
