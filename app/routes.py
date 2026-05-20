from __future__ import annotations

from fastapi import APIRouter, HTTPException

from analytics_dashboard import render_analytics_dashboard
from app.core import GROQ_MODEL, LANGDETECT_AVAILABLE, chunks, index, logger
from app.schemas import PreguntaRequest, RespuestaResponse, FuenteItem
from app.services import (
    actualizar_resumen,
    cargar_historial,
    crear_sesion,
    detectar_idioma,
    es_intencion_negocio,
    formatear_contexto,
    generar,
    guardar_mensaje,
    obtener_analiticas,
    obtener_resumen,
    retrieve,
)

router = APIRouter()


@router.get("/")
def root():
    return {
        "status": "ok",
        "version": "3.1.0 (Groq + Country)",
        "modelo": GROQ_MODEL,
        "docs": "/docs",
    }


@router.get("/analytics")
def analytics():
    return obtener_analiticas()


@router.get("/analytics/dashboard")
def analytics_dashboard():
    data = obtener_analiticas()
    return render_analytics_dashboard(
        leads=data["leads_por_pais"] or [],
        diaria=data["actividad_diaria"] or [],
        faqs=data["faqs"] or [],
    )


@router.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_cargados": len(chunks),
        "vectores_faiss": int(index.ntotal),
        "modelo_llm": GROQ_MODEL,
        "langdetect": LANGDETECT_AVAILABLE,
    }


@router.post("/preguntar", response_model=RespuestaResponse)
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

    if body.action == "initial":
        lang = detectar_idioma("", fallback_lang=body.language) or "es"
        logger.info("Initial message resolved language=%s", lang)
        country_label = body.country or ("Colombia" if lang == "es" else "your country")

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

    if not resultados:
        from app.core import FALLBACK, CTA
        reply = FALLBACK.get(lang, FALLBACK["es"])
        if lead:
            reply += CTA[lang]
        logger.info("Returning fallback response language=%s lead=%s", lang, lead)
        guardar_mensaje(session_id, "user", pregunta, is_lead=lead)
        guardar_mensaje(session_id, "assistant", reply, is_lead=lead)
        history_actualizado = cargar_historial(session_id, ultimos_n=6)
        actualizar_resumen(session_id, history_actualizado, lang)
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

    from app.core import CTA
    if lead and "http" not in respuesta:
        respuesta += CTA[lang]

    logger.info("Returning response language=%s lead=%s fuentes=%s", lang, lead, len(resultados))

    fuentes = [FuenteItem(seccion=r["seccion"], score=r["score"]) for r in resultados]

    guardar_mensaje(session_id, "user", pregunta, is_lead=lead)
    guardar_mensaje(session_id, "assistant", respuesta, is_lead=lead)
    history_actualizado = cargar_historial(session_id, ultimos_n=6)
    actualizar_resumen(session_id, history_actualizado, lang)

    return RespuestaResponse(
        reply=respuesta,
        fuentes=fuentes,
        chunks_encontrados=len(resultados),
        idioma_detectado=lang,
        es_lead=lead,
        session_id=session_id,
    )
