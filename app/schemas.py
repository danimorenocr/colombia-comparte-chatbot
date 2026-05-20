from __future__ import annotations

from pydantic import BaseModel, Field


class PreguntaRequest(BaseModel):
    message: str
    session_id: str | None = None
    country: str | None = None
    language: str | None = None
    action: str | None = None
    history: list[dict] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Tengo un emprendimiento de café artesanal, ¿pueden ayudarme?",
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            }
        }
    }


class FuenteItem(BaseModel):
    seccion: str
    score: float


class RespuestaResponse(BaseModel):
    reply: str
    fuentes: list[FuenteItem] = Field(default_factory=list)
    chunks_encontrados: int = 0
    idioma_detectado: str = "es"
    es_lead: bool = False
    session_id: str | None = None
