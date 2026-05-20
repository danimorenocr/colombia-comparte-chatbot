# Colombia Comparte Backend

Backend oficial de **Colombia Comparte / Latinoamérica Comparte** construido con **FastAPI**, **Groq**, **FAISS** y **Supabase**.

Este servicio expone una API RAG para responder preguntas sobre el programa, guardar sesiones y mensajes, detectar intención de negocio, resumir conversaciones y mostrar analíticas operativas desde el propio backend.

---

## Visión General

La arquitectura resuelve cuatro cosas principales:

1. Recupera contenido relevante desde un índice FAISS construido a partir de los documentos base.
2. Genera respuestas con Groq usando únicamente el contexto recuperado.
3. Persiste conversaciones en Supabase para trazabilidad, sesiones y resúmenes.
4. Expone analíticas y un dashboard HTML rápido para consumo interno.

Flujo simplificado:

```text
Usuario -> POST /preguntar
        -> detecta idioma e intención
        -> busca chunks relevantes en FAISS
        -> recupera resumen/historial desde Supabase
        -> genera respuesta con Groq
        -> guarda mensaje, lead y sesión
        -> devuelve respuesta + fuentes
```

---

## Características

- Respuestas RAG con contexto verificado.
- Soporte multilenguaje: español e inglés.
- Detección de idioma con `langdetect` y fallback por campo del request.
- Detección de intención de negocio para activar CTA.
- Manejo de sesiones y mensajes en Supabase.
- Resumen conversacional persistido por sesión.
- Endpoint de analíticas en JSON.
- Dashboard rápido en HTML desde el backend.
- CORS habilitado para consumo desde frontend o herramientas externas.

---

## Stack Técnico

### Core

- **FastAPI** para la API.
- **Uvicorn** como servidor ASGI.
- **Groq** para la generación de respuestas.
- **Sentence Transformers** para embeddings multilingües.
- **FAISS** para búsqueda vectorial.
- **Supabase** para persistencia de sesiones, mensajes y analíticas.

### Librerías clave

- `python-dotenv`
- `pydantic`
- `langdetect`
- `numpy`

---

## Estructura del Proyecto

```text
Intento 3/
├── api.py
├── analytics_dashboard.py
├── app/
│   ├── __init__.py
│   ├── core.py
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   └── services.py
├── requirements.txt
├── README.md
├── .env
├── data/
│   ├── chunks.json
│   └── index.faiss
└── .venv/
```

### Archivos principales

- `api.py`: wrapper mínimo para seguir ejecutando con `uvicorn api:app`.
- `app/main.py`: crea la aplicación FastAPI y registra rutas.
- `app/routes.py`: endpoints del chatbot y analíticas.
- `app/core.py`: configuración, clientes y recursos compartidos.
- `app/schemas.py`: modelos Pydantic.
- `app/services.py`: lógica de negocio RAG, sesiones y analíticas.
- `analytics_dashboard.py`: renderiza el dashboard HTML.
- `data/chunks.json`: chunks recuperables para el motor RAG.
- `data/index.faiss`: índice vectorial de FAISS.

---

## Requisitos

- Python 3.10 o superior.
- Entorno virtual recomendado.
- Acceso a Supabase.
- API key de Groq.

Opcional:

- `langdetect` instalado para mejor detección de idioma.

---

## Variables de Entorno

Crea un archivo `.env` en la raíz del backend con estas variables:

```env
SUPABASE_URL=tu_url_de_supabase
SUPABASE_KEY=tu_clave_de_supabase
GROQ_API_KEY=tu_groq_api_key
```

### Notas

- `SUPABASE_URL` y `SUPABASE_KEY` son obligatorias para sesiones, mensajes y analíticas.
- `GROQ_API_KEY` es obligatoria para generar respuestas.
- Si falta `GROQ_API_KEY`, la aplicación falla al iniciar.

---

## Instalación

### 1. Crear y activar entorno virtual

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

En Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Verificar archivos de datos

Asegúrate de tener:

- `data/chunks.json`
- `data/index.faiss`

Si no existen, la API no podrá responder porque no tendrá base RAG para recuperar contexto.

---

## Ejecución Local

### Opción recomendada

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Alternativa

```bash
python api.py
```

### URLs útiles

- API raíz: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Analíticas JSON: `http://localhost:8000/analytics`
- Dashboard HTML: `http://localhost:8000/analytics/dashboard`

---

## Endpoints

## `GET /`

Estado básico de la API.

### Ejemplo

```bash
curl http://localhost:8000/
```

### Respuesta esperada

```json
{
  "status": "ok",
  "version": "3.1.0 (Groq + Country)",
  "modelo": "llama-3.1-8b-instant",
  "docs": "/docs"
}
```

---

## `GET /health`

Health check técnico para validar carga de datos y motor.

### Respuesta incluye

- cantidad de chunks cargados
- total de vectores en FAISS
- nombre del modelo LLM
- estado de disponibilidad de `langdetect`

---

## `POST /preguntar`

Endpoint principal del chatbot.

### Body

```json
{
  "message": "Tengo un emprendimiento de café, ¿me pueden ayudar?",
  "session_id": null,
  "country": "Colombia",
  "language": "es",
  "action": "chat",
  "history": []
}
```

### Campos

- `message`: pregunta del usuario.
- `session_id`: identificador de sesión existente, opcional.
- `country`: país desde el que se conecta el usuario.
- `language`: preferencia de idioma del usuario.
- `action`: puede ser `chat` o `initial`.
- `history`: historial opcional de conversación.

### Comportamiento

- Si `action = "initial"`, devuelve un saludo personalizado.
- Si `message` supera 500 caracteres, devuelve `400`.
- Si no hay resultados RAG, responde con fallback y CTA si detecta negocio.
- Si sí hay resultados, llama a Groq con contexto + historial + resumen.
- Guarda mensajes en Supabase.

### Respuesta

```json
{
  "reply": "...",
  "fuentes": [
    {
      "seccion": "...",
      "score": 0.82
    }
  ],
  "chunks_encontrados": 3,
  "idioma_detectado": "es",
  "es_lead": true,
  "session_id": "..."
}
```

---

## `GET /analytics`

Devuelve las analíticas en JSON desde Supabase.

### Respuesta

```json
{
  "leads_por_pais": [],
  "actividad_diaria": [],
  "faqs": []
}
```

### Tablas consultadas

- `analytics_leads_por_pais`
- `analytics_actividad_diaria`
- `analytics_preguntas_frecuentes`

---

## `GET /analytics/dashboard`

Dashboard HTML renderizado desde el backend.

### Incluye

- KPIs principales.
- Gráfico de leads por país.
- Gráfico de actividad diaria.
- Tabla de FAQs más frecuentes.

---

## Lógica del Backend

### Recuperación semántica

El backend usa `SentenceTransformer` para convertir la pregunta del usuario en embedding, luego consulta el índice FAISS y conserva los top-k resultados con score suficiente.

### Generación con Groq

La función `generar()` arma un system prompt que:

- prohíbe inventar datos,
- fuerza la respuesta en español o inglés,
- integra contexto de país si existe,
- integra reglas de lead si detecta intención de negocio,
- agrega resumen previo de la conversación si existe,
- añade el contexto RAG al final.

### Memoria conversacional

- Las sesiones se crean en Supabase.
- Cada mensaje se guarda en `messages`.
- El endpoint intenta leer un resumen previo desde `sessions.summary`.
- También actualiza el resumen periódicamente para mantener continuidad.

---

## Esquema de Datos Esperado en Supabase

### `sessions`

Campos usados por el backend:

- `id`
- `country`
- `language`
- `summary`
- `last_active`

### `messages`

Campos usados por el backend:

- `session_id`
- `role`
- `content`
- `is_lead`
- `created_at`

### Vistas / tablas de analíticas

El dashboard espera algo equivalente a esto:

- `analytics_leads_por_pais`
  - `country`
  - `total_sesiones`
  - `sesiones_con_lead`
  - `pct_leads`
- `analytics_actividad_diaria`
  - `dia`
  - `total_mensajes`
  - `sesiones_unicas`
- `analytics_preguntas_frecuentes`
  - `pregunta`
  - `veces`
  - `ultima_vez`

Si tus columnas cambian, el dashboard ya intenta tolerar varios nombres, pero lo ideal es mantener este contrato.

---

## Ejemplos de Uso

### Saludo inicial

```json
{
  "message": "",
  "country": "Colombia",
  "language": "es",
  "action": "initial",
  "history": []
}
```

### Consulta normal

```json
{
  "message": "Tengo una pyme de café artesanal, ¿cómo me pueden orientar?",
  "country": "Colombia",
  "language": "es",
  "action": "chat",
  "history": []
}
```

### Consulta en inglés

```json
{
  "message": "I want to grow my startup and need guidance",
  "country": "Chile",
  "language": "en",
  "action": "chat",
  "history": []
}
```

---

## Desarrollo

### Ejecutar con autoreload

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Ver logs

La aplicación escribe en consola eventos como:

- creación de sesión,
- idioma detectado,
- condición de lead,
- cantidad de resultados recuperados,
- retorno de fallback,
- errores de Groq o Supabase.

---

## Resolución de Problemas

### `ValueError: No se encontró GROQ_API_KEY en .env`

Falta la variable `GROQ_API_KEY` o el archivo `.env` no está en la raíz del backend.

### La API no arranca y dice que no encuentra `chunks.json` o `index.faiss`

Faltan los archivos generados por el pipeline RAG. Debes reconstruirlos antes de iniciar el backend.

### `IndentationError` o errores de importación

Normalmente indican un archivo editado con mala indentación. En este proyecto ya se separó el dashboard en `analytics_dashboard.py` para evitar eso.

### El dashboard muestra ceros o `N/D`

Eso suele significar que la tabla o vista de Supabase tiene nombres de columnas distintos a los que espera el backend.

### Supabase devuelve datos vacíos

Revisa:

- credenciales en `.env`,
- permisos de las tablas/vistas,
- nombres reales de columnas,
- RLS si está activo.

---

## Buenas Prácticas Recomendadas

- Mantener el índice FAISS sincronizado con `chunks.json`.
- No cambiar nombres de columnas en Supabase sin ajustar el backend.
- Guardar las claves en `.env`, nunca en el código.
- Revisar los prompts si cambias el dominio o el tono del asistente.
- Si el dataset crece, evaluar paginación o caché para analíticas.

---

## Roadmap Sugerido

- Filtros por país y rango de fechas en el dashboard.
- Exportación CSV/Excel desde `/analytics/dashboard`.
- Autenticación simple para proteger las analíticas.
- Endpoint para reconstruir el índice FAISS.
- Métricas de calidad de respuesta y tasa de lead.
- Cacheo de consultas frecuentes a Supabase.

---

## Licencia

Este proyecto pertenece a Colombia Comparte / Latinoamérica Comparte. Ajusta esta sección según el uso final o la política de tu organización.

---

## Autor

Backend RAG para Colombia Comparte.

Si quieres, el siguiente paso es dejar este README con un tono más corporativo o más académico según la entrega final.
}
```

### Características de la API

✅ **Anti-alucinación:** Valida que las respuestas se basen en el contexto  
✅ **Fallback inteligente:** Si no hay contexto, responde con mensaje genérico  
✅ **CORS habilitado:** Consumible desde frontend/apps  
✅ **Validación:** Max 500 caracteres por pregunta  
✅ **Carga lenta:** Los modelos se cargan una sola vez al iniciar  

### Usar desde Postman

1. Abre Postman
2. **New Request** → POST
3. URL: `http://localhost:8000/preguntar`
4. Tab **Body** → raw → JSON
5. Ingresa:
   ```json
   {
     "pregunta": "¿Cuál es la misión de Colombia Comparte?"
   }
   ```
6. Click **Send**

### Usar desde JavaScript/Frontend

```javascript
const pregunta = "¿Qué es Comparte Academia?";

fetch('http://localhost:8000/preguntar', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ pregunta })
})
  .then(r => r.json())
  .then(data => {
    console.log(data.respuesta);
    console.log('Fuentes:', data.fuentes);
  });
```

---

## ⚙️ Configuración

Puedes ajustar estos parámetros en cada archivo:

### `rag.py`
```python
TOP_K = 3                    # Cuántos chunks devolver por búsqueda
MODELO_EMBED = "..."         # Modelo de embeddings
```

### `retrieval.py`
```python
MIN_SCORE = 0.20             # Score mínimo para considerar resultado relevante
TOP_K = 3                    # Número de chunks a recuperar
```

### `llm.py`
```python
MAX_NEW_TOKENS = 250         # Máximo tokens en respuesta
TEMPERATURE = 0.2            # Creatividad (0=determinístico, 1=creativo)
TOP_P = 0.85                 # Nucleus sampling
```

### `api.py`
```python
TOP_K = 3                    # Chunks a recuperar por pregunta
MIN_SCORE = 0.20             # Score mínimo de similitud
MAX_NEW_TOKENS = 250         # Máximo tokens en respuesta
TEMPERATURE = 0.2            # Creatividad
TOP_P = 0.85                 # Nucleus sampling
FALLBACK = "No tengo suficiente..."  # Mensaje si no hay contexto
PALABRAS_RIESGO = [...]      # Palabras que trigguean anti-alucinación
```

---

## 📊 Modelos Utilizados

| Componente | Modelo | Tamaño | Idioma |
|-----------|--------|--------|--------|
| Embeddings | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384-dim | 50+ idiomas |
| LLM | Qwen/Qwen2.5-0.5B-Instruct | 0.5B parámetros | Español, Inglés, Chino |

---

## 📊 Logs y Debugging

Todos los procesos generan logs detallados:

```bash
# Ver logs en tiempo real
tail -f rag_logs.log

# O abrir el archivo directamente
cat rag_logs.log
```

Los logs incluyen:
- ✅ Operaciones exitosas
- ⏳ Procesos en progreso
- ❌ Errores con detalles
- 🔍 Debug info (con level DEBUG)

---

## 🐛 Troubleshooting

### Error: "Archivo no encontrado"
```
❌ Error: Archivo no encontrado en data/Colombia_Comparte_BASE_RAG_FINAL.txt
```
**Solución:** Asegúrate que el archivo existe en la carpeta `data/`

### Error: "CUDA out of memory"
```
CUDA out of memory
```
**Solución:** Cambia en `llm.py`:
```python
torch_dtype=torch.float32  # En lugar de float16
```

### Error: "Module not found"
```
ModuleNotFoundError: No module named 'transformers'
```
**Solución:**
```bash
pip install transformers torch sentence-transformers faiss-cpu
```

---

## 🎯 Flujo Completo (Ejemplo)

```bash
# 1. Activar entorno
.venv\Scripts\activate

# 2. SEMANA 1: Preparar datos
python rag.py
# → Genera: chunks.json, index.faiss, rag_logs.log

# 3. SEMANA 2: Probar búsqueda (opcional)
python retrieval.py
# → Verifica que la búsqueda funcione

# 4. SEMANA 3: Conversar con IA (terminal interactiva)
python llm.py
# → Ingresa preguntas y recibe respuestas con contexto

# 5. ALTERNATIVA: Usar API REST (mejor para producción)
python api.py
# → API disponible en http://localhost:8000/docs

# 6. Ver logs
type rag_logs.log
```

**Opción recomendada:** Usar **API** para integrar en aplicaciones  
**Opción para desarrollo:** Usar **llm.py** para debugging

---

## 📈 Rendimiento

- **Embeddings:** ~0.5s por 100 chunks
- **Búsqueda:** <50ms por query
- **Generación LLM:** 2-5s por respuesta (depende de GPU)

---

## 🤝 Contribuciones

Este es un proyecto educativo. Si encuentras issues o mejoras, siéntete libre de modificar los scripts.

---

## 📝 Licencia

Uso interno - Proyecto Universidad

---

## 📞 Contacto

Para preguntas sobre Colombia Comparte, consulta: www.colombiacomparte.org

---

**Última actualización:** Mayo 2026  
**Versión:** 4.0 (API REST + SEMANA 3 - RAG Completo)
