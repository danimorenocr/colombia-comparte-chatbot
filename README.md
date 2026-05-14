# 🤖 RAG Colombia Comparte

Un sistema completo de **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre el programa **Colombia Comparte** usando búsqueda semántica e IA generativa, con **API REST** consumible.

---

## 📋 Descripción General

Este proyecto implementa un pipeline RAG de 3 etapas + API:

```
┌─────────────────────────────────────────────────────────────┐
│ SEMANA 1: CHUNKING + EMBEDDINGS + FAISS                    │
│ Prepara datos: texto → chunks → embeddings → índice        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SEMANA 2: MOTOR DE RECUPERACIÓN                            │
│ Búsqueda semántica: query → embedding → top-k chunks      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SEMANA 3: GENERACIÓN CON LLM                               │
│ Respuesta: contexto + query → LLM → respuesta en español   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ API REST (FastAPI)                                          │
│ Endpoints: GET / GET /health | POST /preguntar            │
│ CORS habilitado → consumible desde cualquier cliente       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Requisitos

- **Python 3.9+**
- **GPU (opcional pero recomendado para LLM)**

### Dependencias principales:
```
sentence-transformers        # Embeddings multilingües
faiss-cpu / faiss-gpu        # Búsqueda vectorial
transformers                  # Cargar LLMs
torch / torch-cuda            # Backend de IA
fastapi + uvicorn             # API REST
numpy                          # Álgebra lineal
```

---

## 📦 Instalación

### 1. Crear entorno virtual
```bash
python -m venv .venv
.venv\Scripts\activate  # En Windows
# source .venv/bin/activate  # En Linux/Mac
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
# O instalar manualmente:
pip install sentence-transformers faiss-cpu numpy
pip install transformers accelerate torch
```

### 3. Preparar datos
- Coloca tu archivo de datos en: `data/Colombia_Comparte_BASE_RAG_FINAL.txt`
- El formato debe tener secciones: `[SECCIÓN 1 - TEMA]`

---

## 📁 Estructura de Archivos

```
.
├── README.md                              # Este archivo
├── requirements.txt                       # Dependencias
│
├── rag.py                                 # SEMANA 1: Preparación de datos
├── retrieval.py                           # SEMANA 2: Motor de búsqueda
├── llm.py                                 # SEMANA 3: Generación con IA
├── api.py                                 # API REST (FastAPI)
│
├── rag_logs.log                           # Log de ejecución (auto-generado)
├── data/
│   ├── Colombia_Comparte_BASE_RAG_FINAL.txt  # Archivo de entrada (tu dato)
│   ├── chunks.json                           # Chunks en JSON (auto-generado)
│   └── index.faiss                           # Índice FAISS (auto-generado)
└── .venv/                                 # Entorno virtual
```

---

## 🔧 Cómo Usar

### **PASO 1: Ejecutar SEMANA 1 (Preparación)**

Genera chunks, embeddings e índice FAISS:

```bash
python rag.py
```

**Qué hace:**
- ✅ Carga el archivo de texto
- ✅ Divide en chunks por secciones
- ✅ Genera embeddings multilingües
- ✅ Crea índice FAISS para búsqueda rápida
- ✅ Valida búsquedas con ejemplos

**Archivos generados:**
- `data/chunks.json` — Chunks con metadata
- `data/index.faiss` — Índice vectorial

**Logs:** Se guardan en `rag_logs.log`

---

### **PASO 2: Ejecutar SEMANA 2 (Recuperación)**

Prueba el motor de búsqueda semántica:

```bash
python retrieval.py
```

**Qué puedes hacer:**
- Importar `retrieve_context()` en otros scripts
- Buscar documentos relevantes por similitud semántica
- Filtrar resultados por puntuación mínima

**Ejemplo:**
```python
from retrieval import cargar_sistema, retrieve_context

modelo, index, chunks = cargar_sistema()
resultados = retrieve_context("¿Cómo me inscribo?", modelo, index, chunks)

for r in resultados:
    print(f"[{r['score']}] {r['seccion']}")
    print(f"  {r['texto'][:200]}...")
```

---

### **PASO 3: Ejecutar SEMANA 3 (Generación)**

Genera respuestas con LLM usando el contexto recuperado:

```bash
python llm.py
```

**Qué hace:**
- 🤖 Carga modelo de embeddings + índice FAISS + chunks
- 🧠 Carga LLM (Qwen2.5-0.5B-Instruct)
- 💬 Acepta preguntas del usuario
- 🔍 Busca contexto relevante
- 📝 Genera respuestas basadas en contexto

**Ejemplo de interacción:**
```
> ¿Qué es Colombia Comparte?

[Contexto recuperado: 3 chunks relevantes]
[LLM generando respuesta...]

Respuesta: Colombia Comparte es un programa que...
```

---

## 🌐 API REST (FastAPI)

### Iniciar servidor

```bash
# Asegúrate de haber completado SEMANA 1 primero
python rag.py  # Genera index.faiss y chunks.json

# Iniciar API
python api.py
# O con reload automático:
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**API disponible en:**
- Base URL: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs` ← Interfaz interactiva
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

#### 1. **GET /** — Status de la API

```bash
curl http://localhost:8000/
```

**Respuesta:**
```json
{
  "status": "ok",
  "mensaje": "API RAG Colombia Comparte activa",
  "uso": "POST /preguntar con body { 'pregunta': 'tu pregunta aquí' }",
  "docs": "/docs"
}
```

#### 2. **GET /health** — Health check

```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "chunks_cargados": 45,
  "vectores_faiss": 45,
  "dispositivo": "cuda"
}
```

#### 3. **POST /preguntar** — Hacer pregunta (⭐ Principal)

```bash
curl -X POST http://localhost:8000/preguntar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué es Colombia Comparte?"}'
```

**Request:**
```json
{
  "pregunta": "¿Cómo me inscribo al programa EDIFICA?"
}
```

**Response:**
```json
{
  "pregunta": "¿Cómo me inscribo al programa EDIFICA?",
  "respuesta": "Para inscribirse al programa EDIFICA, debe...",
  "fuentes": [
    {
      "seccion": "SECCIÓN 5 - EDIFICA",
      "score": 0.8234
    },
    {
      "seccion": "SECCIÓN 3 - Inscripción",
      "score": 0.7561
    }
  ],
  "chunks_encontrados": 2
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
