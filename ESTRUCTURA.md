# Estructura de Proyecto - Colombia Comparte Chatbot

## 📁 Layout reorganizado

```
colombia-comparte-chatbot/
├── src/                          # Código Python
│   ├── api.py                    # API FastAPI (RAG)
│   ├── rag.py                    # Procesamiento: chunking + embeddings + FAISS
│   ├── llm.py                    # Generación: LLM (semana 3)
│   └── retrieval.py              # Motor de recuperación (semana 2)
├── data/                         # Datos y modelos indexados
│   ├── Colombia_Comparte_BASE_RAG_v2.txt
│   ├── Colombia_Comparte_RAG_unificado.txt
│   ├── chunks.json               # Chunks procesados
│   └── index.faiss               # Índice FAISS
├── logs/                         # Archivos de log
│   └── rag_logs.log
├── requirements.txt
├── README.md
└── ESTRUCTURA.md                 # Este archivo
```

## 🚀 Ejecución desde raíz

Todos los scripts deben correrse desde **la raíz del proyecto** (`colombia-comparte-chatbot/`):

### 1. Preparar embeddings + índice (ejecutar una sola vez)
```bash
cd colombia-comparte-chatbot
python src/rag.py
```
✅ Genera: `data/chunks.json` y `data/index.faiss`

### 2. Iniciar API RAG
```bash
cd colombia-comparte-chatbot
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
📍 Docs: http://localhost:8000/docs

### 3. Usar motor de retrieval (módulo importable)
```bash
cd colombia-comparte-chatbot
python -c "from src.retrieval import cargar_sistema, retrieve_context; ...usar funciones..."
```

### 4. Prueba LLM + RAG completo
```bash
cd colombia-comparte-chatbot
python src/llm.py
```

## 📝 Notas
- ✅ **Rutas actualizadas**: Todos los archivos `src/*.py` ahora usan `../data/` y `../logs/` automáticamente
- ✅ **Sin cambios en funcionalidad**: Solo reorganización de carpetas + ajuste de rutas relativas
- ✅ **Importaciones**: Si necesitas importar módulos desde `src/`, asegúrate de estar en la raíz del proyecto
