# ============================================================
# SEMANA 3 - RAG Colombia Comparte
# Generación con LLM (Qwen2.5-0.5B-Instruct)
# Requiere haber corrido semana1 y semana2 primero
# pip install transformers accelerate
# ============================================================

import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ── CONFIG ──────────────────────────────────────────────────
MODELO_EMBED  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_LLM    = "Qwen/Qwen2.5-0.5B-Instruct"
CHUNKS_JSON   = "data/chunks.json"
INDEX_FAISS   = "data/index.faiss"
TOP_K         = 3
MIN_SCORE     = 0.20

# Parámetros de generación
MAX_NEW_TOKENS = 250
TEMPERATURE    = 0.2
TOP_P          = 0.85
# ────────────────────────────────────────────────────────────


# 1. CARGAR SISTEMA DE RETRIEVAL (semana 2)
def cargar_retrieval():
    print("⏳ Cargando modelo de embeddings...")
    embed = SentenceTransformer(MODELO_EMBED)
    print("⏳ Cargando índice FAISS...")
    index = faiss.read_index(INDEX_FAISS)
    print("⏳ Cargando chunks...")
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Retrieval listo — {len(chunks)} chunks\n")
    return embed, index, chunks


# 2. CARGAR LLM
def cargar_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Cargando LLM: {MODELO_LLM} en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_LLM)
    model = AutoModelForCausalLM.from_pretrained(
        MODELO_LLM,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"✅ LLM listo\n")
    return tokenizer, model, device


# 3. RETRIEVE CONTEXT (copiado de semana 2)
def retrieve_context(query, embed_model, index, chunks):
    if not query.strip():
        return []
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_vec, TOP_K)
    resultados = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        if float(score) < MIN_SCORE:
            continue
        resultados.append({
            "seccion": chunks[idx]["seccion"],
            "texto":   chunks[idx]["texto"],
            "score":   round(float(score), 4)
        })
    return resultados


def formatear_contexto(resultados):
    if not resultados:
        return ""
    return "\n\n---\n\n".join(
        f"[{r['seccion']}]\n{r['texto']}" for r in resultados
    )


# 4. CONSTRUIR PROMPT
def construir_prompt(query, contexto):
    system = (
        "Eres el asistente virtual oficial de Colombia Comparte / Latinoamérica Comparte. "
        "Tu única fuente de información es el contexto proporcionado. "
        "Reglas estrictas:\n"
        "- Responde SOLO con información del contexto. NUNCA inventes datos.\n"
        "- Si la respuesta no está en el contexto, di exactamente: "
        "'No tengo suficiente información para responder esa pregunta con los datos disponibles.'\n"
        "- Responde siempre en español, de forma clara, directa y amable.\n"
        "- No menciones que tienes un 'contexto' o 'documento'; habla naturalmente."
    )
    user = f"Contexto:\n{contexto}\n\nPregunta: {query}"
    return system, user


# 5. GENERAR RESPUESTA
def generate_answer(query, embed_model, index, chunks, tokenizer, model, device):
    # Retrieval
    resultados = retrieve_context(query, embed_model, index, chunks)

    # Fallback si no hay contexto relevante
    if not resultados:
        return (
            "No tengo suficiente información para responder esa pregunta con los datos disponibles.",
            []
        )

    contexto = formatear_contexto(resultados)
    system, user = construir_prompt(query, contexto)

    # Formato chat de Qwen
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    # Tokenizar
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decodificar solo los tokens nuevos
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    respuesta = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Limpiar texto repetido o artefactos
    respuesta = limpiar_respuesta(respuesta, query)

    return respuesta, resultados


# 6. LIMPIAR SALIDA DEL MODELO
def limpiar_respuesta(texto, query):
    # Quitar repeticiones de la pregunta al inicio
    if texto.lower().startswith(query.lower()):
        texto = texto[len(query):].strip()

    # Quitar líneas vacías múltiples
    lineas = [l for l in texto.splitlines() if l.strip()]
    texto = "\n".join(lineas)

    # Control de alucinación: si la respuesta es muy corta o vacía → fallback
    if len(texto) < 10:
        return "No tengo suficiente información para responder esa pregunta con los datos disponibles."

    return texto


# 7. PRUEBAS CONTROLADAS
def ejecutar_pruebas(embed_model, index, chunks, tokenizer, model, device):
    casos = [
        # Preguntas que SÍ están en el contexto
        ("¿Qué es Colombia Comparte?",              True),
        ("¿Cuánto dura el programa ESTRUCTURA?",    True),
        ("¿Cómo me inscribo a Comparte Academia?",  True),
        ("¿Quiénes son los fundadores?",            True),
        ("¿Cuál es el teléfono de contacto?",       True),
        # Preguntas que NO están en el contexto
        ("¿Cuánto cuesta el programa?",             False),
        ("¿Tienen sede en Medellín?",               False),
    ]

    print("=" * 65)
    print("PRUEBAS CONTROLADAS — GENERACIÓN CON LLM")
    print("=" * 65)

    for pregunta, en_contexto in casos:
        tipo = "✅ EN CONTEXTO" if en_contexto else "🚫 FUERA DE CONTEXTO"
        print(f"\n{tipo}")
        print(f"Pregunta: {pregunta}")
        print("─" * 50)

        respuesta, resultados = generate_answer(
            pregunta, embed_model, index, chunks, tokenizer, model, device
        )

        print(f"Respuesta: {respuesta}")
        if resultados:
            print(f"Fuentes: {[r['seccion'] for r in resultados]}")

        # Verificar control de alucinaciones
        if not en_contexto:
            if "no tengo suficiente información" in respuesta.lower():
                print("✅ Fallback correcto — no alucinó")
            else:
                print("⚠️  ATENCIÓN: posible alucinación — revisar prompt")

    print("\n" + "=" * 65)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Cargar todo
    embed_model, index, chunks = cargar_retrieval()
    tokenizer, model, device   = cargar_llm()

    # Demo rápido
    print("─" * 65)
    print("DEMO: Pipeline completo query → retrieval → LLM → respuesta")
    print("─" * 65)
    query_demo = "¿Qué es Comparte Academia y cuánto dura?"
    respuesta, resultados = generate_answer(
        query_demo, embed_model, index, chunks, tokenizer, model, device
    )
    print(f"Query: {query_demo}")
    print(f"\nFuentes usadas:")
    for r in resultados:
        print(f"  [{r['score']}] {r['seccion']}")
    print(f"\nRespuesta:\n{respuesta}\n")

    # Pruebas controladas
    ejecutar_pruebas(embed_model, index, chunks, tokenizer, model, device)

    print("\n🎉 Semana 3 completa.")
    print("   generate_answer() listo para integrar en el chatbot (Semana 4).")