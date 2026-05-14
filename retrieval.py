# ============================================================
# SEMANA 2 v2 - RAG Colombia Comparte
# Motor de Recuperación — evaluador corregido + MIN_SCORE ajustado
# ============================================================

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── CONFIG ──────────────────────────────────────────────────
MODELO_EMBED  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS_JSON   = "data/chunks.json"
INDEX_FAISS   = "data/index.faiss"
TOP_K         = 3
MIN_SCORE     = 0.20   # más bajo para no filtrar preguntas de contacto/datos
# ────────────────────────────────────────────────────────────


# 1. CARGAR SISTEMA
def cargar_sistema():
    print("⏳ Cargando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_EMBED)
    print("⏳ Cargando índice FAISS...")
    index = faiss.read_index(INDEX_FAISS)
    print("⏳ Cargando chunks...")
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Sistema listo — {index.ntotal} vectores | {len(chunks)} chunks\n")
    return modelo, index, chunks


# 2. RETRIEVE CONTEXT
def retrieve_context(query, modelo, index, chunks, top_k=TOP_K, min_score=MIN_SCORE):
    if not query.strip():
        return []
    q_vec = modelo.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_vec, top_k)
    resultados = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        if float(score) < min_score:
            continue
        resultados.append({
            "seccion": chunks[idx]["seccion"],
            "texto":   chunks[idx]["texto"],
            "score":   round(float(score), 4)
        })
    return resultados


# 3. FORMATEAR CONTEXTO PARA EL LLM
def formatear_contexto(resultados):
    if not resultados:
        return ""
    partes = []
    for r in resultados:
        partes.append(f"[{r['seccion']}]\n{r['texto']}")
    return "\n\n---\n\n".join(partes)


# 4. EVALUACIÓN CORREGIDA
# La lógica real de RAG: lo que importa es que ALGUNO de los top_k chunks
# contenga la info necesaria, no que el chunk #1 sea exactamente el esperado.
def evaluar_retrieval(modelo, index, chunks):
    casos = [
        # (pregunta, palabras clave que deben aparecer en ALGUNO de los chunks recuperados, es_irrelevante)
        ("¿Qué es Colombia Comparte?",               ["colombia comparte", "organización", "autosostenible"],  False),
        ("¿Cómo me inscribo a Comparte Academia?",   ["inscri", "formulario", "edifica"],                     False),
        ("¿Cuántas personas han sido impactadas?",   ["1.200", "impactado", "trayectoria"],                   False),
        ("¿Qué hace el programa NODUS?",             ["nodus", "liderazgo", "comparte liderazgo"],            False),
        ("¿Qué es la pobreza oculta?",               ["pobreza oculta", "invisible", "crisis"],               False),
        ("¿Quiénes fundaron la organización?",       ["carolina", "eduardo", "fundador"],                     False),
        ("¿Qué eventos organiza Colombia Comparte?", ["red que transforma", "evento", "2025"],                False),
        ("¿Cómo puedo donar?",                       ["donacion", "donación", "bancolombia"],                 False),
        ("¿Qué es DESKUBRE?",                        ["deskubre", "1 mes", "exploración"],                   False),
        ("¿Cuál es el teléfono de contacto?",        ["316", "3087", "teléfono"],                            False),
        # Preguntas irrelevantes — deben retornar vacío o score muy bajo
        ("¿Cuánto cuesta una pizza?",                [],                                                      True),
        ("¿Cómo funciona un motor de avión?",        [],                                                      True),
    ]

    print("=" * 65)
    print("EVALUACIÓN DE CALIDAD DE RECUPERACIÓN (criterio real RAG)")
    print("=" * 65)

    correctos = 0

    for pregunta, keywords, es_irrelevante in casos:
        resultados = retrieve_context(pregunta, modelo, index, chunks)
        texto_total = " ".join(r["texto"].lower() for r in resultados)
        secciones   = [r["seccion"] for r in resultados]
        scores_str  = [r["score"] for r in resultados]

        if es_irrelevante:
            # Correcto si no devuelve nada o todos los scores son muy bajos
            max_score = max((r["score"] for r in resultados), default=0)
            if not resultados or max_score < 0.40:
                print(f"\n🔍 {pregunta}")
                print(f"   ✅ CORRECTO — rechazado (max score {max_score})")
                correctos += 1
            else:
                print(f"\n🔍 {pregunta}")
                print(f"   ⚠️  FALSO POSITIVO — score {max_score} → {secciones[0]}")
        else:
            # Correcto si alguna keyword aparece en el contexto combinado
            encontrado = any(kw in texto_total for kw in keywords)
            if encontrado:
                print(f"\n🔍 {pregunta}")
                print(f"   ✅ CORRECTO — info encontrada en: {secciones}")
                print(f"   scores: {scores_str}")
                correctos += 1
            else:
                print(f"\n🔍 {pregunta}")
                print(f"   ❌ FALLO — keywords {keywords} no encontradas")
                print(f"   Chunks devueltos: {secciones}")
                print(f"   scores: {scores_str}")

    total = len(casos)
    pct   = round(correctos / total * 100, 1)
    print("\n" + "=" * 65)
    print(f"RESULTADO: {correctos}/{total} correctos → {pct}%")
    if pct >= 80:
        print("🎉 Supera el umbral del 80% requerido")
    else:
        print("⚠️  Por debajo del 80% — revisar chunks o ampliar sinónimos en el txt")
    print("=" * 65)
    return pct


# 5. ANÁLISIS DE CASOS BORDE
def analizar_fallas(modelo, index, chunks):
    preguntas = [
        "¿Tienen sede en Medellín?",
        "¿El programa es gratis?",
        "¿Cuánto dura ESTRUCTURA?",
        "¿Qué es Latinoamérica Comparte?",
    ]
    print("\n" + "=" * 65)
    print("ANÁLISIS DE CASOS BORDE")
    print("=" * 65)
    for p in preguntas:
        res = retrieve_context(p, modelo, index, chunks, top_k=3, min_score=0.0)
        print(f"\n🔎 {p}")
        for r in res:
            print(f"   score={r['score']} | {r['seccion']}")
            print(f"   → {r['texto'][:100]}...")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    modelo, index, chunks = cargar_sistema()

    # Demo
    print("─" * 65)
    query_demo = "¿Qué programas ofrece Colombia Comparte para emprendedores?"
    resultados = retrieve_context(query_demo, modelo, index, chunks)
    contexto   = formatear_contexto(resultados)
    print(f"Query: {query_demo}\n")
    print(f"Chunks recuperados: {len(resultados)}")
    for r in resultados:
        print(f"  [{r['score']}] {r['seccion']}")
    print(f"\nContexto (primeros 400 chars):\n{contexto[:400]}...\n")

    # Evaluación
    evaluar_retrieval(modelo, index, chunks)

    # Casos borde
    analizar_fallas(modelo, index, chunks)

    print("\n🎉 Semana 2 lista. retrieve_context() y formatear_contexto() listos para Semana 3.")