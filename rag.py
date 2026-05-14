# ============================================================
# SEMANA 1 - RAG Colombia Comparte
# Chunking + Embeddings + FAISS
# Instalar: pip install sentence-transformers faiss-cpu numpy
# ============================================================

import re
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ── CONFIGURAR LOGGING ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("rag_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────
ARCHIVO_TXT   = "data/Colombia_Comparte_BASE_RAG_v2.txt"
MODELO_EMBED  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS_JSON   = "data/chunks.json"
INDEX_FAISS   = "data/index.faiss"
TOP_K         = 3          # cuántos chunks devolver por búsqueda
# ────────────────────────────────────────────────────────────


# 1. CARGAR TEXTO
def cargar_texto(path):
    logger.info(f"Iniciando carga de archivo: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            texto = f.read()
        logger.info(f"✅ Archivo cargado exitosamente. Tamaño: {len(texto)} caracteres")
        return texto
    except FileNotFoundError:
        logger.error(f"❌ Error: Archivo no encontrado en {path}")
        raise
    except Exception as e:
        logger.error(f"❌ Error al cargar archivo: {str(e)}")
        raise


# 2. CHUNKING POR SECCIONES  [SECCIÓN N - TEMA]
def hacer_chunks(texto):
    logger.info("Iniciando proceso de chunking...")
    try:
        partes = re.split(r"\[SECCIÓN \d+.*?\]", texto)
        titulos = re.findall(r"\[SECCIÓN \d+.*?\]", texto)
        logger.debug(f"Encontradas {len(titulos)} secciones en el texto")

        chunks = []
        for i, contenido in enumerate(partes[1:]):  # partes[0] es el encabezado
            bloque = contenido.strip()
            if not bloque:
                logger.debug(f"Sección {i+1} vacía, omitida")
                continue
            chunks.append({
                "id": i + 1,
                "seccion": titulos[i].strip("[]"),
                "texto": bloque
            })
        logger.info(f"✅ {len(chunks)} chunks generados exitosamente")
        return chunks
    except Exception as e:
        logger.error(f"❌ Error durante el chunking: {str(e)}")
        raise


# 3. GENERAR EMBEDDINGS
def generar_embeddings(chunks, modelo_nombre):
    logger.info(f"Iniciando generación de embeddings")
    logger.info(f"Modelo a utilizar: {modelo_nombre}")
    try:
        logger.info(f"⏳ Cargando modelo...")
        modelo = SentenceTransformer(modelo_nombre)
        logger.info(f"✅ Modelo cargado exitosamente")
        
        textos = [c["texto"] for c in chunks]
        logger.info(f"⏳ Generando embeddings para {len(textos)} textos...")
        embeddings = modelo.encode(textos, show_progress_bar=True, normalize_embeddings=True)
        logger.info(f"✅ Embeddings generados: shape {embeddings.shape} (dim: {embeddings.shape[1]})")
        return modelo, embeddings
    except Exception as e:
        logger.error(f"❌ Error durante la generación de embeddings: {str(e)}")
        raise


# 4. GUARDAR CHUNKS EN JSON
def guardar_chunks(chunks, path):
    logger.info(f"Guardando chunks en: {path}")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ {len(chunks)} chunks guardados exitosamente en {path}")
    except Exception as e:
        logger.error(f"❌ Error al guardar chunks: {str(e)}")
        raise


# 5. CREAR Y GUARDAR ÍNDICE FAISS
def crear_indice(embeddings, path):
    logger.info(f"Iniciando creación de índice FAISS")
    logger.info(f"Dimensión de embeddings: {embeddings.shape[1]}")
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # Inner Product = cosine si vectores normalizados
        logger.info(f"⏳ Agregando {embeddings.shape[0]} vectores al índice...")
        index.add(embeddings.astype("float32"))
        logger.info(f"✅ Índice creado con {index.ntotal} vectores")
        
        logger.info(f"⏳ Guardando índice en: {path}")
        faiss.write_index(index, path)
        logger.info(f"✅ Índice FAISS guardado exitosamente en {path}")
        return index
    except Exception as e:
        logger.error(f"❌ Error al crear/guardar índice FAISS: {str(e)}")
        raise


# 6. FUNCIÓN DE BÚSQUEDA SEMÁNTICA
def retrieve_context(query, modelo, index, chunks, top_k=TOP_K):
    logger.debug(f"Iniciando búsqueda semántica. Query: '{query}' (top_k={top_k})")
    try:
        q_vec = modelo.encode([query], normalize_embeddings=True).astype("float32")
        logger.debug(f"Query embedding generado: shape {q_vec.shape}")
        
        scores, ids = index.search(q_vec, top_k)
        logger.debug(f"Búsqueda ejecutada. Resultados encontrados: {len(ids[0])}")
        
        resultados = []
        for score, idx in zip(scores[0], ids[0]):
            resultados.append({
                "seccion": chunks[idx]["seccion"],
                "texto":   chunks[idx]["texto"],
                "score":   round(float(score), 4)
            })
        logger.debug(f"✅ Búsqueda completada con {len(resultados)} resultados")
        return resultados
    except Exception as e:
        logger.error(f"❌ Error durante la búsqueda semántica: {str(e)}")
        raise


# 7. VALIDACIÓN: prueba de búsqueda
def validar_busqueda(modelo, index, chunks):
    logger.info("═" * 60)
    logger.info("INICIANDO VALIDACIÓN DE BÚSQUEDA SEMÁNTICA")
    logger.info("═" * 60)
    
    preguntas_prueba = [
        "¿Qué es Colombia Comparte?",
        "¿Cómo me inscribo al programa EDIFICA?",
        "¿Qué es Comparte Academia?",
        "¿Cuántas personas han sido impactadas?",
        "¿Cómo puedo contactarlos?"
    ]
    
    try:
        for i, pregunta in enumerate(preguntas_prueba, 1):
            logger.info(f"\n[Prueba {i}/{len(preguntas_prueba)}] Pregunta: {pregunta}")
            resultados = retrieve_context(pregunta, modelo, index, chunks)
            for j, r in enumerate(resultados, 1):
                logger.info(f"  Resultado {j}: [{r['score']}] {r['seccion']}")
                logger.debug(f"    → {r['texto'][:120]}...")
        logger.info("\n" + "═" * 60)
        logger.info("✅ VALIDACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("═" * 60)
    except Exception as e:
        logger.error(f"❌ Error durante la validación: {str(e)}")
        raise


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║" + "INICIANDO RAG - Colombia Comparte".center(58) + "║")
    logger.info("║" + f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(58) + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    
    try:
        os.makedirs("data", exist_ok=True)
        logger.info("✅ Directorio 'data' verificado/creado")

        logger.info("\n📖 PASO 1: Cargando texto...")
        texto = cargar_texto(ARCHIVO_TXT)

        logger.info("\n✂️  PASO 2: Chunking del texto...")
        chunks = hacer_chunks(texto)

        logger.info("\n💾 PASO 3: Guardando chunks...")
        guardar_chunks(chunks, CHUNKS_JSON)

        logger.info("\n🤖 PASO 4: Generando embeddings...")
        modelo, embeddings = generar_embeddings(chunks, MODELO_EMBED)

        logger.info("\n🗂️  PASO 5: Creando índice FAISS...")
        index = crear_indice(embeddings, INDEX_FAISS)

        logger.info("\n🧪 PASO 6: Validando búsqueda semántica...")
        validar_busqueda(modelo, index, chunks)

        logger.info("\n" + "╔" + "═" * 58 + "╗")
        logger.info("║" + "🎉 RAG COMPLETADO EXITOSAMENTE 🎉".center(58) + "║")
        logger.info("║" + " " * 58 + "║")
        logger.info(f"║ Archivos generados:".ljust(60) + "║"[:60])
        logger.info(f"║   ✓ {CHUNKS_JSON}".ljust(60) + "║"[:60])
        logger.info(f"║   ✓ {INDEX_FAISS}".ljust(60) + "║"[:60])
        logger.info(f"║   ✓ rag_logs.log (este archivo)".ljust(60) + "║"[:60])
        logger.info("║" + " " * 58 + "║")
        logger.info("╚" + "═" * 58 + "╝")
        
    except Exception as e:
        logger.critical(f"❌ ERROR CRÍTICO: {str(e)}")
        logger.critical("El proceso se detuvo debido a un error fatal")
        raise