"""Hybrid search combining BM25 (keyword) and vector (semantic) search."""
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sqlalchemy import text

from app.db.session import engine
from app.services.embed import embed_query
from app.services.rrf import reciprocal_rank_fusion
from app.config import settings

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    k: int = 6,
    doc_ids: Optional[List[str]] = None,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    min_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining BM25 (keyword) and vector (semantic) search.

    This gives better results than vector search alone because:
    - BM25 catches exact keyword matches that vectors might miss
    - Vector search catches semantic similarities
    - Combining both gives best of both worlds

    Args:
        query: The search query
        k: Number of results to return
        doc_ids: Optional filter by document IDs
        vector_weight: Weight for vector similarity (0-1)
        bm25_weight: Weight for BM25 score (0-1)
        min_score: Minimum combined score threshold

    Returns:
        List of chunks ranked by hybrid score
    """
    if min_score is None:
        min_score = settings.MIN_SIMILARITY_SCORE

    logger.info(f"Hybrid search: query='{query[:50]}...', k={k}, weights=(vec:{vector_weight}, bm25:{bm25_weight})")

    # Step 1: Fetch all candidate chunks from database
    chunks = _fetch_all_chunks(doc_ids)
    if not chunks:
        logger.warning("No chunks found in database")
        return []

    logger.info(f"Fetched {len(chunks)} candidate chunks")

    # Step 2: Rank by BM25
    bm25_scores = _compute_bm25_scores(query, chunks)
    bm25_ranked = [(chunks[i], bm25_scores[i]) for i in range(len(chunks))]
    bm25_ranked.sort(key=lambda x: x[1], reverse=True)
    bm25_results = [chunk for chunk, score in bm25_ranked]

    # Step 3: Rank by Vector similarity
    vector_scores = _compute_vector_scores(query, chunks)
    vector_ranked = [(chunks[i], vector_scores[i]) for i in range(len(chunks))]
    vector_ranked.sort(key=lambda x: x[1], reverse=True)
    vector_results = [chunk for chunk, score in vector_ranked]

    # Step 4: Use RRF to combine rankings (better than weighted average!)
    # RRF is used by Elasticsearch, Pinecone, and other production systems
    logger.info("Using RRF (Reciprocal Rank Fusion) to combine rankings")
    combined_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        k=60  # Standard RRF constant
    )

    # Step 5: Return top k (NO threshold filtering for RRF scores)
    # RRF scores are much lower than cosine similarity (typically 0.01-0.05)
    # so we don't filter by MIN_SIMILARITY_SCORE here
    top_results = combined_results[:k]

    top_score = f"{top_results[0]['rrf_score']:.4f}" if top_results else "N/A"
    logger.info(
        f"Hybrid search (RRF) returned {len(top_results)} chunks "
        f"(top RRF score: {top_score})"
    )

    if top_results:
        logger.info(
            f"Top result: RRF={top_results[0]['rrf_score']:.4f}, "
            f"vector_rank={top_results[0].get('vector_rank', 'N/A')}, "
            f"bm25_rank={top_results[0].get('bm25_rank', 'N/A')}"
        )

    return top_results


def _fetch_all_chunks(doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Fetch all chunks from database (for BM25 indexing)."""
    try:
        if doc_ids:
            sql_query = text("""
                SELECT
                    c.id,
                    c.doc_id,
                    c.chunk_id,
                    c.page,
                    c.text,
                    c.embedding,
                    d.title,
                    d.filename
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.doc_id = ANY(:doc_ids)
            """)
            params = {"doc_ids": doc_ids}
        else:
            sql_query = text("""
                SELECT
                    c.id,
                    c.doc_id,
                    c.chunk_id,
                    c.page,
                    c.text,
                    c.embedding,
                    d.title,
                    d.filename
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
            """)
            params = {}

        with engine.begin() as conn:
            result = conn.execute(sql_query, params)
            rows = result.mappings().all()

        chunks = [dict(row) for row in rows]
        return chunks

    except Exception as e:
        logger.error(f"Error fetching chunks: {e}", exc_info=True)
        return []


def _compute_bm25_scores(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    """
    Compute BM25 scores for all chunks.

    BM25 is a keyword-based ranking function that works well for:
    - Exact keyword matches
    - Technical terms
    - Proper nouns (names, places, etc.)
    """
    # Tokenize all documents
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Tokenize query
    tokenized_query = query.lower().split()

    # Get BM25 scores
    raw_scores = bm25.get_scores(tokenized_query)

    # Normalize scores to 0-1 range
    max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
    normalized_scores = [score / max_score for score in raw_scores]

    logger.debug(f"BM25 scores: min={min(normalized_scores):.3f}, max={max(normalized_scores):.3f}, avg={sum(normalized_scores)/len(normalized_scores):.3f}")

    return normalized_scores


def _compute_vector_scores(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    """
    Compute vector similarity scores for all chunks.

    Uses cosine similarity between query embedding and chunk embeddings.
    """
    # Generate query embedding
    query_vec = embed_query(query)

    scores = []
    for chunk in chunks:
        # Chunk embedding is already stored
        chunk_vec = chunk["embedding"]

        # Compute cosine similarity
        # Note: pgvector stores embeddings as lists
        similarity = _cosine_similarity(query_vec, chunk_vec)
        scores.append(similarity)

    logger.debug(f"Vector scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")

    return scores


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns value between 0 and 1.
    """
    # Handle case where vectors might be different types
    if isinstance(vec1, str):
        vec1 = [float(x.strip()) for x in vec1.strip('[]').split(',')]
    elif not isinstance(vec1, list):
        vec1 = list(vec1)

    if isinstance(vec2, str):
        vec2 = [float(x.strip()) for x in vec2.strip('[]').split(',')]
    elif not isinstance(vec2, list):
        vec2 = list(vec2)

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    similarity = dot_product / (magnitude1 * magnitude2)

    # Clamp to [0, 1] range
    return max(0.0, min(1.0, similarity))
