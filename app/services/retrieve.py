"""Retrieval service for vector search."""
from sqlalchemy import text
from typing import List, Optional
import logging
from app.db.session import engine
from app.services.embed import embed_query
from app.config import settings

logger = logging.getLogger(__name__)


def calculate_adaptive_threshold(scores: List[float], base_threshold: float = 0.3) -> float:
    """
    Calculate adaptive similarity threshold based on score distribution.

    Strategy:
    - If top scores are very high (> 0.7): Be more selective (higher threshold)
    - If top scores are low (< 0.4): Be more lenient (lower threshold)
    - Otherwise: Use base threshold

    Args:
        scores: List of similarity scores (descending order)
        base_threshold: Default threshold to use

    Returns:
        Adaptive threshold value
    """
    if not scores:
        return base_threshold

    top_score = scores[0]

    # High confidence - be more selective
    if top_score > 0.7:
        adaptive = 0.5
        logger.debug(f"High top score ({top_score:.3f}) - using stricter threshold: {adaptive}")
        return adaptive

    # Low confidence - be more lenient
    elif top_score < 0.4:
        adaptive = 0.2
        logger.debug(f"Low top score ({top_score:.3f}) - using lenient threshold: {adaptive}")
        return adaptive

    # Medium confidence - use base threshold
    else:
        logger.debug(f"Medium top score ({top_score:.3f}) - using base threshold: {base_threshold}")
        return base_threshold


def search_top_k(
    query: str,
    k: int = 6,
    doc_ids: Optional[List[str]] = None,
    min_score: Optional[float] = None
) -> List[dict]:
    """
    Search for top-k most similar chunks to the query.

    Args:
        query: The search query
        k: Number of results to return
        doc_ids: Optional filter by document IDs
        min_score: Minimum similarity score (0-1). Defaults to config MIN_SIMILARITY_SCORE

    Returns:
        List of chunks with metadata and similarity scores above threshold
    """
    # Set default minimum score from config if not provided
    if min_score is None:
        min_score = settings.MIN_SIMILARITY_SCORE
    # Generate query embedding
    query_vec = embed_query(query)
    logger.info(f"Query vector length: {len(query_vec)}")

    # Convert list to string format for pgvector
    vec_str = "[" + ",".join(map(str, query_vec)) + "]"
    logger.info(f"Vector string preview: {vec_str[:100]}...")

    # Build SQL query - use string replacement for vector since SQLAlchemy has issues with ::vector
    if doc_ids:
        # Filter by specific documents
        sql_query = text(f"""
            SELECT
                c.id,
                c.doc_id,
                c.chunk_id,
                c.page,
                c.text,
                d.title,
                d.filename,
                1 - (c.embedding <=> '{vec_str}'::vector) AS score
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.doc_id = ANY(:doc_ids)
            ORDER BY c.embedding <=> '{vec_str}'::vector
            LIMIT :k
        """)
        params = {"k": k, "doc_ids": doc_ids}
    else:
        # Search all documents
        sql_query = text(f"""
            SELECT
                c.id,
                c.doc_id,
                c.chunk_id,
                c.page,
                c.text,
                d.title,
                d.filename,
                1 - (c.embedding <=> '{vec_str}'::vector) AS score
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            ORDER BY c.embedding <=> '{vec_str}'::vector
            LIMIT :k
        """)
        params = {"k": k}

    # Execute search
    try:
        logger.info(f"Executing query with params: k={params.get('k')}, doc_ids={params.get('doc_ids', 'None')}")
        # Log first 200 chars of SQL for debugging
        sql_preview = str(sql_query)[:200] if len(str(sql_query)) > 200 else str(sql_query)
        logger.info(f"SQL preview: {sql_preview}...")
        with engine.begin() as conn:
            result = conn.execute(sql_query, params)
            rows = result.mappings().all()
            logger.info(f"Raw query returned {len(rows)} rows")

        # Convert rows to dicts
        all_results = [dict(row) for row in rows]

        # Calculate adaptive threshold based on score distribution
        scores = [r["score"] for r in all_results]
        adaptive_threshold = calculate_adaptive_threshold(scores, base_threshold=min_score)

        # Filter by adaptive threshold
        results = [r for r in all_results if r["score"] >= adaptive_threshold]

        logger.info(
            f"Retrieved {len(results)}/{len(all_results)} chunks above adaptive threshold "
            f"{adaptive_threshold:.2f} (base: {min_score:.2f}) for query: '{query[:50]}...'"
        )

        if results:
            logger.info(f"Top score: {results[0]['score']:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error during vector search: {e}", exc_info=True)
        raise
