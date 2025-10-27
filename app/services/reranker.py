"""Cross-encoder reranking for improved result quality."""
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Use smaller, faster cross-encoder model
# ms-marco-MiniLM-L-6-v2: ~50ms per pair, good quality
_reranker = None


def get_reranker():
    """Lazy load cross-encoder model."""
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder model: ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder loaded successfully")
    return _reranker


def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Rerank chunks using cross-encoder for better quality.

    Strategy:
    - Input: 10 candidates from hybrid search
    - Cross-encoder scores each (query, chunk) pair
    - Return top_k highest-scoring chunks

    Args:
        query: Search query
        chunks: List of candidate chunks from retrieval
        top_k: Number of results to return after reranking

    Returns:
        Reranked list of chunks with updated scores
    """
    if not chunks:
        return []

    logger.info(f"Reranking {len(chunks)} chunks with cross-encoder")

    try:
        reranker = get_reranker()

        # Prepare (query, chunk) pairs for batch scoring
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Batch inference (much faster than one-by-one)
        scores = reranker.predict(pairs)

        # Add reranker scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)
            chunk["original_score"] = chunk.get("score", 0.0)
            chunk["score"] = float(score)  # Replace with reranker score

        # Sort by reranker score and take top_k
        reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)[:top_k]

        logger.info(
            f"Reranking complete: top score={reranked[0]['reranker_score']:.4f}, "
            f"returned {len(reranked)} chunks"
        )

        return reranked

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        # Fallback to original ranking
        logger.warning("Falling back to original ranking")
        return chunks[:top_k]
