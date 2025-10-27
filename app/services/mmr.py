"""Maximal Marginal Relevance (MMR) for result diversification."""
import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity_np(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors using numpy.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0-1)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def mmr_diversify(
    chunks: List[Dict[str, Any]],
    query_embedding: List[float],
    top_k: int = 6,
    lambda_param: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance to diversify results.

    MMR ensures selected results are:
    1. Relevant to the query (high similarity)
    2. Diverse from each other (low redundancy)

    Formula: MMR = argmax[λ × Sim(query, chunk) - (1-λ) × max(Sim(chunk, selected))]

    Args:
        chunks: List of candidate chunks (should have 'embedding' field)
        query_embedding: Query embedding vector
        top_k: Number of diverse results to return
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
                     0.7 = balanced (70% relevance, 30% diversity)

    Returns:
        Diversified list of top_k chunks
    """
    if not chunks or not query_embedding:
        return chunks[:top_k]

    if len(chunks) <= top_k:
        # No need for MMR if we have fewer chunks than top_k
        return chunks

    logger.info(f"Applying MMR diversification: {len(chunks)} candidates → {top_k} diverse results (λ={lambda_param})")

    try:
        selected = []
        remaining = chunks.copy()

        while len(selected) < top_k and remaining:
            mmr_scores = []

            for chunk in remaining:
                chunk_emb = chunk.get("embedding")
                if not chunk_emb:
                    # Fallback to original score if no embedding
                    mmr_scores.append(chunk.get("score", 0.0))
                    continue

                # Relevance: similarity to query
                relevance = cosine_similarity_np(query_embedding, chunk_emb)

                if not selected:
                    # First selection: just pick most relevant
                    max_similarity = 0
                else:
                    # Max similarity to already selected chunks
                    similarities = [
                        cosine_similarity_np(chunk_emb, s.get("embedding", []))
                        for s in selected
                    ]
                    max_similarity = max(similarities) if similarities else 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)

            # Select best MMR score
            best_idx = np.argmax(mmr_scores)
            best_chunk = remaining.pop(best_idx)
            best_chunk["mmr_score"] = float(mmr_scores[best_idx])
            selected.append(best_chunk)

        logger.info(f"MMR complete: selected {len(selected)} diverse chunks")
        return selected

    except Exception as e:
        logger.error(f"Error in MMR diversification: {e}")
        # Fallback to original ranking
        logger.warning("Falling back to original ranking")
        return chunks[:top_k]
