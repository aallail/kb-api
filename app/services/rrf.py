"""Reciprocal Rank Fusion (RRF) for combining search results."""
import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.

    RRF is a simple but effective method for combining multiple ranked lists.
    It's used by Elasticsearch, Pinecone, and other search systems.

    Formula: RRF_score = sum(1 / (k + rank_i))
    where rank_i is the rank in the i-th result list

    Benefits over weighted average:
    - No need to normalize scores across different systems
    - Rank-based (robust to score magnitude differences)
    - Simple and effective

    Args:
        vector_results: Ranked list from vector search
        bm25_results: Ranked list from BM25 search
        k: Constant (default 60, standard in literature)

    Returns:
        Combined and re-ranked list
    """
    logger.info(f"Applying RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 results")

    # Accumulate RRF scores
    rrf_scores = defaultdict(float)
    chunk_data = {}  # Store chunk data by ID

    # Process vector results
    for rank, chunk in enumerate(vector_results):
        chunk_id = chunk["id"]
        rrf_scores[chunk_id] += 1 / (k + rank + 1)
        chunk_data[chunk_id] = chunk
        if "vector_rank" not in chunk_data[chunk_id]:
            chunk_data[chunk_id]["vector_rank"] = rank + 1

    # Process BM25 results
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["id"]
        rrf_scores[chunk_id] += 1 / (k + rank + 1)
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = chunk
        chunk_data[chunk_id]["bm25_rank"] = rank + 1

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final result list
    results = []
    for chunk_id, rrf_score in sorted_ids:
        chunk = chunk_data[chunk_id]
        chunk["rrf_score"] = float(rrf_score)
        chunk["score"] = float(rrf_score)  # Use RRF as primary score
        results.append(chunk)

    logger.info(f"RRF fusion complete: {len(results)} combined results")

    if results:
        logger.debug(
            f"Top result: RRF={results[0]['rrf_score']:.4f}, "
            f"vector_rank={results[0].get('vector_rank', 'N/A')}, "
            f"bm25_rank={results[0].get('bm25_rank', 'N/A')}"
        )

    return results
