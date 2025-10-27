"""Analytics and metrics tracking."""
import logging
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# In-memory analytics storage (simple, no database needed)
_analytics = {
    "queries": [],  # Last 1000 queries
    "uploads": [],  # Last 1000 uploads
    "stats": {
        "total_queries": 0,
        "total_uploads": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
}

MAX_HISTORY = 1000  # Keep last 1000 events


def log_query(query: str, response_time_ms: float, cache_hit: bool,
              num_results: int, search_method: str, use_reranker: bool = False):
    """
    Log a query event.

    Args:
        query: User query
        response_time_ms: Response time in milliseconds
        cache_hit: Whether response was from cache
        num_results: Number of results returned
        search_method: "vector", "hybrid", or "hybrid+reranker"
        use_reranker: Whether reranker was used
    """
    event = {
        "query": query[:100],  # Truncate for privacy
        "timestamp": datetime.now().isoformat(),
        "response_time_ms": round(response_time_ms, 2),
        "cache_hit": cache_hit,
        "num_results": num_results,
        "search_method": search_method,
        "use_reranker": use_reranker
    }

    _analytics["queries"].append(event)
    _analytics["stats"]["total_queries"] += 1

    if cache_hit:
        _analytics["stats"]["cache_hits"] += 1
    else:
        _analytics["stats"]["cache_misses"] += 1

    # Keep only last MAX_HISTORY
    if len(_analytics["queries"]) > MAX_HISTORY:
        _analytics["queries"] = _analytics["queries"][-MAX_HISTORY:]

    logger.debug(f"Query logged: {query[:50]}... ({response_time_ms:.0f}ms, cache={cache_hit})")


def log_upload(filename: str, file_size: int, processing_time_ms: float,
               num_chunks: int, is_duplicate: bool = False):
    """
    Log a document upload event.

    Args:
        filename: Uploaded filename
        file_size: File size in bytes
        processing_time_ms: Processing time in milliseconds
        num_chunks: Number of chunks created
        is_duplicate: Whether it was a duplicate
    """
    event = {
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "file_size_kb": round(file_size / 1024, 2),
        "processing_time_ms": round(processing_time_ms, 2),
        "num_chunks": num_chunks,
        "is_duplicate": is_duplicate
    }

    _analytics["uploads"].append(event)
    _analytics["stats"]["total_uploads"] += 1

    # Keep only last MAX_HISTORY
    if len(_analytics["uploads"]) > MAX_HISTORY:
        _analytics["uploads"] = _analytics["uploads"][-MAX_HISTORY:]

    logger.debug(f"Upload logged: {filename} ({file_size/1024:.1f}KB, {num_chunks} chunks)")


def get_analytics() -> Dict[str, Any]:
    """
    Get analytics summary.

    Returns:
        Dictionary with analytics data
    """
    queries = _analytics["queries"]
    uploads = _analytics["uploads"]
    stats = _analytics["stats"]

    # Calculate averages
    if queries:
        avg_response_time = sum(q["response_time_ms"] for q in queries) / len(queries)
        avg_results = sum(q["num_results"] for q in queries) / len(queries)
    else:
        avg_response_time = 0
        avg_results = 0

    if uploads:
        avg_processing_time = sum(u["processing_time_ms"] for u in uploads) / len(uploads)
        avg_chunks = sum(u["num_chunks"] for u in uploads) / len(uploads)
    else:
        avg_processing_time = 0
        avg_chunks = 0

    # Cache hit rate
    total_queries = stats["cache_hits"] + stats["cache_misses"]
    cache_hit_rate = (stats["cache_hits"] / total_queries * 100) if total_queries > 0 else 0

    # Popular queries (top 10)
    query_counts = defaultdict(int)
    for q in queries:
        query_counts[q["query"]] += 1
    popular_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Search method distribution
    search_methods = defaultdict(int)
    for q in queries:
        method = q["search_method"]
        if q.get("use_reranker"):
            method += "+reranker"
        search_methods[method] += 1

    return {
        "overview": {
            "total_queries": stats["total_queries"],
            "total_uploads": stats["total_uploads"],
            "recent_queries": len(queries),
            "recent_uploads": len(uploads)
        },
        "query_performance": {
            "avg_response_time_ms": round(avg_response_time, 2),
            "avg_results_per_query": round(avg_results, 2),
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_hits": stats["cache_hits"],
            "cache_misses": stats["cache_misses"]
        },
        "upload_performance": {
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "avg_chunks_per_document": round(avg_chunks, 2)
        },
        "popular_queries": [{"query": q, "count": c} for q, c in popular_queries],
        "search_method_distribution": dict(search_methods),
        "recent_queries": queries[-10:],  # Last 10 queries
        "recent_uploads": uploads[-10:]   # Last 10 uploads
    }


def clear_analytics():
    """Clear all analytics data."""
    _analytics["queries"].clear()
    _analytics["uploads"].clear()
    _analytics["stats"] = {
        "total_queries": 0,
        "total_uploads": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    logger.info("Analytics cleared")
