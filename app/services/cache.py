"""Response caching to reduce API costs and improve performance."""
import hashlib
import json
import logging
from functools import lru_cache
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# In-memory cache using LRU (Least Recently Used)
# Stores up to 500 query responses
# This saves money by not calling the LLM API for repeated queries

class ResponseCache:
    """Simple in-memory cache for LLM responses."""

    def __init__(self, max_size: int = 500, ttl_hours: int = 24):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached responses
            ttl_hours: Time-to-live in hours before cache expires
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        logger.info(f"Initialized response cache (max_size={max_size}, ttl={ttl_hours}h)")

    def _generate_cache_key(self, query: str, doc_ids: Optional[list], top_k: int) -> str:
        """
        Generate a unique cache key for the query parameters.

        Args:
            query: The user's question
            doc_ids: List of document IDs (if filtering)
            top_k: Number of chunks to retrieve

        Returns:
            SHA256 hash of the parameters
        """
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()

        # Sort doc_ids for consistency
        sorted_doc_ids = sorted(doc_ids) if doc_ids else []

        # Create unique key from query params
        key_data = {
            "query": normalized_query,
            "doc_ids": sorted_doc_ids,
            "top_k": top_k
        }

        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()

        return cache_key

    def get(self, query: str, doc_ids: Optional[list], top_k: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available and not expired.

        Args:
            query: The user's question
            doc_ids: List of document IDs
            top_k: Number of chunks

        Returns:
            Cached response dict or None if not found/expired
        """
        cache_key = self._generate_cache_key(query, doc_ids, top_k)

        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            cached_time = cached_item["cached_at"]
            expiry_time = cached_time + timedelta(hours=self.ttl_hours)

            # Check if cache entry has expired
            if datetime.now() < expiry_time:
                logger.info(f"Cache HIT for query: '{query[:50]}...' (saves $$$)")
                return cached_item["response"]
            else:
                # Expired - remove from cache
                logger.info(f"Cache EXPIRED for query: '{query[:50]}...'")
                del self.cache[cache_key]

        logger.info(f"Cache MISS for query: '{query[:50]}...'")
        return None

    def set(self, query: str, doc_ids: Optional[list], top_k: int, response: Dict[str, Any]) -> None:
        """
        Store response in cache.

        Args:
            query: The user's question
            doc_ids: List of document IDs
            top_k: Number of chunks
            response: The LLM response to cache
        """
        cache_key = self._generate_cache_key(query, doc_ids, top_k)

        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            # Remove oldest item (simple FIFO for now)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache full - evicted oldest entry")

        self.cache[cache_key] = {
            "response": response,
            "cached_at": datetime.now()
        }

        logger.info(f"Cached response for query: '{query[:50]}...' (cache size: {len(self.cache)})")

    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_hours
        }


# Global cache instance
_cache = ResponseCache(max_size=500, ttl_hours=24)


def get_cache() -> ResponseCache:
    """Get the global cache instance."""
    return _cache
