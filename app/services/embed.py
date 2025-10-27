"""Embedding service using local sentence-transformers."""
from sentence_transformers import SentenceTransformer
from typing import List
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize local embedding model (loaded once at startup)
_model = None


def _get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(f"Model loaded successfully. Dimension: {settings.EMBEDDING_DIMENSION}")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using local model.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    try:
        model = _get_model()
        # Generate embeddings using the local model
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list = [emb.tolist() for emb in embeddings]

        logger.info(f"Generated {len(embeddings_list)} embeddings locally (FREE!)")
        return embeddings_list

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single query.

    Args:
        query: The query text

    Returns:
        Embedding vector
    """
    try:
        model = _get_model()
        embedding = model.encode(query, convert_to_numpy=True, show_progress_bar=False)

        logger.info(f"Generated query embedding locally (FREE!)")
        return embedding.tolist()

    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise
