"""Text chunking utilities using Anthropic tokenizer."""
import anthropic
from typing import List
from app.config import settings


# Initialize Anthropic client for tokenization
_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def chunk_text(
    text: str,
    max_tokens: int = None,
    overlap: int = None
) -> List[str]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk (default from config)
        overlap: Token overlap between chunks (default from config)

    Returns:
        List of text chunks
    """
    if max_tokens is None:
        max_tokens = settings.CHUNK_SIZE
    if overlap is None:
        overlap = settings.CHUNK_OVERLAP

    # Use character-based approximation (4 chars ≈ 1 token)
    # This is faster than calling the API for every chunk
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    step = max_chars - overlap_chars

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = text[start:end].rfind(punct)
                if last_punct != -1:
                    end = start + last_punct + len(punct)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += step

        # Prevent infinite loop
        if start >= len(text):
            break

    return chunks


def count_tokens(text: str, model: str = None) -> int:
    """
    Count tokens in text using Anthropic's tokenizer.

    Args:
        text: The text to count tokens for
        model: The model to use for encoding (defaults to config LLM_MODEL)

    Returns:
        Number of tokens
    """
    if model is None:
        model = settings.LLM_MODEL

    try:
        # Use Anthropic's count_tokens API
        token_count = _client.count_tokens(text)
        return token_count
    except Exception:
        # Fallback to character-based approximation if API fails
        return len(text) // 4
