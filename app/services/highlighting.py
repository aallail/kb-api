"""Smart text highlighting to show why chunks matched."""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def highlight_matches(text: str, query_terms: List[str], max_length: int = 500) -> str:
    """
    Highlight matching terms in text with **bold** markdown.

    Args:
        text: Text to highlight
        query_terms: List of terms to highlight
        max_length: Maximum text length (truncate if longer)

    Returns:
        Text with highlighted matches
    """
    if not query_terms or not text:
        return text[:max_length]

    highlighted = text
    matched_count = 0

    for term in query_terms:
        if len(term) < 2:  # Skip very short terms
            continue

        # Match whole words (case-insensitive)
        pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)

        # Count matches
        matches = pattern.findall(highlighted)
        if matches:
            matched_count += len(matches)
            # Highlight
            highlighted = pattern.sub(r'**\1**', highlighted)

    # Truncate if needed (but try to keep highlights)
    if len(highlighted) > max_length:
        # Find first highlight
        first_highlight = highlighted.find('**')
        if first_highlight > 0:
            # Center around first highlight
            start = max(0, first_highlight - max_length // 2)
            highlighted = highlighted[start:start + max_length]
            if start > 0:
                highlighted = "..." + highlighted
            if len(highlighted) >= max_length:
                highlighted = highlighted + "..."
        else:
            highlighted = highlighted[:max_length] + "..."

    return highlighted


def get_matched_terms(text: str, query_terms: List[str]) -> List[str]:
    """
    Get list of query terms that actually matched in the text.

    Args:
        text: Text to search
        query_terms: List of terms to find

    Returns:
        List of terms that were found
    """
    matched = []
    text_lower = text.lower()

    for term in query_terms:
        if len(term) < 2:
            continue
        if term.lower() in text_lower:
            matched.append(term)

    return matched


def generate_snippet(text: str, query_terms: List[str], snippet_length: int = 200) -> str:
    """
    Generate a smart snippet centered around the best match.

    Args:
        text: Full text
        query_terms: Query terms to find
        snippet_length: Desired snippet length

    Returns:
        Snippet centered around best match
    """
    if not query_terms or len(text) <= snippet_length:
        return text[:snippet_length]

    # Find best match position (most query terms nearby)
    best_pos = 0
    best_score = 0

    # Sliding window
    window_size = snippet_length
    for i in range(0, len(text) - window_size, 50):
        window = text[i:i + window_size].lower()
        score = sum(1 for term in query_terms if term.lower() in window)
        if score > best_score:
            best_score = score
            best_pos = i

    # Extract snippet
    snippet = text[best_pos:best_pos + snippet_length]

    # Add ellipsis
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + snippet_length < len(text):
        snippet = snippet + "..."

    return snippet
