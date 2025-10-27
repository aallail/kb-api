"""Query preprocessing for better search results."""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Common abbreviations and expansions
ABBREVIATIONS = {
    "pls": "please",
    "thx": "thanks",
    "ty": "thank you",
    "btw": "by the way",
    "fyi": "for your information",
    "asap": "as soon as possible",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tl;dr": "summary",
    "tldr": "summary",
    "afaik": "as far as I know",
    "iirc": "if I recall correctly",
    "etc": "et cetera",
    "vs": "versus",
    "e.g": "for example",
    "i.e": "that is",
}

# Common typos (can expand this list)
TYPOS = {
    "teh": "the",
    "taht": "that",
    "waht": "what",
    "dont": "don't",
    "cant": "can't",
    "wont": "won't",
    "didnt": "didn't",
    "doesnt": "doesn't",
}


def preprocess_query(query: str) -> str:
    """
    Preprocess query for better search results.

    Steps:
    1. Normalize whitespace
    2. Fix common typos
    3. Expand abbreviations
    4. Remove extra punctuation

    Args:
        query: Raw user query

    Returns:
        Cleaned query
    """
    if not query:
        return query

    original = query

    # 1. Normalize whitespace
    query = " ".join(query.split())

    # 2. Fix common typos
    words = query.split()
    words = [TYPOS.get(w.lower(), w) for w in words]

    # 3. Expand abbreviations
    words = [ABBREVIATIONS.get(w.lower(), w) for w in words]

    # 4. Remove excessive punctuation (keep basic)
    query = " ".join(words)
    query = re.sub(r'([.!?]){2,}', r'\1', query)  # Multiple punctuation → single
    query = re.sub(r'\s+([.!?,;:])', r'\1', query)  # Space before punctuation

    # 5. Trim
    query = query.strip()

    if query != original:
        logger.debug(f"Query preprocessed: '{original}' → '{query}'")

    return query


def extract_keywords(query: str) -> List[str]:
    """
    Extract important keywords from query.

    Useful for highlighting and analytics.

    Args:
        query: Search query

    Returns:
        List of important keywords
    """
    # Remove common stopwords
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "must", "can", "of", "to", "in", "on", "at",
        "by", "for", "with", "about", "as", "from", "that", "this", "what",
        "which", "who", "when", "where", "why", "how", "it", "its"
    }

    words = query.lower().split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2]

    return keywords


# Alias for compatibility
extract_query_terms = extract_keywords
