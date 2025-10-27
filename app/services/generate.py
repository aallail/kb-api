"""LLM generation service for answering questions using Claude."""
import anthropic
from typing import List
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based strictly on the provided context.

Rules:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I don't have enough information to answer that question based on the provided documents."
3. Always cite your sources using [1], [2], etc. inline where you reference information
4. Be concise and accurate
5. Do not make up or infer information not present in the context"""


def answer_with_context(query: str, ranked_chunks: List[dict]) -> str:
    """
    Generate an answer to the query using the provided context chunks.

    Args:
        query: The user's question
        ranked_chunks: List of relevant chunks with metadata

    Returns:
        Generated answer with citations
    """
    if not ranked_chunks:
        return "I don't have any relevant information to answer that question."

    # Build context block
    context_parts = []
    for i, chunk in enumerate(ranked_chunks[:8], start=1):  # Limit to top 8
        doc_info = f"doc={chunk['doc_id']}"
        if chunk.get('page'):
            doc_info += f", page={chunk['page']}"
        if chunk.get('filename'):
            doc_info += f", file={chunk['filename']}"

        context_parts.append(f"[{i}] ({doc_info})\n{chunk['text']}")

    context_block = "\n\n".join(context_parts)

    # Build the prompt
    user_prompt = f"""Question: {query}

Context:
{context_block}

Please provide a concise answer based on the context above. Include citations like [1], [2] where relevant."""

    # Call Claude
    try:
        logger.info(f"Calling {settings.LLM_MODEL} with {len(ranked_chunks)} chunks for query: '{query[:50]}...'")

        response = client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.content[0].text.strip()
        logger.info(
            f"Generated answer ({len(answer)} chars) using {settings.LLM_MODEL}. "
            f"Usage: {response.usage.input_tokens} in / {response.usage.output_tokens} out"
        )
        return answer

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e.status_code} - {e.message}")
        raise Exception(f"Failed to generate answer: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error generating answer: {e}", exc_info=True)
        raise Exception(f"Failed to generate answer: {str(e)}")
