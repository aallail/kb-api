"""Question answering endpoint."""
from fastapi import APIRouter, Depends, HTTPException, status, Request
import logging
import time
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import AskRequest, AskResponse, Source, ResponseMetadata
from datetime import datetime
from app.utils.security import require_password
from app.services.retrieve import search_top_k
from app.services.hybrid_search import hybrid_search
from app.services.generate import answer_with_context
from app.services.cache import get_cache
from app.services.query_preprocessing import preprocess_query, extract_query_terms
from app.services.reranker import rerank_chunks
from app.services.mmr import mmr_diversify
from app.services.highlighting import highlight_matches
from app.services.analytics import log_query
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question"
)
@limiter.limit("30/hour")  # Limit queries to 30 per hour
def ask_question(
    request: Request,
    body: AskRequest,
    _: str = Depends(require_password)
):
    """
    Ask a question and get an answer based on your uploaded documents.

    The system will:
    1. Find the most relevant chunks from your documents
    2. Use them as context for an LLM
    3. Generate a cited answer

    **Parameters:**
    - `query`: Your question (required)
    - `top_k`: Number of document chunks to retrieve (default: 6)
    - `doc_ids`: Filter to specific documents (optional)

    **Response:**
    - `answer`: The generated answer with inline citations like [1], [2]
    - `sources`: List of source chunks used, with scores and metadata
    """
    try:
        start_time = time.time()
        top_k = body.top_k or settings.DEFAULT_TOP_K
        cache = get_cache()

        # 1. Preprocess query (fix typos, expand abbreviations)
        preprocessed_query = preprocess_query(body.query)
        query_terms = extract_query_terms(preprocessed_query)

        if preprocessed_query != body.query:
            logger.info(f"Preprocessed: '{body.query}' -> '{preprocessed_query}'")

        # Check cache first (saves money!)
        cache_key_suffix = f"_hybrid" if body.use_hybrid else ""
        cache_key_suffix += f"_reranker" if body.use_reranker else ""
        cache_key_suffix += f"_mmr" if body.use_mmr else ""
        cached_response = cache.get(
            query=preprocessed_query + cache_key_suffix,
            doc_ids=body.doc_ids,
            top_k=top_k
        )

        if cached_response:
            logger.info(f"Returning cached response for: '{body.query[:50]}...'")
            # Track analytics (cache hit)
            elapsed_ms = (time.time() - start_time) * 1000
            search_method = "hybrid" if body.use_hybrid else "vector"
            log_query(
                query=body.query,
                response_time_ms=elapsed_ms,
                cache_hit=True,
                num_results=len(cached_response.get("sources", [])),
                search_method=search_method,
                use_reranker=body.use_reranker
            )
            # Update metadata for cached response
            cached_response["metadata"] = ResponseMetadata(
                response_time_ms=elapsed_ms,
                cached=True,
                search_method=search_method,
                reranker_used=body.use_reranker,
                mmr_used=body.use_mmr,
                num_chunks_retrieved=len(cached_response.get("sources", [])),
                timestamp=datetime.now()
            )
            return AskResponse(**cached_response)

        # Cache miss - proceed with search
        # For reranker/MMR, fetch more candidates initially
        initial_k = top_k * 3 if (body.use_reranker or body.use_mmr) else top_k

        # Choose search method
        if body.use_hybrid:
            logger.info("Using HYBRID search (BM25 + Vector)")
            chunks = hybrid_search(
                query=preprocessed_query,
                k=initial_k,
                doc_ids=body.doc_ids
            )
        else:
            logger.info("Using VECTOR-ONLY search")
            chunks = search_top_k(
                query=preprocessed_query,
                k=initial_k,
                doc_ids=body.doc_ids
            )

        # 2. Apply reranking if requested (improves quality significantly)
        if body.use_reranker and chunks:
            logger.info(f"Applying cross-encoder reranking: {len(chunks)} -> {min(10, len(chunks))}")
            chunks = rerank_chunks(
                query=preprocessed_query,
                chunks=chunks,
                top_k=min(10, len(chunks))  # Rank top 10 candidates as requested by user
            )

        # 3. Apply MMR diversity if requested (reduces redundancy)
        if body.use_mmr and chunks and len(chunks) > top_k:
            logger.info(f"Applying MMR diversification: {len(chunks)} -> {top_k}")
            # Need embeddings for MMR - use query embedding
            from app.services.embed import embed_query
            query_embedding = embed_query(preprocessed_query)
            chunks = mmr_diversify(
                chunks=chunks,
                query_embedding=query_embedding,
                top_k=top_k,
                lambda_param=0.7  # 70% relevance, 30% diversity
            )
        elif not body.use_mmr and len(chunks) > top_k:
            # Just truncate to top_k if not using MMR
            chunks = chunks[:top_k]

        if not chunks:
            # Provide helpful error message with suggestions
            error_detail = {
                "message": "I couldn't find relevant information to answer your question.",
                "suggestions": [
                    "Try rephrasing your question with different keywords",
                    "Make your question more specific",
                    "Check if your documents cover this topic",
                    "Try using hybrid search (set use_hybrid=true) for better keyword matching"
                ],
                "query": body.query,
                "search_method": "hybrid" if body.use_hybrid else "vector",
                "documents_available": "Check /documents to see what's uploaded"
            }
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_detail
            )

        # Generate answer
        answer = answer_with_context(
            query=body.query,
            ranked_chunks=chunks
        )

        # Format sources with smart highlighting
        sources = [
            Source(
                chunk_id=chunk["id"],
                doc_id=str(chunk["doc_id"]),  # Convert UUID to string
                page=chunk.get("page"),
                score=round(chunk["score"], 4),
                # Highlight matching terms in preview
                text_preview=highlight_matches(
                    text=chunk["text"],
                    query_terms=query_terms,
                    max_length=200
                )
            )
            for chunk in chunks
        ]

        # Track analytics and timing (cache miss)
        elapsed_ms = (time.time() - start_time) * 1000
        search_method = "hybrid" if body.use_hybrid else "vector"
        log_query(
            query=body.query,
            response_time_ms=elapsed_ms,
            cache_hit=False,
            num_results=len(sources),
            search_method=search_method,
            use_reranker=body.use_reranker
        )

        response_data = {
            "answer": answer,
            "sources": sources,
            "query": body.query,
            "metadata": ResponseMetadata(
                response_time_ms=elapsed_ms,
                cached=False,
                search_method=search_method,
                reranker_used=body.use_reranker,
                mmr_used=body.use_mmr,
                num_chunks_retrieved=initial_k,
                timestamp=datetime.now()
            )
        }

        # Cache the response for next time (without metadata which changes)
        cache_data = {
            "answer": answer,
            "sources": sources,
            "query": body.query
        }
        cache.set(
            query=preprocessed_query + cache_key_suffix,
            doc_ids=body.doc_ids,
            top_k=top_k,
            response=cache_data
        )

        return AskResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}"
        )
