"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    doc_id: str
    filename: str
    chunks: int
    status: str = "processed"
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    uploaded_at: Optional[datetime] = Field(None, description="Upload timestamp")


class DocumentMetadata(BaseModel):
    """Metadata about a document."""
    doc_id: str
    title: Optional[str] = None
    filename: str
    mime: Optional[str] = None
    status: str
    created_at: datetime


class Source(BaseModel):
    """Source citation for an answer."""
    chunk_id: int
    doc_id: str
    page: Optional[int] = None
    score: float
    text_preview: Optional[str] = None


class AskRequest(BaseModel):
    """Request to ask a question."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask",
        examples=["What is machine learning?", "How does photosynthesis work?"]
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (default: 6)",
        examples=[6]
    )
    doc_ids: Optional[List[str]] = Field(
        None,
        description="Filter by specific document IDs",
        examples=[["550e8400-e29b-41d4-a716-446655440000"]]
    )
    use_hybrid: Optional[bool] = Field(
        False,
        description="Use hybrid search (BM25 + Vector) for better keyword matching",
        examples=[True]
    )
    use_reranker: Optional[bool] = Field(
        False,
        description="Use cross-encoder reranking for highest quality results (+50ms latency)",
        examples=[True]
    )
    use_mmr: Optional[bool] = Field(
        False,
        description="Use MMR (Maximal Marginal Relevance) for diverse, non-redundant results",
        examples=[False]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the benefits of renewable energy?",
                    "top_k": 6,
                    "use_hybrid": True,
                    "use_reranker": True
                }
            ]
        }
    }


class ResponseMetadata(BaseModel):
    """Metadata about the response."""
    response_time_ms: float = Field(description="Response time in milliseconds")
    cached: bool = Field(description="Whether this response was cached")
    search_method: str = Field(description="Search method used (vector/hybrid)")
    reranker_used: bool = Field(default=False, description="Whether reranking was applied")
    mmr_used: bool = Field(default=False, description="Whether MMR diversification was used")
    num_chunks_retrieved: int = Field(description="Number of chunks initially retrieved")
    timestamp: datetime = Field(description="Response timestamp")


class AskResponse(BaseModel):
    """Response with answer and sources."""
    answer: str
    sources: List[Source]
    query: str
    metadata: ResponseMetadata


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    database: str = "connected"
