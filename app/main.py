"""Main FastAPI application."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import text
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.routers import documents, ask
from app.models.schemas import HealthResponse
from app.db.session import engine
from app.services.analytics import get_analytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Knowledge Base API...")
    logger.info("API documentation available at /docs")
    yield
    # Shutdown
    logger.info("Shutting down Knowledge Base API...")


# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Knowledge Base API",
    description="""
## 🚀 Advanced RAG-powered Knowledge Base

Upload documents and ask questions powered by state-of-the-art AI retrieval.

### Features
- **📄 Multi-format Support**: PDF, DOCX, Markdown, TXT
- **🔍 Hybrid Search**: Combines BM25 keyword + Vector semantic search
- **🎯 Cross-Encoder Reranking**: Improves result quality significantly
- **🎨 Smart Highlighting**: Shows why chunks matched your query
- **📊 Analytics**: Track usage, performance, and popular queries
- **🔒 Deduplication**: Prevents uploading the same file twice
- **⚡ Response Caching**: Saves API costs for repeated queries

### Quick Start
1. **Upload a document**: `POST /documents`
2. **Ask a question**: `POST /ask`
3. **View analytics**: `GET /analytics`

### Advanced Options
- Enable `use_hybrid` for better keyword matching
- Enable `use_reranker` for highest quality results
- Enable `use_mmr` for diverse, non-redundant results
    """,
    version="1.0.0",
    contact={
        "name": "Knowledge Base API",
        "url": "https://github.com/yourusername/kb-api",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Question Answering",
            "description": "Ask questions and get AI-powered answers from your documents",
        },
        {
            "name": "Documents",
            "description": "Upload, manage, and delete documents",
        },
        {
            "name": "Analytics",
            "description": "View usage statistics and performance metrics",
        },
        {
            "name": "Health",
            "description": "Health check and system status",
        },
    ]
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"]
)

app.include_router(
    ask.router,
    tags=["Question Answering"]
)


@app.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
def health_check():
    """
    Health check endpoint.

    Returns the health status of the API and database connection.
    """
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return HealthResponse(
            status="healthy",
            database="connected"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database="disconnected"
        )


@app.get("/api", tags=["Root"])
def api_info():
    """
    API information endpoint.
    """
    return {
        "name": "Knowledge Base API",
        "version": "1.0.0",
        "description": "LLM-powered document Q&A with RAG",
        "ui": "/",
        "docs": "/docs",
        "health": "/healthz",
        "analytics": "/analytics"
    }


@app.get("/", tags=["Root"])
def serve_ui():
    """
    Serve the web UI.
    """
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    else:
        return {
            "message": "Web UI not available",
            "docs": "/docs",
            "api": "/api"
        }


@app.get("/analytics", tags=["Analytics"])
def analytics():
    """
    Get analytics and usage metrics.

    Provides insights into:
    - Query performance (response times, cache hit rate)
    - Upload performance (processing times, avg chunks)
    - Popular queries
    - Search method distribution
    - Recent activity
    """
    return get_analytics()
