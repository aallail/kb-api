# Knowledge Base API

An LLM-powered Knowledge Base API that lets you upload documents, index them for semantic search, and answer questions grounded in those documents using RAG (Retrieval-Augmented Generation).

Think "private ChatGPT over your files."

## Features

- 📄 **Document Upload**: Support for PDF, DOCX, Markdown, and plain text
- 🔍 **Semantic Search**: Vector similarity search using PostgreSQL pgvector
- 🤖 **AI-Powered Q&A**: Answers grounded in your documents with citations
- 🔐 **Simple Authentication**: API key-based access control
- 🐳 **Docker Ready**: Complete Docker Compose setup
- 📊 **Interactive Docs**: Auto-generated Swagger UI
- 🎨 **Multiple UIs**: Web frontend, CLI tool, and enhanced API docs
- ⚡ **Response Metadata**: Timing, cache status, and search method info

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone and setup**:
   ```bash
   cd kb-api
   cp .env.example .env
   ```

2. **Configure environment**:
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...
   API_KEY=your-secret-key
   ```

3. **Start the services**:
   ```bash
   docker compose up --build
   ```

4. **Access the API**:
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/healthz
   - CLI Tool: `python kb.py health`

## Usage

You can interact with the API in three ways:
1. **Web UI** - Visit http://localhost:8000 for a beautiful web interface
2. **CLI Tool** - Use `kb.py` for command-line access (see [UI_IMPROVEMENTS.md](UI_IMPROVEMENTS.md))
3. **Direct API** - Use curl/Postman/code as shown below

### Upload a Document

```bash
curl -X POST "http://localhost:8000/documents/" \
  -H "x-api-key: your-secret-key" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "doc_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "document.pdf",
  "chunks": 42,
  "status": "processed"
}
```

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "x-api-key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main features?",
    "top_k": 5
  }'
```

Response:
```json
{
  "answer": "Based on the documents, the main features are: ...",
  "sources": [
    {
      "chunk_id": 1,
      "doc_id": "123e4567...",
      "page": 3,
      "score": 0.89,
      "text_preview": "..."
    }
  ],
  "query": "What are the main features?"
}
```

### Get Document Info

```bash
curl "http://localhost:8000/documents/{doc_id}" \
  -H "x-api-key: your-secret-key"
```

### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/documents/{doc_id}" \
  -H "x-api-key: your-secret-key"
```

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation with:
- All endpoints and parameters
- Request/response schemas
- Try-it-out functionality

## Architecture

### System Flow

```
1. Upload → Extract Text → Chunk → Embed → Store in Vector DB
2. Question → Embed → Vector Search → Retrieve Context → LLM → Answer
```

### Tech Stack

- **FastAPI**: Modern Python web framework
- **PostgreSQL + pgvector**: Vector database for semantic search
- **OpenAI**: Embeddings (text-embedding-3-small) and LLM (gpt-4o-mini)
- **PyMuPDF**: PDF text extraction
- **Docker Compose**: Local development and deployment

### Components

| Component | Purpose |
|-----------|---------|
| `app/routers/` | API endpoints (documents, ask) |
| `app/services/` | Business logic (ingest, embed, retrieve, generate) |
| `app/db/` | Database session and schema |
| `app/utils/` | Helpers (auth, chunking, parsing) |
| `app/models/` | Pydantic schemas |

## Configuration

Environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `API_KEY` | API authentication key | `dev-key` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://postgres:postgres@db:5432/kb` |
| `MAX_UPLOAD_SIZE_MB` | Maximum file size | `10` |
| `DEFAULT_TOP_K` | Default chunks to retrieve | `6` |
| `CHUNK_SIZE` | Tokens per chunk | `500` |
| `CHUNK_OVERLAP` | Token overlap between chunks | `100` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | OpenAI LLM model | `gpt-4o-mini` |

## Development

### Local Setup (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start only the database
docker compose up db

# Run the API locally
uvicorn app.main:app --reload
```

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest

# With coverage
pytest --cov=app tests/
```

### Database Access

```bash
# Connect to the database
docker compose exec db psql -U postgres -d kb

# Useful queries
SELECT doc_id, title, filename FROM documents;
SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id;
```

## Supported File Formats

- ✅ PDF (`.pdf`)
- ✅ Word Documents (`.docx`)
- ✅ Markdown (`.md`)
- ✅ Plain Text (`.txt`)

## Limitations & Future Improvements

**Current Limitations:**
- API key authentication only (no OAuth/JWT)
- No rate limiting
- No query caching
- Basic chunking strategy (character-based)
- No reranking of search results

**Potential Enhancements:**
- Add support for HTML, CSV, Excel files
- Implement streaming responses (SSE)
- Add query feedback mechanism
- Implement caching for repeated queries
- Add cross-encoder reranking
- Support for local LLMs (Ollama, LlamaCPP)
- Multi-tenancy with user accounts
- Advanced chunking (semantic, recursive)
- Metadata filtering (tags, dates)

## Troubleshooting

**"Database connection failed"**
- Ensure PostgreSQL container is running: `docker compose ps`
- Check logs: `docker compose logs db`

**"Invalid API key"**
- Verify `OPENAI_API_KEY` in `.env` is correct
- Check you have credits: https://platform.openai.com/account/usage

**"No relevant documents found"**
- Upload documents first via `POST /documents`
- Verify documents processed: `GET /documents/{doc_id}`

**"File too large"**
- Increase `MAX_UPLOAD_SIZE_MB` in `.env`
- Or split the document into smaller files

## License

MIT License - feel free to use this for your projects!

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Check the [API documentation](http://localhost:8000/docs)
- Review the troubleshooting section above
- Open an issue on GitHub
