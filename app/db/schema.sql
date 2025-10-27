-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: stores metadata about uploaded documents
CREATE TABLE IF NOT EXISTS documents (
    doc_id       UUID PRIMARY KEY,
    title        TEXT,
    filename     TEXT,
    mime         TEXT,
    path         TEXT,
    status       TEXT DEFAULT 'processing',
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now(),
    tags         TEXT[],  -- For categorization/filtering
    category     VARCHAR(100),  -- Document category
    file_size    BIGINT,  -- Size in bytes
    chunk_count  INT DEFAULT 0  -- Number of chunks
);

-- Chunks table: stores document chunks with embeddings
-- Using 768 dimensions for all-mpnet-base-v2 (local model - upgraded from 384d)
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id    INT,
    page        INT,
    text        TEXT,
    embedding   VECTOR(768),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embed ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category) WHERE category IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING GIN(tags);  -- For tag searching
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(doc_id, page);  -- For page-based retrieval

-- Full-text search index on chunk text (optional - can be heavy)
-- CREATE INDEX IF NOT EXISTS idx_chunks_text_fts ON chunks USING GIN(to_tsvector('english', text));

-- Performance optimization: analyze tables after bulk inserts
-- Run: ANALYZE documents; ANALYZE chunks;
