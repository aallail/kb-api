"""Document ingestion service."""
import os
import uuid
from pathlib import Path
from typing import Tuple
from sqlalchemy import text
import logging

from app.db.session import engine
from app.utils.parsers import extract_text
from app.utils.chunking import chunk_text
from app.services.embed import embed_texts

logger = logging.getLogger(__name__)


def ingest_document(
    file_path: str,
    filename: str,
    mime_type: str = None,
    title: str = None,
    file_hash: str = None
) -> Tuple[str, int]:
    """
    Ingest a document: extract text, chunk, embed, and store.

    Args:
        file_path: Path to the uploaded file
        filename: Original filename
        mime_type: MIME type of the file
        title: Optional document title
        file_hash: SHA256 hash of file content for deduplication

    Returns:
        Tuple of (doc_id, num_chunks)
    """
    doc_id = str(uuid.uuid4())
    title = title or filename

    logger.info(f"Ingesting document: {filename} (doc_id={doc_id})")

    # 1. Extract text
    try:
        logger.info(f"Extracting text from {filename}...")
        full_text, metadata = extract_text(file_path, mime_type)
        logger.info(f"Extracted {len(full_text)} characters from {filename}")
    except Exception as e:
        logger.error(f"Failed to extract text from {filename}: {e}", exc_info=True)
        raise

    if not full_text.strip():
        raise ValueError("Document contains no extractable text")

    # 2. Chunk the text
    logger.info(f"Chunking text from {filename}...")
    chunks = chunk_text(full_text)
    logger.info(f"Split document into {len(chunks)} chunks")

    if not chunks:
        raise ValueError("No chunks created from document")

    # 3. Generate embeddings
    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embed_texts(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        raise

    # 4. Store in database
    try:
        logger.info(f"Storing {len(chunks)} chunks in database...")

        # Get file size
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        with engine.begin() as conn:
            # Insert document metadata with hash and file size
            conn.execute(
                text("""
                    INSERT INTO documents (doc_id, title, filename, mime, path, status, file_hash, file_size, chunk_count)
                    VALUES (:doc_id, :title, :filename, :mime, :path, :status, :file_hash, :file_size, :chunk_count)
                """),
                {
                    "doc_id": doc_id,
                    "title": title,
                    "filename": filename,
                    "mime": mime_type,
                    "path": file_path,
                    "status": "processed",
                    "file_hash": file_hash,
                    "file_size": file_size,
                    "chunk_count": len(chunks)
                }
            )

            # Insert chunks with embeddings
            for i, (chunk_text_content, embedding) in enumerate(zip(chunks, embeddings)):
                # Extract page number from metadata if available
                page_num = None
                if metadata.get("page_metadata"):
                    # Simple heuristic: assign page based on character position
                    # In production, you'd want more sophisticated page tracking
                    page_num = 1  # Default to page 1 for now

                conn.execute(
                    text("""
                        INSERT INTO chunks (doc_id, chunk_id, page, text, embedding)
                        VALUES (:doc_id, :chunk_id, :page, :text, :embedding)
                    """),
                    {
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "page": page_num,
                        "text": chunk_text_content,
                        "embedding": embedding
                    }
                )

        logger.info(f"✓ Successfully ingested document {doc_id} with {len(chunks)} chunks")
        return doc_id, len(chunks)

    except Exception as e:
        logger.error(f"Database error during ingestion: {e}", exc_info=True)
        # Clean up file on database error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file {file_path} after database error")
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up file {file_path}: {cleanup_err}")
        raise


def delete_document(doc_id: str) -> bool:
    """
    Delete a document and all its chunks.

    Args:
        doc_id: The document ID to delete

    Returns:
        True if deleted, False if not found
    """
    try:
        with engine.begin() as conn:
            # Get file path before deleting
            result = conn.execute(
                text("SELECT path FROM documents WHERE doc_id = :doc_id"),
                {"doc_id": doc_id}
            ).fetchone()

            if not result:
                return False

            file_path = result[0]

            # Delete from database (chunks will cascade)
            conn.execute(
                text("DELETE FROM documents WHERE doc_id = :doc_id"),
                {"doc_id": doc_id}
            )

            # Delete file from disk
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

        logger.info(f"Deleted document {doc_id}")
        return True

    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise


def get_document_metadata(doc_id: str) -> dict:
    """
    Get metadata for a document.

    Args:
        doc_id: The document ID

    Returns:
        Document metadata dict or None if not found
    """
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    SELECT doc_id, title, filename, mime, status, created_at,
                           (SELECT COUNT(*) FROM chunks WHERE doc_id = :doc_id) as chunk_count
                    FROM documents
                    WHERE doc_id = :doc_id
                """),
                {"doc_id": doc_id}
            ).mappings().fetchone()

            if result:
                result_dict = dict(result)
                # Convert UUID to string for Pydantic validation
                result_dict['doc_id'] = str(result_dict['doc_id'])
                return result_dict
            return None

    except Exception as e:
        logger.error(f"Error fetching document metadata: {e}")
        raise
