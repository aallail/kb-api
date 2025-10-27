"""Document management endpoints."""
import os
import shutil
import time
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from typing import Optional
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import DocumentUploadResponse, DocumentMetadata
from app.utils.security import require_password
from app.services.ingest import ingest_document, delete_document, get_document_metadata
from app.services.deduplication import compute_file_hash, check_duplicate
from app.services.analytics import log_upload
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Ensure data directory exists
# Use relative path for local dev, absolute for Docker
DATA_DIR = Path("/data") if os.path.exists("/data") and os.access("/data", os.W_OK) else Path("./data")
DATA_DIR.mkdir(exist_ok=True)


@router.post(
    "/",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document"
)
@limiter.limit("10/hour")  # Limit uploads to 10 per hour
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    _: str = Depends(require_password)
):
    """
    Upload and process a document.

    Supported formats:
    - PDF (.pdf)
    - Word (.docx)
    - Markdown (.md)
    - Plain text (.txt)

    The document will be:
    1. Saved to disk
    2. Text extracted
    3. Chunked into smaller pieces
    4. Embedded using OpenAI
    5. Stored in the vector database
    """
    # Validate file size
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Convert to bytes
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    # Validate file extension
    allowed_extensions = {'.pdf', '.docx', '.md', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    start_time = time.time()

    # Read file content for hashing and deduplication
    file_content = await file.read()
    file.file.seek(0)  # Reset for later use

    # Check for duplicates
    file_hash = compute_file_hash(file_content)
    is_duplicate, existing_doc_id = check_duplicate(file_hash)

    if is_duplicate:
        logger.info(f"Duplicate file detected: {file.filename} (matches doc_id: {existing_doc_id})")

        # Get existing document metadata
        existing_doc = get_document_metadata(existing_doc_id)

        # Log analytics (duplicate upload)
        elapsed_ms = (time.time() - start_time) * 1000
        log_upload(
            filename=file.filename,
            file_size=file_size,
            processing_time_ms=elapsed_ms,
            num_chunks=existing_doc.get("chunk_count", 0) if existing_doc else 0,
            is_duplicate=True
        )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "This file has already been uploaded",
                "existing_doc_id": existing_doc_id,
                "filename": existing_doc.get("filename") if existing_doc else file.filename,
                "suggestion": "The file is already in the knowledge base. Use the existing document or delete it first."
            }
        )

    # Save uploaded file
    try:
        # Generate unique filename
        safe_filename = f"{os.urandom(8).hex()}_{file.filename}"
        file_path = DATA_DIR / safe_filename

        with file_path.open("wb") as buffer:
            buffer.write(file_content)

        logger.info(f"Saved uploaded file to {file_path}")

    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )

    # Ingest the document
    try:
        doc_id, num_chunks = ingest_document(
            file_path=str(file_path),
            filename=file.filename,
            mime_type=file.content_type,
            title=title,
            file_hash=file_hash  # Pass hash for storage
        )

        # Log analytics (successful upload)
        elapsed_ms = (time.time() - start_time) * 1000
        log_upload(
            filename=file.filename,
            file_size=file_size,
            processing_time_ms=elapsed_ms,
            num_chunks=num_chunks,
            is_duplicate=False
        )

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunks=num_chunks,
            status="processed",
            processing_time_ms=elapsed_ms,
            uploaded_at=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get(
    "/{doc_id}",
    response_model=DocumentMetadata,
    summary="Get document metadata"
)
def get_document(
    doc_id: str,
    _: str = Depends(require_password)
):
    """
    Retrieve metadata for a specific document.
    """
    metadata = get_document_metadata(doc_id)

    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found"
        )

    return DocumentMetadata(**metadata)


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document"
)
def remove_document(
    doc_id: str,
    _: str = Depends(require_password)
):
    """
    Delete a document and all its associated chunks from the system.
    """
    deleted = delete_document(doc_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found"
        )

    return None
