"""Document deduplication using content hashing."""
import hashlib
import logging
from sqlalchemy import text
from app.db.session import engine

logger = logging.getLogger(__name__)


def compute_file_hash(file_content: bytes) -> str:
    """
    Compute SHA256 hash of file content.

    Args:
        file_content: Raw file bytes

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(file_content).hexdigest()


def check_duplicate(file_hash: str) -> tuple[bool, str | None]:
    """
    Check if file hash already exists in database.

    Args:
        file_hash: SHA256 hash of file content

    Returns:
        Tuple of (is_duplicate, existing_doc_id)
    """
    try:
        # Note: We'll add file_hash column to documents table
        sql = text("""
            SELECT doc_id
            FROM documents
            WHERE file_hash = :file_hash
            LIMIT 1
        """)

        with engine.begin() as conn:
            result = conn.execute(sql, {"file_hash": file_hash})
            row = result.first()

            if row:
                doc_id = str(row[0])
                logger.info(f"Duplicate detected: hash={file_hash[:16]}..., existing doc_id={doc_id}")
                return True, doc_id
            else:
                logger.debug(f"No duplicate found for hash={file_hash[:16]}...")
                return False, None

    except Exception as e:
        logger.error(f"Error checking for duplicates: {e}")
        # If there's an error (e.g., column doesn't exist yet), assume not duplicate
        return False, None


def store_file_hash(doc_id: str, file_hash: str) -> None:
    """
    Store file hash for a document.

    Args:
        doc_id: Document ID
        file_hash: SHA256 hash of file content
    """
    try:
        sql = text("""
            UPDATE documents
            SET file_hash = :file_hash
            WHERE doc_id = :doc_id
        """)

        with engine.begin() as conn:
            conn.execute(sql, {"doc_id": doc_id, "file_hash": file_hash})
            logger.debug(f"Stored file hash for doc_id={doc_id}")

    except Exception as e:
        logger.warning(f"Could not store file hash: {e}")
        # Non-critical error, continue
