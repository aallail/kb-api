"""Document parsing utilities for various file formats."""
import fitz  # PyMuPDF
from docx import Document
import markdown
from bs4 import BeautifulSoup
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> Tuple[str, List[dict]]:
    """
    Extract text from PDF file with page metadata.

    Args:
        file_path: Path to the PDF file

    Returns:
        Tuple of (full_text, page_metadata)
        page_metadata contains {page_num, text, char_count}
    """
    doc = fitz.open(file_path)
    full_text = []
    page_metadata = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text.append(text)
        page_metadata.append({
            "page": page_num,
            "text": text,
            "char_count": len(text)
        })

    doc.close()
    return "\n\n".join(full_text), page_metadata


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX file.

    Args:
        file_path: Path to the DOCX file

    Returns:
        Extracted text
    """
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_markdown(file_path: str) -> str:
    """
    Extract text from Markdown file.

    Args:
        file_path: Path to the Markdown file

    Returns:
        Extracted text (converted to plain text)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Convert markdown to HTML then to plain text
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from plain text file.

    Args:
        file_path: Path to the text file

    Returns:
        File contents
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def extract_text(file_path: str, mime_type: str = None) -> Tuple[str, dict]:
    """
    Extract text from a file based on its type.

    Args:
        file_path: Path to the file
        mime_type: MIME type of the file

    Returns:
        Tuple of (extracted_text, metadata)
    """
    metadata = {"pages": None, "extraction_method": None}

    # Determine file type
    if mime_type:
        file_type = mime_type.lower()
    else:
        file_type = file_path.lower()

    try:
        if 'pdf' in file_type:
            text, page_meta = extract_text_from_pdf(file_path)
            metadata["pages"] = len(page_meta)
            metadata["extraction_method"] = "pymupdf"
            metadata["page_metadata"] = page_meta
            return text, metadata

        elif 'word' in file_type or file_type.endswith('.docx'):
            text = extract_text_from_docx(file_path)
            metadata["extraction_method"] = "python-docx"
            return text, metadata

        elif 'markdown' in file_type or file_type.endswith('.md'):
            text = extract_text_from_markdown(file_path)
            metadata["extraction_method"] = "markdown"
            return text, metadata

        else:  # Treat as plain text
            text = extract_text_from_txt(file_path)
            metadata["extraction_method"] = "plain_text"
            return text, metadata

    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        raise ValueError(f"Failed to extract text: {str(e)}")
