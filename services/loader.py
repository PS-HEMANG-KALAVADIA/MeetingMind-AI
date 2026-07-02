"""
services/loader.py - Transcript File Loader

This module reads transcript files from disk and converts them into
TranscriptDocument objects. Currently supports .txt files, but is
designed so PDF and DOCX support can be added easily.

Why a separate loader?
- Separates file I/O from business logic
- Easy to add new file formats without changing other modules
- Single responsibility: just read files
"""

import os
from models.document import TranscriptDocument


def load_transcript(file_path: str) -> TranscriptDocument:
    """
    Read a transcript file and return a TranscriptDocument.

    Args:
        file_path: Path to the transcript file (.txt)

    Returns:
        TranscriptDocument with the file's text content

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    # Get the file extension to determine how to read it
    file_extension = os.path.splitext(file_path)[1].lower()

    # Read the file based on its format
    if file_extension == ".txt":
        text = _read_txt_file(file_path)
    # --- Future format support ---
    # elif file_extension == ".pdf":
    #     text = _read_pdf_file(file_path)
    # elif file_extension == ".docx":
    #     text = _read_docx_file(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            f"Currently supported: .txt"
        )

    # Derive a meeting name from the filename (without extension)
    # Example: "team_standup_2024.txt" -> "team_standup_2024"
    meeting_name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"✅ Loaded transcript: {meeting_name} ({len(text)} characters)")

    return TranscriptDocument(
        text=text,
        meeting_name=meeting_name,
        file_path=file_path
    )


def load_transcript_from_bytes(content: bytes, filename: str) -> TranscriptDocument:
    """
    Load a transcript from raw bytes (used for Streamlit file uploads).

    Streamlit's file_uploader gives us bytes, not a file path.
    This function handles that case.

    Args:
        content: Raw bytes of the uploaded file
        filename: Original filename (e.g., "meeting.txt")

    Returns:
        TranscriptDocument with the decoded text content
    """
    # Decode bytes to string (UTF-8 is the standard encoding)
    text = content.decode("utf-8")

    # Derive meeting name from filename
    meeting_name = os.path.splitext(filename)[0]

    print(f"✅ Loaded uploaded transcript: {meeting_name} ({len(text)} characters)")

    return TranscriptDocument(
        text=text,
        meeting_name=meeting_name,
        file_path=filename  # No actual path for uploaded files
    )


def _read_txt_file(file_path: str) -> str:
    """
    Read a plain text file and return its contents.

    This is a private helper function (prefixed with _).
    Each file format gets its own reader function for clarity.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# --- Future format readers ---
# def _read_pdf_file(file_path: str) -> str:
#     """Read a PDF file and return extracted text."""
#     # Use PyPDF2 or pdfplumber here
#     pass

# def _read_docx_file(file_path: str) -> str:
#     """Read a DOCX file and return extracted text."""
#     # Use python-docx here
#     pass
