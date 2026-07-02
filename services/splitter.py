"""
services/splitter.py - Transcript Text Splitter

This module splits a full transcript into smaller chunks for embedding
and storage in ChromaDB. We use LangChain's RecursiveCharacterTextSplitter
because it intelligently splits on natural boundaries (paragraphs, sentences).

Why split text?
- LLMs have token limits — we can't send the entire transcript at once
- Smaller chunks give more precise retrieval results
- Embeddings work better on focused, smaller pieces of text
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from models.document import TranscriptDocument, TranscriptChunk


def split_document(document: TranscriptDocument) -> list[TranscriptChunk]:
    """
    Split a TranscriptDocument into a list of TranscriptChunks.

    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Spaces
    4. Characters (last resort)

    This preserves natural text boundaries as much as possible.

    Args:
        document: The full transcript to split

    Returns:
        List of TranscriptChunk objects ready for embedding
    """
    # Initialize the splitter with our configured chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # Count characters (simple and predictable)
        separators=["\n\n", "\n", " ", ""]  # Try these separators in order
    )

    # Split the text into raw string chunks
    raw_chunks = splitter.split_text(document.text)

    # Convert raw strings into TranscriptChunk objects with metadata
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk = TranscriptChunk(
            text=chunk_text,
            chunk_number=i,
            meeting_name=document.meeting_name,
            metadata={
                "meeting_name": document.meeting_name,
                "chunk_number": i,
                "total_chunks": len(raw_chunks),
                "source_file": document.file_path
            }
        )
        chunks.append(chunk)

    print(f"✅ Split transcript into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks
