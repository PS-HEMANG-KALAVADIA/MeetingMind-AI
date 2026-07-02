"""
services/retriever.py - Context Retriever for RAG

This module retrieves relevant transcript chunks from ChromaDB
based on a user's question. It acts as the "R" in RAG
(Retrieval-Augmented Generation).

Why a separate retriever?
- Keeps retrieval logic isolated from the vector store internals
- Easy to add re-ranking, filtering, or other retrieval strategies later
- Clean interface for the RAG service to call
"""

from config import TOP_K_RESULTS
from models.document import TranscriptChunk
from services import vector_store


def retrieve_context(
    query: str,
    meeting_name: str,
    k: int = None
) -> list[TranscriptChunk]:
    """
    Retrieve the most relevant transcript chunks for a query.

    This is the core retrieval step in RAG:
    1. The query is embedded using the same model as the stored chunks
    2. ChromaDB finds the closest chunks by cosine similarity
    3. The top-k chunks are returned as context for the LLM

    Args:
        query: The user's natural language question
        meeting_name: Which meeting to search within
        k: Number of chunks to retrieve (defaults to config.TOP_K_RESULTS)

    Returns:
        List of TranscriptChunk objects, ranked by relevance
    """
    # Use the configured default if k is not specified
    if k is None:
        k = TOP_K_RESULTS

    # Search the vector store for similar chunks
    chunks = vector_store.search(
        query=query,
        meeting_name=meeting_name,
        k=k
    )

    if not chunks:
        print(f"⚠️ No relevant chunks found for: '{query}'")
    else:
        print(f"📄 Retrieved {len(chunks)} chunks for context")

    return chunks
