"""
services/vector_store.py - ChromaDB Vector Store Manager

This module handles all interactions with ChromaDB:
- Initializing the persistent database
- Generating embeddings using Sentence Transformers
- Storing transcript chunks with metadata
- Performing similarity search for retrieval
- Checking for duplicate transcripts

Why ChromaDB?
- Simple API, great for learning
- Persistent storage (data survives app restarts)
- Built-in embedding function support
- No external server needed (runs in-process)
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from models.document import TranscriptChunk


# ============================================================
# Module-level variables (initialized once, reused everywhere)
# ============================================================

# These are set up when initialize() is called
_client = None
_collection = None
_embedding_function = None


def initialize() -> None:
    """
    Initialize the ChromaDB client, embedding function, and collection.

    This should be called once when the application starts.
    It creates or loads the persistent database and collection.

    Why persistent storage?
    - Data survives app restarts
    - No need to re-embed transcripts every time
    - Supports accumulating multiple meetings over time
    """
    global _client, _collection, _embedding_function

    # Create the embedding function using Sentence Transformers
    # This model converts text into 384-dimensional vectors
    _embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Create a persistent ChromaDB client (stores data on disk)
    _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Get or create the collection (like a "table" in a database)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function
    )

    print(f"✅ ChromaDB initialized at: {CHROMA_DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME} ({_collection.count()} documents)")


def add_chunks(chunks: list[TranscriptChunk]) -> None:
    """
    Add transcript chunks to ChromaDB.

    Each chunk gets:
    - A unique ID (meeting_name + chunk_number)
    - The chunk text (for embedding and retrieval)
    - Metadata (meeting name, chunk number, etc.)

    ChromaDB automatically generates embeddings using our
    SentenceTransformerEmbeddingFunction.

    Args:
        chunks: List of TranscriptChunk objects to store
    """
    if not _collection:
        raise RuntimeError("Vector store not initialized. Call initialize() first.")

    if not chunks:
        print("⚠️ No chunks to add")
        return

    # Prepare data in the format ChromaDB expects
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # Create a unique ID for each chunk
        # Format: "meeting_name_chunk_0", "meeting_name_chunk_1", etc.
        chunk_id = f"{chunk.meeting_name}_chunk_{chunk.chunk_number}"
        ids.append(chunk_id)
        documents.append(chunk.text)
        metadatas.append(chunk.metadata)

    # Add all chunks to ChromaDB in one batch (more efficient)
    _collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✅ Added {len(chunks)} chunks for meeting: '{chunks[0].meeting_name}'")


def search(query: str, meeting_name: str = None, k: int = 5) -> list[TranscriptChunk]:
    """
    Search ChromaDB for chunks similar to the query.

    Uses cosine similarity between the query embedding and stored
    chunk embeddings to find the most relevant pieces of transcript.

    Args:
        query: The user's question or search text
        meeting_name: Optional filter to search within a specific meeting
        k: Number of results to return (default: 5)

    Returns:
        List of TranscriptChunk objects ranked by relevance
    """
    if not _collection:
        raise RuntimeError("Vector store not initialized. Call initialize() first.")

    # Build the where filter for meeting-specific search
    where_filter = None
    if meeting_name:
        where_filter = {"meeting_name": meeting_name}

    # Perform similarity search
    results = _collection.query(
        query_texts=[query],
        n_results=k,
        where=where_filter
    )

    # Convert ChromaDB results back to TranscriptChunk objects
    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            chunk = TranscriptChunk(
                text=doc_text,
                chunk_number=metadata.get("chunk_number", 0),
                meeting_name=metadata.get("meeting_name", ""),
                metadata=metadata
            )
            chunks.append(chunk)

    print(f"🔍 Found {len(chunks)} relevant chunks for query: '{query[:50]}...'")
    return chunks


def is_meeting_ingested(meeting_name: str) -> bool:
    """
    Check if a meeting has already been ingested into ChromaDB.

    This prevents duplicate ingestion — if a user uploads the same
    transcript twice, we skip the re-ingestion.

    Args:
        meeting_name: The name of the meeting to check

    Returns:
        True if the meeting already exists in the database
    """
    if not _collection:
        raise RuntimeError("Vector store not initialized. Call initialize() first.")

    # Query for any documents with this meeting name
    results = _collection.get(
        where={"meeting_name": meeting_name},
        limit=1  # We only need to know if at least one exists
    )

    exists = len(results["ids"]) > 0

    if exists:
        print(f"ℹ️ Meeting '{meeting_name}' is already in the database")
    else:
        print(f"ℹ️ Meeting '{meeting_name}' is new — will be ingested")

    return exists


def get_collection_count() -> int:
    """Return the total number of documents in the collection."""
    if not _collection:
        return 0
    return _collection.count()
