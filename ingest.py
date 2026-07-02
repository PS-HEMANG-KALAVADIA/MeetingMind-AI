"""
ingest.py - Transcript Ingestion Pipeline

This module orchestrates the complete ingestion process:
1. Load the transcript file
2. Check for duplicates
3. Split into chunks
4. Store in ChromaDB
5. Generate meeting insights

This is the "glue" that connects all the services together
for the ingestion workflow.

Usage:
    from ingest import ingest_transcript
    insights = ingest_transcript("path/to/meeting.txt")
"""

from models.document import MeetingInsights, TranscriptDocument
from services.loader import load_transcript, load_transcript_from_bytes
from services.splitter import split_document
from services import vector_store
from services.meeting_analyzer import analyze_meeting
import os


def ingest_transcript(file_path: str, progress_callback=None) -> MeetingInsights:
    """
    Ingest a transcript file from disk.

    Complete pipeline:
    1. Load file → TranscriptDocument
    2. Check if already ingested (skip if duplicate)
    3. Split into chunks
    4. Store chunks in ChromaDB (with embeddings)
    5. Analyze meeting → MeetingInsights

    Args:
        file_path: Path to the transcript file
        progress_callback: Optional callback to notify the caller of progress steps

    Returns:
        MeetingInsights object with all generated insights
    """
    print("\n" + "=" * 60)
    print("🚀 Starting Transcript Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load the transcript
    msg = "📂 Loading transcript from disk..."
    if progress_callback:
        progress_callback("load", msg)
    print(f"\n{msg}")
    document = load_transcript(file_path)

    # Step 2-5: Process the document
    return _process_document(document, progress_callback)


def ingest_uploaded_file(content: bytes, filename: str, progress_callback=None) -> MeetingInsights:
    """
    Ingest a transcript from a Streamlit file upload.

    Same pipeline as ingest_transcript, but starts from bytes
    instead of a file path.

    Args:
        content: Raw bytes of the uploaded file
        filename: Original filename
        progress_callback: Optional callback to notify the caller of progress steps

    Returns:
        MeetingInsights object with all generated insights
    """
    print("\n" + "=" * 60)
    print("🚀 Starting Transcript Ingestion Pipeline (Upload)")
    print("=" * 60)

    # Step 1: Load from bytes
    msg = "📂 Decoding uploaded transcript file..."
    if progress_callback:
        progress_callback("load", msg)
    print(f"\n{msg}")
    document = load_transcript_from_bytes(content, filename)

    # Step 2-5: Process the document
    return _process_document(document, progress_callback)


def _process_document(document: TranscriptDocument, progress_callback=None) -> MeetingInsights:
    """
    Internal function that processes a TranscriptDocument through
    the ingestion pipeline (steps 2-5).

    This is shared between file-based and upload-based ingestion.
    """
    # Step 2: Initialize vector store and check for duplicates
    msg = "🗄️ Checking for duplicate meetings..."
    if progress_callback:
        progress_callback("duplicate_check", msg)
    print(f"\n{msg}")
    vector_store.initialize()

    if vector_store.is_meeting_ingested(document.meeting_name):
        msg_ingested = f"⚠️ Meeting '{document.meeting_name}' already ingested. Skipping chunking."
        if progress_callback:
            progress_callback("duplicate_found", msg_ingested)
        print(msg_ingested)
        
        msg_insights = "🧠 Loading cached insights (or generating fresh)..."
        if progress_callback:
            progress_callback("insights", msg_insights)
        print(msg_insights)
        
        insights = analyze_meeting(document.text, document.meeting_name)
        return insights

    # Step 3: Split into chunks
    msg = "✂️ Splitting transcript into chunks..."
    if progress_callback:
        progress_callback("chunking", msg)
    print(msg)
    chunks = split_document(document)

    # Step 4: Store in ChromaDB
    msg = "💾 Storing chunks in ChromaDB (Generating Embeddings)..."
    if progress_callback:
        progress_callback("storing", msg)
    print(msg)
    vector_store.add_chunks(chunks)

    # Step 5: Generate insights
    msg = "🧠 Analyzing meeting using Groq LLM..."
    if progress_callback:
        progress_callback("insights", msg)
    print(msg)
    insights = analyze_meeting(document.text, document.meeting_name)

    print("\n" + "=" * 60)
    print("✅ Ingestion Pipeline Complete!")
    print(f"   Meeting: {document.meeting_name}")
    print(f"   Chunks stored: {len(chunks)}")
    print(f"   Total documents in DB: {vector_store.get_collection_count()}")
    print("=" * 60 + "\n")

    return insights


# ============================================================
# CLI Entry Point — Run with: python ingest.py
# ============================================================

if __name__ == "__main__":
    """
    When executed directly, this script:
    1. Scans data/raw/ for all .txt transcript files
    2. Ingests each one through the full pipeline
    3. Prints a summary of successes and failures
    """
    import sys
    from config import validate_config, RAW_DATA_DIR

    print("\n" + "=" * 60)
    print("🚀 MeetingMind AI — Batch Transcript Ingestion")
    print("=" * 60)

    # --- Validate configuration before doing any work ---
    if not validate_config():
        print("\n❌ Please set up your .env file and try again.")
        sys.exit(1)

    # --- Discover all .txt transcript files in data/raw/ ---
    transcript_files = [
        os.path.join(RAW_DATA_DIR, f)
        for f in os.listdir(RAW_DATA_DIR)
        if f.lower().endswith(".txt")
    ]

    # Sort alphabetically for consistent ordering
    transcript_files.sort()

    total_found = len(transcript_files)
    print(f"\n📂 Scanning directory: {RAW_DATA_DIR}")
    print(f"   Transcript files found: {total_found}")

    # --- Handle empty directory ---
    if total_found == 0:
        print(f"\nNo transcript files found in data/raw/")
        print("Place one or more .txt files in that folder and re-run.")
        sys.exit(0)

    # --- Process each transcript ---
    success_count = 0
    failed_count = 0
    failed_files = []  # Track which files failed (for the summary)

    for i, file_path in enumerate(transcript_files, start=1):
        filename = os.path.basename(file_path)
        print(f"\n{'─' * 60}")
        print(f"📄 [{i}/{total_found}] Processing: {filename}")
        print(f"{'─' * 60}")

        try:
            ingest_transcript(file_path)
            success_count += 1
            print(f"✅ Successfully processed: {filename}")
        except Exception as e:
            failed_count += 1
            failed_files.append(filename)
            print(f"❌ Failed to process: {filename}")
            print(f"   Error: {str(e)}")
            # Continue processing remaining files — don't stop on failure

    # --- Print final summary ---
    print("\n" + "=" * 40)
    print("  MeetingMind AI Ingestion Summary")
    print("=" * 40)
    print(f"  Files Found            : {total_found}")
    print(f"  Successfully Processed : {success_count}")
    print(f"  Failed                 : {failed_count}")
    print("=" * 40)

    # List any failures so the user knows what to fix
    if failed_files:
        print("\n⚠️ Failed files:")
        for name in failed_files:
            print(f"   - {name}")

    print()  # Final blank line for clean output
