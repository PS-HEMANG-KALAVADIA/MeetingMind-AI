"""
chat.py - Terminal Chat Interface for MeetingMind AI

This is a simple command-line interface for testing the RAG pipeline
without needing the Streamlit UI. Great for debugging and development.

Usage:
    python chat.py

It will:
1. Ask for a transcript file path
2. Ingest the transcript
3. Let you ask questions in a loop
4. Display answers with source evidence
"""

import sys
import os

from config import validate_config
from ingest import ingest_transcript
from services import vector_store
from services.rag_service import ask_question


def main():
    """Main function for the terminal chat interface."""
    print("\n" + "=" * 60)
    print("🧠 MeetingMind AI - Terminal Chat")
    print("=" * 60)

    # Validate configuration before starting
    if not validate_config():
        print("\n❌ Please set up your .env file and try again.")
        sys.exit(1)

    # Get transcript file path from user
    print("\nEnter the path to a transcript file (.txt):")
    file_path = input(">>> ").strip()

    # Remove quotes if user wraps path in them
    file_path = file_path.strip("\"'")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\n❌ File not found: {file_path}")
        sys.exit(1)

    # Ingest the transcript
    print("\n📥 Ingesting transcript...")
    
    def cli_progress(step: str, message: str) -> None:
        print(f"   {message}")

    insights = ingest_transcript(file_path, progress_callback=cli_progress)

    # Display generated insights
    _display_insights(insights)

    # Get the meeting name for querying
    meeting_name = insights.meeting_name

    # Initialize vector store for chat
    vector_store.initialize()

    # Start the chat loop
    print("\n" + "=" * 60)
    print("💬 Chat with your meeting!")
    print("   Type your questions below.")
    print("   Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        print("\n")
        question = input("You: ").strip()

        # Check for exit commands
        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        # Skip empty questions
        if not question:
            print("Please enter a question.")
            continue

        # Get the answer using RAG
        result = ask_question(question, meeting_name)

        # Display the answer
        print(f"\n🤖 MeetingMind AI:")
        print(result["answer"])

        # Display sources
        if result["sources"]:
            print(f"\n📄 Sources ({len(result['sources'])} chunks used):")
            print("-" * 40)
            for chunk in result["sources"]:
                text_preview = chunk.text[:100].replace('\n', ' ').strip()
                print(f"  ✓ Chunk {chunk.chunk_number + 1}: {text_preview}...")


def _display_insights(insights):
    """Display meeting insights in a formatted way."""
    print("\n" + "=" * 60)
    print("📊 MEETING INSIGHTS")
    print("=" * 60)

    sections = [
        ("📋 Executive Summary", insights.summary),
        ("✅ Key Decisions", insights.decisions),
        ("📌 Action Items", insights.action_items),
        ("⏰ Deadlines", insights.deadlines),
        ("⚠️ Risks", insights.risks),
        ("❓ Open Questions", insights.open_questions),
        ("👥 Participants", insights.participants),
        ("📑 Topics Discussed", insights.topics),
    ]

    for title, content in sections:
        print(f"\n{title}")
        print("-" * 40)
        print(content if content else "No information available.")
        print()


if __name__ == "__main__":
    main()
