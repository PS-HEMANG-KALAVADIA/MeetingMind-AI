"""
services/rag_service.py - RAG (Retrieval-Augmented Generation) Orchestrator

This module coordinates the full RAG pipeline:
1. Retrieve relevant chunks from ChromaDB
2. Build a grounded prompt with the retrieved context
3. Send the prompt to the LLM
4. Return the answer along with source chunks

Why a separate RAG service?
- Orchestrates multiple services into a clean workflow
- Single function call for the UI to get an answer
- Easy to understand the complete RAG flow in one place
"""

from models.document import TranscriptChunk
from services.retriever import retrieve_context
from services.prompt_builder import build_qa_prompt
from services.llm_service import get_llm_response


def ask_question(question: str, meeting_name: str) -> dict:
    """
    Answer a question about a meeting using RAG.

    This is the main entry point for the chat feature.
    It follows the RAG pattern:
    1. RETRIEVE: Find relevant transcript chunks
    2. AUGMENT: Build a prompt with the retrieved context
    3. GENERATE: Get an answer from the LLM

    Args:
        question: The user's natural language question
        meeting_name: Which meeting to search within

    Returns:
        Dictionary with:
        - "answer": The LLM's response (str)
        - "sources": List of TranscriptChunk objects used as context
    """
    print(f"\n💬 Question: {question}")
    print(f"📁 Meeting: {meeting_name}")
    print("-" * 40)

    # Step 1: RETRIEVE — Find relevant chunks
    print("Step 1: Retrieving relevant context...")
    source_chunks = retrieve_context(
        query=question,
        meeting_name=meeting_name
    )

    # If no relevant chunks found, return a clear message
    if not source_chunks:
        return {
            "answer": "I couldn't find that information in the uploaded meeting transcript.",
            "sources": []
        }

    # Step 2: AUGMENT — Build the prompt with context
    print("Step 2: Building grounded prompt...")
    prompt = build_qa_prompt(question, source_chunks)

    # Step 3: GENERATE — Get the LLM's answer
    print("Step 3: Generating answer...")
    answer = get_llm_response(prompt)

    print("-" * 40)
    print(f"✅ Answer generated ({len(answer)} characters)")

    return {
        "answer": answer,
        "sources": source_chunks
    }
