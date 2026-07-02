"""
services/prompt_builder.py - Prompt Construction for LLM

This module builds the prompts that are sent to the LLM.
Good prompts are critical for getting good AI responses.

Why a separate prompt builder?
- Prompts are the most important part of an AI application
- Keeping them in one place makes them easy to iterate on
- Separates prompt engineering from business logic
- Easy to test and improve prompts without changing other code
"""

from models.document import TranscriptChunk


def build_qa_prompt(question: str, context_chunks: list[TranscriptChunk]) -> str:
    """
    Build a prompt for answering questions about a meeting.

    This combines the user's question with retrieved transcript chunks
    to create a grounded prompt. The LLM can only use this context
    to answer — preventing hallucination.

    Args:
        question: The user's natural language question
        context_chunks: Retrieved transcript chunks relevant to the question

    Returns:
        A formatted prompt string ready to send to the LLM
    """
    # Format each chunk with its number for reference
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(
            f"[Chunk {chunk.chunk_number + 1}] "
            f"(Meeting: {chunk.meeting_name})\n"
            f"{chunk.text}"
        )

    # Join all chunks into one context block
    context_text = "\n\n---\n\n".join(context_parts)

    # Build the complete prompt
    prompt = f"""Based on the following meeting transcript excerpts, answer the question.

IMPORTANT RULES:
1. Answer ONLY using the provided transcript context below. Do not extrapolate, assume, or bring in outside knowledge.
2. If the answer is not in the context, say: "I couldn't find that information in the uploaded meeting transcript."
3. Be professional, direct, and concise. Never write one huge dense paragraph. Use clean markdown lists and spacing.
4. Format your response structurally. If your response describes:
   - Summary, start the section with: `📋 Executive Summary`
   - Action Items, start with: `📌 Action Items` and list each item exactly as: `• Owner — Task — Deadline` (use "Unassigned" or "No deadline" if not stated in context)
   - Decisions, start with: `✅ Key Decisions`
   - Deadlines, start with: `⏰ Deadlines`
   - Risks, start with: `⚠ Risks`
   - Participants, start with: `👥 Participants`
   - Topics, start with: `📚 Topics Discussed`
5. Always reference which chunk number(s) your information comes from by appending the citation format `✓ Chunk X` (e.g., "as discussed in ✓ Chunk 2" or "- Action details ✓ Chunk 3").

=== TRANSCRIPT CONTEXT ===
{context_text}
=== END CONTEXT ===

QUESTION: {question}

ANSWER:"""

    return prompt


