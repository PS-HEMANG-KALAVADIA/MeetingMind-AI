"""
models/document.py - Data Models for MeetingMind AI

This file defines the data structures used throughout the application.
We use Python dataclasses because they are simple, built-in, and
beginner-friendly (no extra libraries needed).

Why dataclasses?
- Auto-generate __init__, __repr__, __eq__
- Clean and readable
- Easy to explain in interviews
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptDocument:
    """
    Represents a complete meeting transcript loaded from a file.

    This is the raw document before any processing (splitting/chunking).
    Think of it as the "whole meeting" in memory.

    Attributes:
        text: The full transcript text content
        meeting_name: A human-readable name for this meeting (derived from filename)
        file_path: Path to the original file on disk
    """
    text: str
    meeting_name: str
    file_path: str


@dataclass
class TranscriptChunk:
    """
    Represents a single chunk/piece of a meeting transcript.

    After a transcript is split into smaller pieces for embedding,
    each piece becomes a TranscriptChunk. This is what gets stored
    in ChromaDB and retrieved during RAG.

    Attributes:
        text: The chunk's text content
        chunk_number: Position of this chunk in the original document (0-indexed)
        meeting_name: Which meeting this chunk belongs to
        metadata: Additional info stored alongside the chunk in ChromaDB
    """
    text: str
    chunk_number: int
    meeting_name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class MeetingInsights:
    """
    Holds all AI-generated insights for a meeting.

    After uploading a transcript, the meeting analyzer generates these
    insights once. They are cached and reused instead of regenerating
    every time the user views them.

    Attributes:
        meeting_name: Which meeting these insights are for
        summary: Executive summary of the meeting
        decisions: Key decisions made during the meeting
        action_items: Tasks and assignments identified
        deadlines: Any deadlines or timelines mentioned
        risks: Risks or concerns raised
        open_questions: Unresolved questions from the meeting
        participants: People who participated (if identifiable)
        topics: Main topics discussed
    """
    meeting_name: str
    summary: str = ""
    decisions: str = ""
    action_items: str = ""
    deadlines: str = ""
    risks: str = ""
    open_questions: str = ""
    participants: str = ""
    topics: str = ""

    @classmethod
    def from_dict(cls, meeting_name: str, data: dict) -> "MeetingInsights":
        """
        Create a MeetingInsights instance from a dictionary of insights.

        Args:
            meeting_name: The name of the meeting
            data: Dictionary containing the key-value pairs matching the fields

        Returns:
            A new MeetingInsights instance populated with the dictionary contents
        """
        return cls(
            meeting_name=meeting_name,
            summary=data.get("summary", ""),
            decisions=data.get("decisions", ""),
            action_items=data.get("action_items", ""),
            deadlines=data.get("deadlines", ""),
            risks=data.get("risks", ""),
            open_questions=data.get("open_questions", ""),
            participants=data.get("participants", ""),
            topics=data.get("topics", "")
        )
