"""
config.py - Central Configuration for MeetingMind AI

This file loads all environment variables and defines constants
used throughout the application. Every other module imports from here
instead of reading .env directly.

Why this file exists:
- Single source of truth for all configuration
- Easy to change settings without touching business logic
- Keeps API keys out of source code
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# API Configuration
# ============================================================

# Groq API key for LLM access (required)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM model name hosted on Groq
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# ============================================================
# Embedding Configuration
# ============================================================

# Sentence Transformer model for generating embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ============================================================
# ChromaDB Configuration
# ============================================================

# Name of the ChromaDB collection to store transcript chunks
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "meeting_transcripts")

# ============================================================
# Path Configuration
# ============================================================

# Base directory of the project (where this file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for raw transcript files
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Directory for cached meeting insights
INSIGHTS_DIR = os.path.join(BASE_DIR, "data", "insights")

# Directory for ChromaDB persistent storage
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ============================================================
# Text Splitting Configuration
# ============================================================

# Size of each text chunk (in characters)
CHUNK_SIZE = 800

# Overlap between consecutive chunks (helps maintain context)
CHUNK_OVERLAP = 100

# ============================================================
# Retrieval Configuration
# ============================================================

# Number of chunks to retrieve for each query
TOP_K_RESULTS = 3

# ============================================================
# Validate critical configuration
# ============================================================

def validate_config() -> bool:
    """
    Check that required configuration values are set.
    Returns True if valid, False otherwise.
    """
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY is not set in .env file")
        print("   Get your free API key at: https://console.groq.com")
        return False
    return True
