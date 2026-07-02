#  MeetingMind AI — Intelligent Meeting Intelligence Dashboard

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3.11-green.svg)](https://github.com/langchain-ai/langchain)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-blueviolet.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, production-grade meeting analytics dashboard that extracts structured insights and lets you chat with your meeting transcripts using Retrieval-Augmented Generation (RAG). Powered by Llama 3.3 (hosted on Groq's high-speed LPU inference engine), ChromaDB, and Sentence Transformers.

---

##  Project Overview

MeetingMind AI solves the problem of information overload from corporate and sprint meetings. By importing standard meeting transcripts, it instantly compiles executive summaries, action items (with owners and deadlines), and risk reports in a single dashboard. Users can converse naturally with the meeting contents, retrieving exact statements and decisions with citation markers highlighting the origin of each answer.

---

##  System Architecture

Below is the conceptual flow of the MeetingMind AI pipeline, tracing how transcripts are processed, cached, embedded, and queried.

```text
                                 +-----------------------+
                                 |   User Transcript     |
                                 +-----------+-----------+
                                             |
                                             v
                                 +-----------+-----------+
                                 |  Streamlit Frontend   |
                                 +-----------+-----------+
                                             |
                                             | (Upload File)
                                             v
                                 +-----------+-----------+
                                 |   ingest_transcript   |
                                 +-----------+-----------+
                                             |
                                             | (Load txt file)
                                             v
                                 +-----------+-----------+
                                 |     loader.py         |
                                 +-----------+-----------+
                      _______________________|_______________________
                     |                                               |
                     | (New Meeting)                                 | (Duplicate/Insights Only)
                     v                                               v
         +-----------+-----------+                       +-----------+-----------+
         |     splitter.py       |                       |   meeting_analyzer.py |
         |   (Semantic Chunking) |                       | (Single API Call JSON)|
         +-----------+-----------+                       +-----------+-----------+
                     |                                               |
                     | (Chunks list)                                 | (Cache JSON/disk)
                     v                                               v
         +-----------+-----------+                       +-----------+-----------+
         |   vector_store.py     |                       |    data/insights/     |
         | (Embeddings & Chroma) |                       +-----------------------+
         +-----------+-----------+
                     |
                     | (Add chunks)
                     v
         +-----------+-----------+
         |   ChromaDB Directory  |
         +-----------------------+
                     ^
                     | (Query / Similarity Search)
                     v
         +-----------+-----------+
         |     retriever.py      |
         +-----------+-----------+
                     ^
                     | (Top-k context chunks)
                     v
         +-----------+-----------+
         |   prompt_builder.py   |
         |  (Context + Prompt)   |
         +-----------+-----------+
                     |
                     v
         +-----------+-----------+
         |     llm_service.py    |
         |     (Groq Llama 3)    |
         +-----------+-----------+
                     |
                     v
         +-----------+-----------+
         |     Streamlit Chat    |
         |   (Formatted Answer)  |
         +-----------------------+
```

---

##  Features

- ** Live Ingestion Loader**: A multi-step visual status tracker (`st.status`) providing transparency as transcripts load, split, embed, store, and analyze.
- ** Auto-Generated Executive Insights**: Compiles meeting topics, participants, action items, deadlines, decisions, and risks in expandable UI cards.
- ** Disk Cache Optimization**: Insights are stored locally as JSON. Re-uploading or viewing the same meeting loads instantly, avoiding repeated API costs.
- ** Duplicate Detection**: Automatically prevents re-embedding the same transcript twice, protecting the vector database from redundant records.
- ** Structured RAG Chat**: Chat with meetings using context-grounded prompts that format output points (e.g. `• Owner — Task — Deadline`) and append source citations (`✓ Chunk X`).
- ** Resilient Exception Mapping**: Catch-all boundary maps API rate limits, invalid credentials, and offline states to helpful, markdown-styled troubleshooting steps.

---

##  Repository Structure

```text
MeetingMind/
│
├── .vscode/               # Workspace settings (Interpreter bindings)
├── chroma_db/             # Local database directory for ChromaDB embeddings
├── data/
│   ├── raw/               # Drop area for raw meeting transcript text files
│   └── insights/          # Local JSON file cache for generated insights
│
├── models/
│   ├── __init__.py
│   └── document.py        # Typed data structures (TranscriptChunk, MeetingInsights)
│
├── services/
│   ├── __init__.py
│   ├── error_handler.py   # Exception translation logic
│   ├── llm_service.py     # ChatGroq integration and configurations
│   ├── loader.py          # Plaintext loader and parser logic
│   ├── meeting_analyzer.py# Single-call insight structuring and caching
│   ├── prompt_builder.py  # Prompt engineering and system instruction schemas
│   ├── rag_service.py     # Orchestrator for retrieval-generation loop
│   ├── retriever.py       # Query vector cosine similarity selector
│   ├── splitter.py        # RecursiveCharacterTextSplitter chunk builder
│   └── vector_store.py    # Persistent Chroma client and SentenceTransformers embedding setup
│
├── app.py                 # Premium Streamlit Dashboard application
├── chat.py                # Command-Line / Terminal Chat Interface utility
├── ingest.py              # CLI batch ingest runner
├── requirements.txt       # Pinpoint dependency list
├── .env.example           # Secure configuration environment template
└── LICENSE                # MIT License
```

---

##  Tech Stack

- **Front-end / Dashboard**: [Streamlit](https://streamlit.io/) (v1.58.0)
- **Orchestration**: [LangChain](https://github.com/langchain-ai/langchain) (v1.3.11)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (v1.5.9)
- **Local Embeddings**: [Sentence Transformers](https://huggingface.co/sentence-transformers) (`all-MiniLM-L6-v2`)
- **Large Language Model**: [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Core Language**: Python (v3.12)

---

##  Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.12 installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/MeetingMind.git
cd MeetingMind
```

### 3. Create & Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open the `.env` file and input your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

---

##  Usage

### Running the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`. Use the sidebar to upload a `.txt` transcript file, click **Analyze Meeting**, and watch the pipeline progress live!

### Running the Terminal CLI Chat
For testing or terminal debugging:
```bash
python chat.py
```
Input the file path to a raw transcript, and query it directly through the CLI.

### Running Batch Ingestion
To pre-ingest transcripts placed in the `data/raw/` directory:
```bash
python ingest.py
```

---

##  Retrieval-Augmented Generation (RAG) Pipeline

1. **Ingest & Clean**: Raw transcripts are loaded and standardized.
2. **Chunking**: Text is split using LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 800 characters and 100 character overlap to maintain speaker continuity.
3. **Embedding Vectorization**: Each chunk is embedded locally into a 384-dimensional vector space using `SentenceTransformer('all-MiniLM-L6-v2')`.
4. **Storage**: Vectors and chunk text are written to a persistent disk-bound ChromaDB.
5. **Context Retrieve**: The user's query is embedded, and a cosine similarity query retrieves the top $K$ ($K=3$) matching chunks from the index.
6. **Augment & Generate**: A custom system prompt joins the retrieved text blocks with the user's question, sending it to Llama 3.3. The LLM formats answers structurally (e.g. `• Owner — Task — Deadline`), appending a source citation tag `✓ Chunk X` referencing the precise evidence blocks.

---

##  Future Improvements
- **PDF & DOCX Support**: Expand `services/loader.py` to process additional meeting record formats.
- **Multi-Meeting Context**: Enable cross-meeting query scopes to compare timeline changes over time.
- **Audio Processing**: Integrate OpenAI Whisper to accept recorded MP3/WAV files directly.

---

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Author
**Hemang Kalavadiya**
