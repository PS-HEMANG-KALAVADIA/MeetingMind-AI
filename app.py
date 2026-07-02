"""
app.py - Streamlit Web Application for MeetingMind AI

This is the main entry point for the web application.
It provides a polished UI with two tabs:

Tab 1: Meeting Insights
    - Upload a transcript file
    - View AI-generated insights (summary, decisions, action items, etc.)

Tab 2: Meeting Chat
    - Ask natural language questions about the meeting
    - See answers with source evidence

Usage:
    streamlit run app.py
"""

import streamlit as st

from config import validate_config
from ingest import ingest_uploaded_file
from services import vector_store
from services.rag_service import ask_question
from services.error_handler import get_user_friendly_error


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="MeetingMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for Premium Styling
# ============================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.95) 0%, rgba(124, 58, 237, 0.95) 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.8px;
    }
    .main-header p {
        margin: 0.6rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Custom styling for expanders */
    .stExpander {
        background: var(--background-color);
        border: 1px solid rgba(79, 70, 229, 0.15) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 0.8rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stExpander:hover {
        border-color: rgba(79, 70, 229, 0.4) !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.06) !important;
    }
    
    .stExpander p {
        line-height: 1.7;
        font-size: 0.98rem;
    }

    /* Chat message styling */
    .chat-question {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white !important;
        padding: 1.1rem 1.6rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.8rem 0;
        font-weight: 400;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2);
        max-width: 85%;
        margin-left: auto;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .chat-answer {
        background: var(--secondary-background-color);
        border: 1px solid rgba(79, 70, 229, 0.15);
        padding: 1.3rem 1.6rem;
        border-radius: 4px 18px 18px 18px;
        margin: 0.8rem 0;
        color: var(--text-color);
        line-height: 1.75;
        font-size: 1.02rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
        max-width: 85%;
        border-left: 4px solid #4f46e5;
    }

    /* Source chunk styling */
    .source-chunk {
        background: var(--secondary-background-color);
        border-left: 4px solid #7c3aed;
        border-top: 1px solid rgba(79, 70, 229, 0.1);
        border-right: 1px solid rgba(79, 70, 229, 0.1);
        border-bottom: 1px solid rgba(79, 70, 229, 0.1);
        padding: 1rem 1.3rem;
        margin: 0.6rem 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.92rem;
        color: var(--text-color);
        line-height: 1.65;
    }
    .source-label {
        font-weight: 700;
        color: #7c3aed;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.4rem;
        display: inline-block;
    }

    /* Upload area styling */
    .upload-section {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 2px dashed rgba(79, 70, 229, 0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .upload-section:hover {
        border-color: #4f46e5;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.08);
    }
    .upload-section h3 {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    .upload-section p {
        font-size: 0.95rem;
        color: var(--text-color);
        opacity: 0.85;
        margin-bottom: 0.5rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 1rem;
    }
    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-info {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid rgba(79, 70, 229, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        border: none !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #4f46e5 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
        border-bottom: 2px solid #4f46e5 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialization
# ============================================================

def init_session_state():
    """
    Initialize Streamlit session state variables.

    Session state persists data across reruns (Streamlit reruns
    the entire script on every interaction). Without session state,
    we'd lose our insights and chat history on every click.
    """
    if "insights" not in st.session_state:
        st.session_state.insights = None  # MeetingInsights object

    if "meeting_name" not in st.session_state:
        st.session_state.meeting_name = None  # Current meeting name

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List of (question, answer, sources)

    if "is_ingested" not in st.session_state:
        st.session_state.is_ingested = False  # Whether a transcript is loaded

    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False


# ============================================================
# Header
# ============================================================

def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MeetingMind AI</h1>
        <p>AI-Powered Meeting Intelligence — Upload, Analyze, and Chat with Your Meetings</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

def render_sidebar():
    """Render the sidebar with upload and status info."""
    with st.sidebar:
        st.markdown("### 📤 Upload Transcript")
        st.markdown("Upload a `.txt` meeting transcript to get started.")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a transcript file",
            type=["txt"],
            help="Upload a plain text (.txt) meeting transcript",
            key="file_uploader"
        )

        if uploaded_file is not None:
            # Show file info
            st.markdown(f"""
            <div class="status-badge status-info">
                📄 {uploaded_file.name}
            </div>
            """, unsafe_allow_html=True)

            # Process button
            if st.button("🚀 Analyze Meeting", use_container_width=True, type="primary"):
                _process_upload(uploaded_file)

        # Status section
        st.markdown("---")
        st.markdown("### 📊 Status")

        if st.session_state.is_ingested:
            st.markdown(f"""
            <div class="status-badge status-success">
                ✅ Meeting Loaded
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Meeting:** {st.session_state.meeting_name}")
        else:
            st.info("No meeting loaded yet. Upload a transcript to begin.")

        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - What decisions were made?
        - Who is responsible for what?
        - Summarize the main discussion.
        - What deadlines were mentioned?
        - What risks were identified?
        - What tasks were assigned?
        """)

        # App info
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; opacity: 0.6; font-size: 0.8rem;'>"
            "Built with ❤️ using Groq + ChromaDB + Streamlit</p>",
            unsafe_allow_html=True
        )


def _process_upload(uploaded_file):
    """Process an uploaded transcript file using st.status and callbacks."""
    # Read file bytes
    content = uploaded_file.read()
    filename = uploaded_file.name

    with st.status("🚀 Starting Ingestion Pipeline...", expanded=True) as status:
        try:
            def progress_callback(step: str, message: str) -> None:
                status.write(message)

            # Run the ingestion pipeline with progress status
            insights = ingest_uploaded_file(
                content=content,
                filename=filename,
                progress_callback=progress_callback
            )

            # Store results in session state
            st.session_state.insights = insights
            st.session_state.meeting_name = insights.meeting_name
            st.session_state.is_ingested = True
            st.session_state.chat_history = []  # Reset chat for new meeting

            # Initialize vector store for chat
            if not st.session_state.vector_store_initialized:
                status.write("🔌 Connecting to database...")
                vector_store.initialize()
                st.session_state.vector_store_initialized = True

            status.update(label="✅ Ingestion Pipeline Complete!", state="complete", expanded=False)
            st.toast(f"✅ Ingested: '{insights.meeting_name}'")
            st.rerun()

        except Exception as e:
            status.update(label="❌ Ingestion Failed", state="error", expanded=True)
            friendly_err = get_user_friendly_error(e, "processing transcript upload")
            st.error(friendly_err)


# ============================================================
# Tab 1: Meeting Insights
# ============================================================

def render_insights_tab():
    """Render the Meeting Insights tab."""
    if not st.session_state.is_ingested:
        # Show upload prompt when no meeting is loaded
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload a Meeting Transcript</h3>
            <p>Use the sidebar to upload a <strong>.txt</strong> transcript file.</p>
            <p>MeetingMind AI will automatically generate insights including:</p>
            <p>📋 Summary &nbsp;|&nbsp; ✅ Decisions &nbsp;|&nbsp; 📌 Action Items &nbsp;|&nbsp; ⏰ Deadlines</p>
            <p>⚠️ Risks &nbsp;|&nbsp; ❓ Open Questions &nbsp;|&nbsp; 👥 Participants &nbsp;|&nbsp; 📑 Topics</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Display meeting name
    st.markdown(f"### 📁 Meeting: {st.session_state.meeting_name}")
    st.markdown("---")

    insights = st.session_state.insights

    # Define insight sections with their icons and data
    insight_sections = [
        ("📋 Executive Summary", insights.summary),
        ("✅ Key Decisions", insights.decisions),
        ("📌 Action Items", insights.action_items),
        ("⏰ Deadlines", insights.deadlines),
        ("⚠️ Risks & Concerns", insights.risks),
        ("❓ Open Questions", insights.open_questions),
        ("👥 Participants", insights.participants),
        ("📑 Topics Discussed", insights.topics),
    ]

    # Render each insight in an expandable card
    for title, content in insight_sections:
        with st.expander(title, expanded=(title.startswith("📋"))):
            if content and not content.startswith("⚠️"):
                st.markdown(content)
            else:
                st.info("No information available for this section.")


# ============================================================
# Tab 2: Meeting Chat
# ============================================================

def render_chat_tab():
    """Render the Meeting Chat tab."""
    if not st.session_state.is_ingested:
        st.info("💡 Upload a meeting transcript first to start chatting.")
        return

    st.markdown(f"### 💬 Chat with: {st.session_state.meeting_name}")
    st.markdown("Ask any question about your meeting and get answers backed by transcript evidence.")
    st.markdown("---")

    # Display chat history
    for q, a, sources in st.session_state.chat_history:
        # User question
        st.markdown(f'<div class="chat-question">🙋 {q}</div>', unsafe_allow_html=True)

        # AI answer
        st.markdown(f'<div class="chat-answer">🤖 {a}</div>', unsafe_allow_html=True)

        # Source evidence
        if sources:
            with st.expander(f"📄 View Sources ({len(sources)} chunks)", expanded=False):
                for chunk in sources:
                    st.markdown(
                        f'<div class="source-chunk">'
                        f'<span class="source-label">Chunk {chunk.chunk_number + 1}</span><br>'
                        f'{chunk.text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        st.markdown("")  # Spacing

    # Question input
    question = st.chat_input(
        "Ask a question about your meeting...",
        key="chat_input"
    )

    if question:
        _handle_question(question)


def _handle_question(question: str):
    """Process a user question, display answers and reference sources."""
    # Ensure vector store is initialized
    if not st.session_state.vector_store_initialized:
        with st.spinner("🔌 Initializing connection to database..."):
            vector_store.initialize()
            st.session_state.vector_store_initialized = True

    # Show a toast when beginning execution
    st.toast("🤔 Analyzing transcript for answers...")

    with st.spinner("🔍 Retrieving context and generating answer..."):
        try:
            # Get answer using RAG
            result = ask_question(question, st.session_state.meeting_name)

            # Add to chat history
            st.session_state.chat_history.append(
                (question, result["answer"], result["sources"])
            )

            st.toast("✅ Answer generated!")
            # Rerun to display the new message
            st.rerun()

        except Exception as e:
            friendly_err = get_user_friendly_error(e, "generating an answer")
            st.error(friendly_err)


# ============================================================
# Main Application
# ============================================================

def main():
    """Main entry point for the Streamlit application."""
    # Initialize session state
    init_session_state()

    # Validate configuration
    if not validate_config():
        st.error(
            "❌ **GROQ_API_KEY is not set!**\n\n"
            "Please add your Groq API key to the `.env` file.\n\n"
            "Get a free key at: [console.groq.com](https://console.groq.com)"
        )
        st.stop()

    # Render header
    render_header()

    # Render sidebar
    render_sidebar()

    # Create tabs
    tab1, tab2 = st.tabs(["📊 Meeting Insights", "💬 Meeting Chat"])

    with tab1:
        render_insights_tab()

    with tab2:
        render_chat_tab()


# Run the app
if __name__ == "__main__":
    main()
else:
    # Streamlit runs the file directly (not as __main__ when using `streamlit run`)
    main()
