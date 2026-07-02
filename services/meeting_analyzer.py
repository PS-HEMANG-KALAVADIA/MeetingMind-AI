"""
services/meeting_analyzer.py

Generates all meeting insights using ONE Groq API call.
"""

import os
import json
import re

from models.document import MeetingInsights
from services.llm_service import get_llm_response
from config import INSIGHTS_DIR


ANALYSIS_SYSTEM_PROMPT = """You are MeetingMind AI.

Analyze the meeting transcript carefully and generate meeting insights.

IMPORTANT:
1. Return ONLY valid JSON.
2. Do NOT wrap the JSON in markdown code blocks like ```json or ```.
3. The response must contain exactly this keys schema:
{
    "summary": "Concise executive summary paragraphs, followed by 3-5 main key takeaways in a bulleted list.",
    "decisions": "A bulleted list of all key decisions made. Format: - [Decision details] (by [Owner/Group] if mentioned). If none, state 'No key decisions were recorded.'",
    "action_items": "A bulleted list of action items. Format: - **Owner** — **Task** — **Deadline**. Use 'Unassigned' or 'No deadline' if not specified. If none, state 'No action items were assigned.'",
    "deadlines": "A bulleted list of deadlines. Format: - **Task/Deliverable** — **Due Date/Timeline**. If none, state 'No deadlines were mentioned.'",
    "risks": "A bulleted list of risks, concerns, or blockers. Format: - **Risk/Blocker** — *Mitigation:* [details if discussed]. If none, state 'No significant risks or blockers were identified.'",
    "open_questions": "A bulleted list of unresolved questions or issues. Format: - [Question/Issue] (raised by [Speaker] if mentioned). If none, state 'No unresolved questions were left.'",
    "participants": "A bulleted list of speakers and participants. Format: - **Name/Role** — Focus area or contribution (if mentioned). If none, state 'No participants could be clearly identified.'",
    "topics": "A bulleted list of main topics discussed. Format: - **Topic** — Short description (ordered by discussion time). If none, state 'No topics were identified.'"
}

Formatting Rules for values:
- Use markdown lists (bullet points starting with '-') for lists.
- Avoid large, dense paragraphs; use line breaks between items.
- Ensure all content in the fields is grounded directly in the transcript.
"""


def analyze_meeting(full_text: str, meeting_name: str) -> MeetingInsights:
    """
    Generate complete meeting insights (summary, action items, etc.) in a single LLM call.
    Uses local file caching to prevent duplicate API costs.

    Args:
        full_text: The complete transcript text.
        meeting_name: The name of the meeting.

    Returns:
        MeetingInsights object containing all parsed information.
    """
    print(f"\n🔍 Analyzing meeting: {meeting_name}")
    print("=" * 50)

    insights = MeetingInsights(meeting_name=meeting_name)
    os.makedirs(INSIGHTS_DIR, exist_ok=True)
    cache_file = os.path.join(INSIGHTS_DIR, f"{meeting_name}.json")

    # Load cache if available
    if os.path.exists(cache_file):
        print("📂 Cached insights found.")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            insights = MeetingInsights.from_dict(meeting_name, data)
            print("✅ Loaded insights from cache.")
            print("=" * 50)
            return insights
        except Exception as e:
            print(f"⚠️ Cache invalid: {e}")
            print("Generating fresh insights...")

    print("🤖 Calling Groq API...")
    prompt = f"Analyze this meeting transcript.\n\nReturn ONLY JSON.\n\nTranscript:\n\n{full_text}"

    try:
        response = get_llm_response(
            prompt,
            system_prompt=ANALYSIS_SYSTEM_PROMPT
        )
        print("✅ Groq response received.")

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response.")

        data = json.loads(match.group())
        insights = MeetingInsights.from_dict(meeting_name, data)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("💾 Insights cached successfully.")
        print("=" * 50)
        return insights

    except Exception as e:
        print(f"\n❌ Meeting analysis failed: {e}")
        insights.summary = "⚠️ Meeting analysis failed. Please verify your Groq API connectivity."
        return insights