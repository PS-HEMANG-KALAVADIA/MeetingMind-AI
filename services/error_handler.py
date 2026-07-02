"""
services/error_handler.py - User-friendly Error Parser for MeetingMind AI

This module translates technical Python exceptions (such as Groq API credential
errors, connection failures, rate limits, or file system errors) into polite,
actionable, user-friendly markdown messages.
"""


def get_user_friendly_error(e: Exception, action: str = "") -> str:
    """
    Parse a technical exception and return a user-friendly Markdown message.

    Args:
        e: The exception that was caught.
        action: A description of the action being attempted (e.g., "loading transcript").

    Returns:
        A formatted markdown string explaining the problem and potential solutions.
    """
    err_msg = str(e).lower()
    action_suffix = f" while {action}" if action else ""

    # Check for authentication / API key issues
    if any(keyword in err_msg for keyword in ["api key", "apikey", "unauthorized", "invalid api key", "401", "authentication"]):
        return (
            "🔑 **Groq API Key Authentication Failed**\n\n"
            "Unable to connect to the Groq AI service. Please verify that your "
            "`GROQ_API_KEY` environment variable is correctly set in your `.env` file and is active.\n\n"
            "Need a key? Generate one at [console.groq.com](https://console.groq.com)."
        )

    # Check for rate limit / token exhausted errors
    if any(keyword in err_msg for keyword in ["rate limit", "429", "too many requests", "limit exceeded"]):
        return (
            "⏳ **API Rate Limit Exceeded**\n\n"
            "We have sent too many requests to the Groq service in a short period. "
            "Please pause for a moment and try again shortly."
        )

    # Check for connection / network issues
    if any(keyword in err_msg for keyword in ["connection", "timeout", "network", "offline", "failed to establish", "dns", "urllib3", "requests"]):
        return (
            "🔌 **AI Service Connection Failure**\n\n"
            "Unable to contact the AI service. Please check your internet connection "
            "and verify that `api.groq.com` is reachable."
        )

    # Check for model-related issues
    if "model" in err_msg and any(keyword in err_msg for keyword in ["not found", "invalid", "unknown"]):
        return (
            "🤖 **AI Model Not Found**\n\n"
            "The configured model was not recognized by the Groq API. Please verify the `MODEL_NAME` "
            "value in your `.env` file."
        )

    # Check for file path/not found issues
    if isinstance(e, FileNotFoundError) or "file not found" in err_msg:
        return (
            "📁 **File Not Found**\n\n"
            f"The system could not locate the specified file{action_suffix}. "
            "Please double-check the file path and try again."
        )

    # Fallback to cleaner generic formatting
    return (
        f"❌ **An unexpected error occurred{action_suffix}**\n\n"
        f"**Error details:** `{str(e)}`"
    )
