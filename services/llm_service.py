"""
services/llm_service.py - Groq LLM Integration

This module handles all communication with the Groq API.
It uses langchain-groq's ChatGroq to send prompts and receive responses.

Why Groq?
- Extremely fast inference (runs LLMs on custom LPU hardware)
- Free tier available for learning
- Supports Llama 3.3 70B which is very capable

Why langchain-groq?
- Handles API communication, retries, and error handling
- Clean interface: just pass a prompt, get a response
- Consistent with other LangChain components
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_API_KEY, MODEL_NAME


# System message that sets the AI's behavior for all interactions
SYSTEM_PROMPT = """You are MeetingMind AI, an intelligent meeting analysis assistant.

Your rules:
1. ONLY answer using the provided meeting transcript context.
2. NEVER make up information or hallucinate facts.
3. If the information is not in the context, say: "I couldn't find that information in the uploaded meeting transcript."
4. Be professional and concise.
5. Use bullet points when listing multiple items.
6. Always reference which part of the transcript your answer comes from.
"""


def get_llm_response(prompt: str, system_prompt: str = None) -> str:
    """
    Send a prompt to Groq's LLM and return the response.

    Args:
        prompt: The user prompt / question with context
        system_prompt: Optional custom system prompt (defaults to SYSTEM_PROMPT)

    Returns:
        The LLM's text response

    Raises:
        Exception: If the API call fails (network error, invalid key, etc.)
    """
    # Use the default system prompt if none is provided
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    try:
        # Initialize the Groq LLM client
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=0.1,  # Low temperature = more focused, factual responses
            max_tokens=1200   # Maximum response length
        )

        # Build the message list (system + human message)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        # Call the LLM and get the response
        response = llm.invoke(messages)

        print(f"✅ LLM response received ({len(response.content)} characters)")
        return response.content

    except Exception as e:
        error_msg = f"❌ Error calling Groq API: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)
