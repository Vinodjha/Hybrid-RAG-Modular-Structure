from typing import List, Dict, Any
from groq import Groq
from app.core.settings import settings

def generate_answer(messages: List[Dict[str, Any]], max_tokens: int = 350) -> str:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment.")
    client = Groq(api_key=settings.GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=max_tokens,
        stream=False
    )
    return completion.choices[0].message.content
