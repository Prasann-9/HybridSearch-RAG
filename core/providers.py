"""core/providers.py — thin provider call helpers.

These read keys from os.environ (set by core.keys.apply_keys). Keep the helpers you use, add
others as needed. Run any of these inside ui.async_utils.run_in_thread when called from the app.
"""
from __future__ import annotations

import os

from core.config import DEFAULT_GROQ_MODEL, EMBED_MODEL


def chat_groq(prompt: str, system: str | None = None, model: str = DEFAULT_GROQ_MODEL,
              temperature: float = 0.0) -> str:
    """One-shot Groq chat completion. Returns the assistant text."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    return ChatGroq(model=model, temperature=temperature).invoke(msgs).content.strip()


def embed_gemini(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed a list of texts with Gemini (API-based, torch-free)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    gkey = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=gkey).embed_documents(texts)


def embed_gemini_query(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single query with Gemini. Uses embed_query so the task-type hint matches."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    gkey = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=gkey).embed_query(text)
