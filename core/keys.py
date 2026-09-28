"""core/keys.py — BYOK key handling: a provider registry + resolution + validation + gating.

To change which keys the app needs, edit the PROVIDERS dict — that's the single source of
truth the sidebar, the env push, and the gating all read from. See references/byok-keys.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import streamlit as st

from core.config import EMBED_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# Validators — one cheap call each; return (ok, short_message). Lazy-import the SDKs.
# ──────────────────────────────────────────────────────────────────────────────
def validate_groq(api_key: str | None) -> tuple[bool, str]:
    if not api_key:
        return False, "no key provided"
    try:
        from groq import Groq
        Groq(api_key=api_key).models.list()
        return True, "key OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def validate_gemini(api_key: str | None) -> tuple[bool, str]:
    if not api_key:
        return False, "no key provided"
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key).embed_query("ping")
        return True, "key OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


# ──────────────────────────────────────────────────────────────────────────────
# Provider registry — edit this to match the use case.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Provider:
    env: str
    label: str
    help_url: str
    validate: Callable[[str | None], tuple[bool, str]]
    required: bool = True


PROVIDERS: dict[str, Provider] = {
    "GROQ_API_KEY":   Provider("GROQ_API_KEY",   "Groq (LLM)",          "https://console.groq.com/keys",       validate_groq,   True),
    "GEMINI_API_KEY": Provider("GEMINI_API_KEY", "Gemini (embeddings)", "https://aistudio.google.com/apikey",  validate_gemini, True),
}


# ──────────────────────────────────────────────────────────────────────────────
# Resolution + gating
# ──────────────────────────────────────────────────────────────────────────────
def get_secret(name: str) -> str | None:
    """Read a key from Streamlit secrets first, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def apply_keys(overrides: dict[str, str]) -> dict[str, bool]:
    """Resolve each provider key (sidebar override → secrets → env) and push to os.environ.

    Returns {env_name: present}. Call on every run before any provider call.
    """
    present = {}
    for name in PROVIDERS:
        val = (overrides.get(name) or "").strip() or get_secret(name)
        if val:
            os.environ[name] = val
        else:
            os.environ.pop(name, None)
        present[name] = bool(val)
    # langchain-google-genai reads GOOGLE_API_KEY; mirror Gemini into it.
    if os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    else:
        os.environ.pop("GOOGLE_API_KEY", None)
    return present


def required_keys() -> list[str]:
    return [n for n, p in PROVIDERS.items() if p.required]


def all_required_valid(valid: dict[str, bool]) -> bool:
    return all(valid.get(n) for n in required_keys())
