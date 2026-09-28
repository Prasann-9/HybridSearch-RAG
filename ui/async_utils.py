"""ui/async_utils.py — run blocking work off Streamlit's script thread.

Streamlit's rerun model breaks libraries that spin up their own asyncio loop (langchain async
paths, langgraph, ragas, some SDKs) with "no current event loop" / "event loop is closed".
Running that work in a fresh worker thread — with its own loop — sidesteps the whole class of
problem WITHOUT nest_asyncio (which is flaky on 3.12+). See references/async-worker.md.
"""
from __future__ import annotations

import asyncio
import concurrent.futures


def _ensure_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def run_in_thread(fn, *args, timeout: int = 1800, **kwargs):
    """Execute a synchronous callable in an isolated worker thread; return its result.

    Do NOT call st.* inside fn — collect results/errors into plain objects and render after.
    """
    def _wrapped():
        _ensure_loop()
        return fn(*args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_wrapped).result(timeout=timeout)
