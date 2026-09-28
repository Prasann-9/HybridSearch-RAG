"""ui/tabs.py — small presentation helpers. Use what you need; add use-case views here."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def df_table(rows: list[dict], wide_cols: tuple[str, ...] = ()):
    """Render a list of dicts as a proper table."""
    df = pd.DataFrame(rows)
    cfg = {c: st.column_config.TextColumn(width="large") for c in wide_cols if c in df.columns}
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=cfg)


def color_scale(v):
    """Red/amber/green for a 0..1 score. Use with df.style.map (NOT applymap — gone in pandas 3)."""
    if v is None or (isinstance(v, float) and v != v):
        return "background-color:#eeeeee;color:#999999"
    if v < 0.5:
        return "background-color:#fde2e1;color:#7f1d1d"
    if v < 0.75:
        return "background-color:#fef3c7;color:#78350f"
    return "background-color:#dcfce7;color:#14532d"


def render_mermaid(code: str, height: int = 420):
    """Render a Mermaid diagram via an iframe (mermaid.js from CDN, no extra dependency).

    Keep node labels plain (letters/numbers/spaces/hyphens); quotes, slashes, dots, and
    parentheses inside labels can trip the parser.
    """
    html = ('<div class="mermaid">\n' + code.strip() + '\n</div>\n'
            '<script type="module">\n'
            "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n"
            "mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });\n"
            '</script>')
    components.html(html, height=height, scrolling=True)
