
from __future__ import annotations

import os
import re
from typing import List, Optional

import pandas as pd
import streamlit as st

import importlib.util
import sys
from pathlib import Path

RAG_CORE_PATH = Path(__file__).with_name("rag_core.py")
spec = importlib.util.spec_from_file_location("rag_core", RAG_CORE_PATH)
rag_core = importlib.util.module_from_spec(spec)
sys.modules["rag_core"] = rag_core
spec.loader.exec_module(rag_core)  # type: ignore

# -------- Streamlit UI --------
st.set_page_config(page_title="Plant Recommender (RAG)", page_icon="🌿", layout="wide")
st.title("🌿 Plant Recommender — RAG Chatbot")
st.caption("Ask for plants (light, care, style, budget). I’ll recommend & show product cards.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Fireworks API Key", type="password", value=os.environ.get("FIREWORKS_API_KEY", ""))
    if api_key:
        os.environ["FIREWORKS_API_KEY"] = api_key

    st.markdown("---")
    uploaded = st.file_uploader("Upload product catalog CSV", type=["csv"])

# --- CSS for cards ---
st.markdown(
    """
    <style>
    .badge {display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; margin-right:6px; background:#eef;}
    .card {padding:16px; border:1px solid #eee; border-radius:16px; box-shadow: 0 1px 4px rgba(0,0,0,.05); height:100%;}
    .card h4 {margin:0 0 8px 0}
    .card img {width:100%; border-radius:12px; margin-bottom:8px;}
    .price {font-weight:700; font-size:1.1rem;}
    .meta {color:#666; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True
)

def product_card(row: pd.Series):
    name = str(row.get("Name", "Plant"))
    price = row.get("Price", "—")
    delivery = row.get("Delivery", "—")
    labels = str(row.get("Labels", "")).split(",") if pd.notna(row.get("Labels", "")) else []
    desc = str(row.get("Description", "")).strip()
    img = row.get("ImageURL") if "ImageURL" in row else None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if img and pd.notna(img) and str(img).strip():
        st.markdown(f'<img src="{img}" alt="{name}">', unsafe_allow_html=True)
    st.markdown(f"<h4>{name}</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='price'>{price}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='meta'>Delivery: {delivery}</div>", unsafe_allow_html=True)
    if labels:
        st.write(" ".join([f"<span class='badge'>{l.strip()}</span>" for l in labels if l.strip()]), unsafe_allow_html=True)
    if desc:
        st.caption(desc[:220] + ("…" if len(desc) > 220 else ""))
    c1, c2 = st.columns(2)
    with c1:
        st.button("Add to cart", key=f"add_{row.get('ID')}")
    with c2:
        st.button("View details", key=f"view_{row.get('ID')}")
    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_catalog(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    required = {"ID", "Name", "Description", "Price", "Delivery", "Labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render past messages
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User input
user_msg = st.chat_input("Describe your space, light level, and style. Example: 'low-light pet-safe plants under $30'.")

# Core RAG invocation helper
def run_rag(query: str):
    """
    Calls the user's RAG graph if present; otherwise tries fallback patterns.
    The notebook code should expose a variable named `graph` with an .invoke(...) method
    and rely on the prompt embedded there. We follow your original pattern:
      full_query = instruction + user_query
    and expect the generator to extract IDs by reading 'ID: <num> |' in context.
    """
    # Construct instruction if available; else use a default
    instruction = (
        "You are an expert botanical assistant and also a sales chatbot. "
        "You will be provided with three retrieved plant entries. Answer the user query by recommending these three plants. "
        "Use the descriptions of the retrieved data to also provide more information about the plants. "
        "Frame your response concisely, while also like a real salesperson. Here is the user question: "
    )
    full_query = instruction + query

    # Prepare call args; align with your snippet
    args = {
        "question": full_query,
        "max_tokens": 1024,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
    }

    if hasattr(rag_core, "graph"):
        try:
            result = rag_core.graph.invoke(args)
            answer = result.get("answer", "")
            ids = result.get("ids", [])
            return answer, ids
        except Exception as e:
            return f"Backend error while invoking graph: {e}", []
    return "Your notebook didn't expose `graph`. Please ensure it builds and assigns a compiled graph to `graph`.", []

catalog_df: Optional[pd.DataFrame] = None
if uploaded is not None:
    try:
        catalog_df = load_catalog(uploaded)
        st.success(f"Loaded catalog with {len(catalog_df)} rows")
    except Exception as e:
        st.error(f"Failed to load catalog: {e}")

if user_msg:
    # show user message
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Run RAG
    with st.spinner("Thinking…"):
        answer, ids = run_rag(user_msg)

    # assistant output
    with st.chat_message("assistant"):
        st.markdown(answer if answer else "_(no answer)_")

        if catalog_df is not None and ids:
            st.subheader("Recommended plants")
            matched = catalog_df[catalog_df["ID"].isin([int(i) for i in ids])].copy()
            if not matched.empty:
                # preserve order of IDs
                order = {int(pid): i for i, pid in enumerate(ids)}
                matched["_order"] = matched["ID"].map(order).fillna(1e9)
                matched = matched.sort_values("_order")
                cols = st.columns(min(3, len(matched)))
                for idx, (_, row) in enumerate(matched.iterrows()):
                    with cols[idx % len(cols)]:
                        product_card(row)
            else:
                st.info("I couldn't match the recommended IDs in your catalog. Check that your CSV 'ID' values match the IDs embedded in documents.")
        elif catalog_df is None:
            st.info("Upload your catalog CSV to see product cards.")

    st.session_state["messages"].append({"role": "assistant", "content": answer})
