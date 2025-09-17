
from __future__ import annotations

import os
from typing import List
from pathlib import Path
import importlib.util
import sys

import pandas as pd
import streamlit as st

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Plantkiezer.nl", page_icon="🌿", layout="wide")
st.title("🌿 Plantkiezer Plant Recommender")
st.caption("Type your plant needs; I'll recommend and show product cards.")

# ---------------- Robust one-time module load ----------------
@st.cache_resource(show_spinner=False)
def load_rag_core_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("rag_core", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rag_core"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod

RAG_CORE_PATH = Path(__file__).with_name("rag_core.py")
if not RAG_CORE_PATH.exists():
    st.error("rag_core.py not found. Make sure it sits next to this file.")
    st.stop()

rag_core = load_rag_core_module(RAG_CORE_PATH)

if getattr(rag_core, "API_KEY", None):
    os.environ.setdefault("FIREWORKS_API_KEY", str(getattr(rag_core, "API_KEY")))

# ---------------- Catalog Loading (fixed path) ----------------
CATALOG_PATH = Path("data/texas_plant_list_cleaned.csv")

@st.cache_data(show_spinner=False)
def load_catalog_fixed() -> pd.DataFrame:
    if CATALOG_PATH.exists():
        df = pd.read_csv(CATALOG_PATH)
    else:
        # Fallback to absolute path used in this environment if relative isn't available
        alt = Path("/mnt/data/data/texas_plant_list_cleaned.csv")
        if alt.exists():
            df = pd.read_csv(alt)
        else:
            raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH} or {alt}")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    catalog_df = load_catalog_fixed()
    st.success(f"Loaded catalog from {CATALOG_PATH if CATALOG_PATH.exists() else '/mnt/data/data/texas_plant_list_cleaned.csv'} with {len(catalog_df)} rows")
except Exception as e:
    st.error(f"Failed to load catalog: {e}")
    st.stop()

# ---------------- Styles ----------------
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
    # Focus on Price, Delivery, Labels; use optional fields if present
    price = row.get("Price", "—")
    delivery = row.get("Delivery", "—")
    labels = str(row.get("Labels", "")).split(",") if pd.notna(row.get("Labels", "")) else []
    name = str(row.get("Name", "Plant"))
    desc = str(row.get("Description", "")).strip()
    img = row.get("ImageURL") if "ImageURL" in row else None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if img is not None and pd.notna(img) and str(img).strip():
        st.markdown(f'<img src="{img}" alt="{name}">', unsafe_allow_html=True)
    st.markdown(f"<h4>{name}</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='price'>{price}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='meta'>Delivery: {delivery}</div>", unsafe_allow_html=True)
    if labels:
        st.write(" ".join([f"<span class='badge'>{l.strip()}</span>" for l in labels if l.strip()]), unsafe_allow_html=True)
    if desc:
        st.caption(desc[:220] + ("…" if len(desc) > 220 else ""))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Chat history ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- RAG Invocation ----------------
def run_rag(user_query: str):
    """
    Invoke the compiled `graph` from the notebook once.
    We DO NOT rebuild any vector stores here to avoid Qdrant re-open issues.
    """
    instruction = (
        "You are an expert botanical assistant and also a sales chatbot. "
        "You will be provided with three retrieved plant entries. Answer the user query by recommending these three plants. "
        "Use the descriptions of the retrieved data to also provide more information about the plants. "
        "Frame your response concisely, while also like a real salesperson. Here is the user question: "
    )
    args = {
        "question": instruction + user_query,
        "max_tokens": 1024,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
    }
    if not hasattr(rag_core, "graph"):
        return "Your backend didn't expose a compiled `graph`.", []

    try:
        result = rag_core.graph.invoke(args)
    except Exception as e:
        return f"Backend error while invoking graph: {e}", []

    answer = result.get("answer", "")
    ids = result.get("ids", [])
    return answer, [id - 1 for id in ids]

# Utility: normalize IDs to DataFrame row indices
def normalize_to_indices(ids: List[int], df_len: int) -> List[int]:
    if not ids:
        return []
    # unique preserve order
    uniq = []
    for x in ids:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in uniq:
            uniq.append(xi)

    # First try as-is (assume 0-based)
    in_bounds = [i for i in uniq if 0 <= i < df_len]
    if in_bounds:
        return in_bounds

    # Fallback: treat as 1-based and convert
    shifted = [i-1 for i in uniq]
    in_bounds2 = [i for i in shifted if 0 <= i < df_len]
    return in_bounds2

# ---------------- Chat input ----------------
user_msg = st.chat_input("Describe your space, light level, and style. Example: 'low-light pet-safe plants under $30'.")

if user_msg:
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.spinner("Thinking…"):
        answer, ids = run_rag(user_msg)

    with st.chat_message("assistant"):
        st.markdown(answer or "_(no answer)_")

        norm_idx = normalize_to_indices(ids, len(catalog_df))
        if norm_idx:
            st.subheader("Recommended plants")
            # Keep order and render up to 3 columns
            cols = st.columns(min(3, len(norm_idx)))
            for j, ridx in enumerate(norm_idx):
                row = catalog_df.iloc[ridx]
                with cols[j % len(cols)]:
                    product_card(row)
        else:
            st.info("No valid row indices matched the recommended items.")

    st.session_state["messages"].append({"role": "assistant", "content": answer})
