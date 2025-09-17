
from __future__ import annotations

import os
from typing import Optional
from pathlib import Path
import importlib.util
import sys

import pandas as pd
import streamlit as st

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Plant Recommender (RAG)", page_icon="🌿", layout="wide")
st.title("🌿 Plant Recommender — RAG Chatbot")
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

# If your notebook code relies on FIREWORKS_API_KEY in env but didn't set API_KEY,
# keep it as-is; we do NOT prompt the user.
if getattr(rag_core, "API_KEY", None):
    os.environ.setdefault("FIREWORKS_API_KEY", str(getattr(rag_core, "API_KEY")))

# ---------------- Catalog Loading ----------------
DEFAULT_CANDIDATES = [
    Path(__file__).with_name("catalog.csv"),
    Path(__file__).parent / "data" / "catalog.csv",
    Path("/mnt/data/catalog.csv"),
]

@st.cache_data(show_spinner=False)
def load_catalog_from_candidates() -> Optional[pd.DataFrame]:
    # If the notebook already created a catalog_df, prefer that
    if hasattr(rag_core, "catalog_df") and isinstance(rag_core.catalog_df, pd.DataFrame):
        return rag_core.catalog_df

    for p in DEFAULT_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            # Normalize
            df.columns = [c.strip() for c in df.columns]
            required = {"ID", "Name", "Description", "Price", "Delivery", "Labels"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"CSV at {p} missing required columns: {sorted(missing)}")
            return df
    return None

catalog_df = load_catalog_from_candidates()
if catalog_df is None:
    st.warning("Catalog not found. Place a 'catalog.csv' next to the app or expose 'catalog_df' in your notebook.")
    # We still allow chatting, but cards won't show.

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
    return answer, ids

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
        if catalog_df is not None and ids:
            st.subheader("Recommended plants")
            # Ensure IDs are ints
            try:
                ids = [int(x) for x in ids]
            except Exception:
                ids = ids
            matched = catalog_df[catalog_df["ID"].isin(ids)].copy()
            if not matched.empty:
                order = {pid: i for i, pid in enumerate(ids)}
                matched["_order"] = matched["ID"].map(order).fillna(1e9)
                matched = matched.sort_values("_order")
                cols = st.columns(min(3, len(matched)))
                for idx, (_, row) in enumerate(matched.iterrows()):
                    with cols[idx % len(cols)]:
                        product_card(row)
            else:
                st.info("No catalog rows matched the recommended IDs.")
        elif catalog_df is None:
            st.info("Catalog not available; expose 'catalog_df' in the notebook or place a 'catalog.csv'.")

    st.session_state["messages"].append({"role": "assistant", "content": answer})

# ---------------- Notes on Qdrant error ----------------
st.markdown(
    """
    <small>
    <b>Note on Qdrant embedded error</b>: “Storage folder ... is already accessed by another instance of Qdrant client.”
    Embedded Qdrant allows only a single client per storage path. This app avoids re-initialization by:
    <ul>
      <li>Loading your notebook backend once via <code>@st.cache_resource</code>.</li>
      <li>Never rebuilding vector stores on rerun.</li>
    </ul>
    If the error still appears, switch to a Qdrant <i>server</i> (Docker) and point your client to it instead of a local path.
    </small>
    """,
    unsafe_allow_html=True
)
