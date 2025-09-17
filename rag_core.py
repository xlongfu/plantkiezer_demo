# Auto-generated from generation.ipynb
# If your original code defines variables like `API_KEY`, `graph`, etc., they will be available
# to the importing app. We add a light shim to pull FIREWORKS_API_KEY from env if API_KEY is missing.

import os

if "API_KEY" not in globals():
    API_KEY = os.environ.get("FIREWORKS_API_KEY", None)

# If your code didn't define START (LangGraph), provide a no-op placeholder to avoid import errors.
try:
    from langgraph.graph import START
except Exception:
    class _START: pass
    START = _START()


import random
import pandas as pd 
import re

from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from typing_extensions import List, TypedDict, Optional
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document


with open('fireworksai_api_key.txt', 'r') as file:
    API_KEY = file.read().strip()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Qdrant.from_existing_collection(
    collection_name="planten",
    embedding=embeddings,
    path="vector_stores/plantkiezer1",
)

data = pd.read_csv('data/texas_plant_list_cleaned.csv')

# Sample query used by the demo code at import-time
query = "Which plants are nice in a humid environment?"

query_expansion_instruct = "You enrich a user query for dense vector search. Return ONE line: the original query first, then up to 5 short synonym/keyword variants separated by ' | '. Preserve intent; prefer domain-specific terms likely found in the corpus. Each variant 2-6 words. No quotes, no explanations, no boolean operators, nothing else."

# --- NEW: keep a dedicated expansion LLM separate from the generation LLM ---
expansion_llm = ChatOllama(
    model="gemma3:4b", 
    keep_alive="30m",
    num_ctx=2048,
    num_predict=256,
    temperature=0.5
)

# The generation LLM (Fireworks)
llm = init_chat_model(
    # "accounts/fireworks/models/deepseek-v3",
    # "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507", 
    # "accounts/fireworks/models/gpt-oss-20b",
    # "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    model_provider="fireworks", 
    api_key=API_KEY
)

# MODEL = "gpt-oss-20b"
# MODEL = "gpt-oss-120b"
# MODEL = "qwen3-30b"
# MODEL = "deepseek-v3"
MODEL = "llama-v3p1-405b-instruct"


prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    ids: List[int]
    
    # generation & retrieval controls
    max_tokens: Optional[int]
    top_p: Optional[float]
    top_k: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    temperature: Optional[float]

def retrieve(state: State):
    """
    Fresh retrieval per user query:
    - Expand the incoming question each time.
    - Run MMR search with k=3.
    """
    user_q = state["question"]  # This is the incoming question (instruction + user text upstream)
    messages = [
        ("system", query_expansion_instruct),
        ("human", user_q),
    ]
    query_expanded = expansion_llm.invoke(messages).content
    retrieved_docs = list(vectorstore.max_marginal_relevance_search(query_expanded, k=3, filter=None))
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    ids = re.findall(r"ID:\s*(\d+)\s*\|", docs_content)
    ids = [int(i) for i in ids]

    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)

    return {"answer": response.content, "ids": ids}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# instruction = "You are an expert botanical assistant. You will be provided with five retrieved plant entries. Choose three of them that you think answers the user query the best and recommend it. Use the descriptions of the retrieved data to also provide more information about the plants. Then after your response to the user, write the IDs of the three plants you recommended in the format 'Recommended plant IDs: ID1, ID2, ID3'."

instruction = "You are an expert botanical assistant and also a sales chatbot. You will be provided with three retrieved plant entries. Answer the user query by recommending these three plants. Use the descriptions of the retrieved data to also provide more information about the plants. Frame your response concisely, while also like a real salesperson. Here is the user question: "

query = instruction + query

response = graph.invoke({
        "question": query,
        "max_tokens": 1024,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
    })

response['answer']

ids = [id - 1 for id in response['ids']]
ids
