import os
import tensorflow_hub as hub
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DB_DIR = os.getenv("RAG_DB_DIR", "./rag_db")

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def embed_text(texts):
    return embed(texts).numpy().tolist()

def rag_query(question: str):
    chroma = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=DB_DIR))
    collection = chroma.get_collection("tf_rag")

    q_vec = embed_text([question])[0]

    results = collection.query(query_embeddings=[q_vec], n_results=3)

    context = "\n".join(results["documents"][0])

    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question: {question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message["content"]
