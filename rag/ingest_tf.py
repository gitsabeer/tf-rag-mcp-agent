import os
import tensorflow_hub as hub
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

DB_DIR = os.getenv("RAG_DB_DIR", "./rag_db")

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def embed_text(texts):
    return embed(texts).numpy().tolist()

def load_corpus():
    return [
        "Retrieval-Augmented Generation (RAG) uses external knowledge.",
        "TensorFlow can generate embeddings for semantic search.",
        "MCP tools allow LLMs to call external functions safely."
    ]

def build_db():
    client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=DB_DIR))
    collection = client.get_or_create_collection("tf_rag")

    docs = load_corpus()
    vectors = embed_text(docs)

    collection.add(
        documents=docs,
        embeddings=vectors,
        ids=[f"doc_{i}" for i in range(len(docs))]
    )

    print("TensorFlow RAG DB built.")

if __name__ == "__main__":
    build_db()
