"""
RAG Database Ingestion Script — Line-by-line explanation:

1. import os: Access environment variables and file paths.
2. import tensorflow_hub as hub: Load pre-trained TensorFlow models from TensorFlow Hub.
3. from chromadb import PersistentClient: Client for accessing/creating ChromaDB vector database.
4. from dotenv import load_dotenv: Load environment variables from .env file.

5. load_dotenv(): Parse and import .env variables into os.environ.

6. DB_DIR = os.getenv("RAG_DB_DIR", "./rag_db"): Set database directory path from env or default.

7. embed = hub.load(...): Load Universal Sentence Encoder model for text-to-vector embedding.

8. def embed_text(texts): Helper function to convert list of strings to embedding vectors.
   - embed(texts).numpy(): Convert TensorFlow tensor to NumPy array.
   - .tolist(): Convert NumPy array to Python list of lists (embeddings).

9. def load_corpus(): Dummy corpus of 3 example documents about RAG/TensorFlow/MCP.
   - Returns hardcoded list of strings to be embedded and stored.

10. def build_db(): Main function to populate ChromaDB with embedded documents.
    - client = PersistentClient(path=DB_DIR): Initialize persistent vector DB.
    - get_or_create_collection("tf_rag"): Create or retrieve collection named "tf_rag".
    - metadata={"hnsw:space": "cosine"}: Use cosine distance for similarity search.
    - docs = load_corpus(): Load 3 sample documents.
    - vectors = embed_text(docs): Embed each doc into vector form.
    - collection.add(...): Insert docs, embeddings, and auto-generated IDs into collection.
    - print(...): Print success message.

11. if __name__ == "__main__": Script entry point—run build_db() when executed directly.
"""

import os
import tensorflow_hub as hub
from chromadb import PersistentClient
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
    client = PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name="tf_rag",
        metadata={"hnsw:space": "cosine"}
    )

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
