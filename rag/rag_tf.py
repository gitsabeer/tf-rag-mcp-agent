# Imports the `os` module for interacting with the operating system, such as accessing environment variables.
import os
# Imports TensorFlow Hub to load pre-trained models.
import tensorflow_hub as hub
# Imports the ChromaDB client for persistent vector database operations.
from chromadb import PersistentClient
# Imports a function to load environment variables from a `.env` file.
from dotenv import load_dotenv

# Imports the OpenAI client for API interactions.
from openai import OpenAI


# Loads environment variables from a `.env` file into the script's environment.
load_dotenv()
# Initializes an OpenAI client instance for making API calls.
client = OpenAI()

# Retrieves the database directory path from the environment variable `RAG_DB_DIR`, defaulting to `"./rag_db"` if not set.
DB_DIR = os.getenv("RAG_DB_DIR", "./rag_db")


# Loads the Universal Sentence Encoder model from TensorFlow Hub for generating text embeddings.
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Defines a function to embed a list of texts into vectors.
    # Embeds the input texts using the loaded model, converts the TensorFlow tensor to a NumPy array, and then to a Python list.
def embed_text(texts):
    return embed(texts).numpy().tolist()

# Defines the main RAG query function that takes a question string as input.
    # Initializes a persistent ChromaDB client using the specified database directory.
    # Retrieves the vector collection named `"tf_rag"` from the database.    
    #  Embeds the question into a vector and extracts the first (and only) result.
    # Queries the collection for the top 3 most similar documents based on the question's embedding.
    # Joins the retrieved documents into a single context string, separated by newlines.
    # Constructs a prompt string for open AI that includes the retrieved context and the original question for the language model.
    # Calls the OpenAI API to generate a completion using the specified model and messages.
    # Returns the generated response content from the first choice. (Note: In newer OpenAI API versions, this might be `.content` instead of `["content"]]`.)

def rag_query(question: str):
    chroma = PersistentClient(path=DB_DIR)
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
