# 🧠 TensorFlow RAG + MCP Tools + Agentic AI

This repository contains a full hybrid AI system combining:

- **TensorFlow** (embeddings + ML models)
- **RAG** (vector search with ChromaDB)
- **MCP Tools** (FastAPI server exposing TensorFlow models)
- **Agentic AI** (LLM orchestrator using tool-calling)

This is a production‑ready architecture for intelligent agents that can:
- Retrieve knowledge using TensorFlow embeddings
- Call TensorFlow models through MCP-style tools
- Plan, reason, and act using an LLM

---

## 🚀 Features

### 🔹 TensorFlow RAG
- Universal Sentence Encoder embeddings
- ChromaDB vector store
- Query‑time retrieval

### 🔹 MCP Tools (FastAPI)
- TensorFlow sentiment classifier
- Math tool
- Easily extendable

### 🔹 Agentic AI
- LLM planning loop
- Automatic tool selection
- Multi-step reasoning

---

## 📦 Installation

```bash
git clone https://github.com/gitsabeer/tf-rag-mcp-agent.git
cd tf-rag-mcp-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
