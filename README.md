# LLM Engineering Patterns

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic%20v2-e92063.svg?logo=pydantic)](https://docs.pydantic.dev/)
[![Ollama](https://img.shields.io/badge/Inference-Local%20Ollama-black.svg?logo=ollama)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/Testing-pytest-green.svg?logo=pytest)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A collection of foundational, test-driven LLM engineering patterns implemented in Python using local inference models via **Ollama**. 

This repository bridges the gap between ad-hoc prompting and production-grade AI systems by enforcing **strict Pydantic schema validation**, **deterministic fallbacks**, **latency & token telemetry**, and **modular provider architectures**.

---

## 🎯 Implemented Patterns

| Pattern | Description | Module | CLI / Entrypoint |
| :--- | :--- | :--- | :--- |
| **Classification** | Categorizes tickets and log error events into typed enums with structured chain-of-thought reasoning. | `app/patterns/classification.py` | `classify_cli.py` |
| **Extraction** | Extracts structured fields from unstructured text with deterministic regex validation fallback. | `app/patterns/extraction.py` | `extract_cli.py` |
| **Generation** | Generates bounded domain outputs with temperature controls and schema parsing. | `app/patterns/generation.py` | `generate_cli.py` |
| **Summarization** | Compresses long-context operational logs and incidents with token-efficient prompts. | `app/patterns/summarization.py` | `summarize_cli.py` |
| **Semantic Similarity** | Computes cosine similarity across text pairs using high-dimensional vector embeddings. | `app/patterns/similarity.py` | `similarity_cli.py` |
| **Semantic Search** | Ranks local knowledge documents against natural-language queries. | `app/patterns/similarity.py` | `semantic_search.py` |
| **Retrieval-Augmented Generation (RAG)** | In-memory vector store with chunking, overlap, top-$k$ retrieval, and grounded answer synthesis. | `app/patterns/rag.py` | Programmatic API |

---

## 🏗️ Architecture & Project Structure

```text
llm-engineering-patterns/
├── app/
│   ├── deterministic/          # Non-LLM fallback logic (e.g., regex extractors)
│   ├── patterns/               # Core AI design patterns
│   │   ├── classification.py
│   │   ├── extraction.py
│   │   ├── generation.py
│   │   ├── rag.py              # In-memory RAG pipeline
│   │   ├── similarity.py       # Cosine similarity & vector comparison
│   │   └── summarization.py
│   ├── providers/              # Extensible LLM & Embedding providers
│   │   ├── base.py             # Abstract base classes & response types
│   │   ├── embeddings.py       # Ollama embeddings (nomic-embed-text)
│   │   └── ollama.py           # Ollama chat/generation inference provider
│   ├── schemas/                # Strongly-typed Pydantic schemas
│   │   ├── classification.py
│   │   ├── extraction.py
│   │   ├── generation.py
│   │   ├── rag.py
│   │   └── summarization.py
│   └── utilities/              # Helpers for telemetry, latency, and profiling
├── sample_data/                # Sample incident logs and operational data
├── tests/                      # Comprehensive pytest test suite
├── classify_cli.py             # CLI: Classification
├── extract_cli.py              # CLI: Extraction
├── generate_cli.py             # CLI: Generation
├── semantic_search.py          # CLI: Vector semantic search
├── similarity_cli.py           # CLI: Embedding similarity
├── summarize_cli.py            # CLI: Log summarization
├── .env.example
└── README.md
```

---

## ⚡ Key Engineering Principles

- **Zero Hallucination through Schemas**: All generative outputs are validated using Pydantic models. Malformed model outputs raise typed exceptions (`ClassificationError`, `RagError`, etc.) rather than silently corrupting data.
- **Provider Agnostic**: Core patterns interact with abstract base classes (`BaseModelProvider`), decoupling application logic from any single LLM backend.
- **Production Observability**: Responses return complete telemetry including `latency_ms`, `input_tokens`, and `output_tokens`.
- **Hybrid Architectures**: Deterministic heuristics (regex, schema guards) run alongside stochastic LLM operations for cost and latency optimization.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.11+**
- **[Ollama](https://ollama.com/)** running locally

Pull the required local models:
```bash
# Text Generation & Reasoning
ollama pull qwen3.5:4b

# High-dimensional Embeddings
ollama pull nomic-embed-text:latest
```

### 2. Installation
Clone the repository:
```bash
git clone https://github.com/kumaradoss16/llm-engineering-patterns.git
cd llm-engineering-patterns
```

Create and activate a virtual environment:
```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install httpx pydantic pytest python-dotenv
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen3.5:4b"
OLLAMA_EMBEDDING_MODEL="nomic-embed-text:latest"
OLLAMA_TIMEOUT="300.0"
```

---

## 💻 CLI Usage Examples

### 1. Classification
Classify customer support tickets or error logs into categorized outputs:
```bash
python classify_cli.py ticket "I was double charged on my invoice #9821."
python classify_cli.py error "OperationalError: connection to server at 'db.internal' failed: timeout expired"
```

### 2. Structured Extraction
Extract specific entities according to Pydantic schemas:
```bash
python extract_cli.py "User Alice Smith (alice@example.com) requested password reset for server-02."
```

### 3. Semantic Similarity & Search
Compare the semantic meaning of two phrases:
```bash
python similarity_cli.py "Restart postgres database" "Reload database service"
```

Perform semantic search over local documentation chunks:
```bash
python semantic_search.py "How do I fix container network resolution issues?"
```

### 4. Summarization
Summarize operational logs with key takeaways:
```bash
python summarize_cli.py sample_data/incident_log.txt
```

---

## 🧪 Running Tests

A full suite of unit tests covers each pattern and schema validation:

```bash
pytest -v
```

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
