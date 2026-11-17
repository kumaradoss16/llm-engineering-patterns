# LLM Engineering Patterns

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic%20v2-e92063.svg?logo=pydantic)](https://docs.pydantic.dev/)
[![Ollama](https://img.shields.io/badge/Inference-Local%20Ollama-black.svg?logo=ollama)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/Testing-68%20Passed-green.svg?logo=pytest)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A collection of foundational, test-driven LLM engineering patterns implemented in Python using local inference models via **Ollama**. 

This repository bridges the gap between ad-hoc prompting and production-grade AI systems by enforcing **strict Pydantic schema validation**, **deterministic fallbacks**, **latency & token telemetry**, **financial cost estimation**, **sanitized JSON handling**, and **modular provider architectures**.

---

## 🎯 Implemented Patterns

| Pattern | Description | Module | CLI / Entrypoint |
| :--- | :--- | :--- | :--- |
| **Classification** | Categorizes tickets and log error events into typed enums with structured chain-of-thought reasoning. | `app/patterns/classification.py` | `classify_cli.py` / `main.py classify` |
| **Extraction (LLM & Hybrid)** | Extracts structured fields from unstructured text with regex validation fallback and Pydantic schemas. | `app/patterns/extraction.py`<br>`app/patterns/llm_extraction.py` | `extract_cli.py` / `main.py extract` |
| **Deterministic Extraction** | High-throughput, zero-cost IPv4/regex extraction with instantaneous telemetry. | `app/deterministic/ip_extractor.py` | `compare_extraction.py` / `main.py compare` |
| **Generation** | Generates bounded domain outputs with temperature controls and schema parsing. | `app/patterns/generation.py` | `generate_cli.py` / `main.py generate` |
| **Summarization** | Compresses long-context operational logs and incidents with token-efficient prompts. | `app/patterns/summarization.py` | `summarize_cli.py` / `main.py summarize` |
| **Semantic Similarity** | Computes cosine similarity across text pairs using high-dimensional vector embeddings. | `app/patterns/similarity.py` | `similarity_cli.py` / `main.py similarity` |
| **Semantic Search** | Ranks local knowledge documents against natural-language queries. | `app/patterns/similarity.py` | `semantic_search.py` |
| **Retrieval-Augmented Generation (RAG)** | In-memory vector store with chunking, overlap, top-$k$ retrieval, and grounded answer synthesis. | `app/patterns/rag.py` | `rag_cli.py` / `main.py rag` |
| **Bounded Tool Calling** | Safe, sandboxed tool invocation with Pydantic request validation, markdown fence sanitization, and deterministic execution. | `app/patterns/tool_calling.py` | `tools_cli.py` / `main.py tools` |
| **Extraction Comparison** | Side-by-side benchmarking pitting deterministic regex extraction against LLMs on latency, accuracy, and dollar cost. | `compare_extraction.py` | `compare_extraction.py` / `main.py compare` |
| **Pattern Evaluation Suite** | Golden dataset benchmark assessing pattern accuracy, latency, and token footprint. | `evaluate_patterns.py` | `evaluate_patterns.py` |

---

## 🏗️ Architecture & Project Structure

```text
llm-engineering-patterns/
├── app/
│   ├── config.py                 # Pydantic environment configuration (Settings)
│   ├── deterministic/            # Non-LLM deterministic heuristics
│   │   └── ip_extractor.py       # Zero-latency regex IPv4/IPv6 extraction
│   ├── patterns/                 # Core AI design patterns
│   │   ├── classification.py     # Intent and error taxonomy
│   │   ├── extraction.py         # Pydantic entity extraction
│   │   ├── generation.py         # Constrained text generation
│   │   ├── llm_extraction.py     # Dedicated LLM extraction pattern
│   │   ├── rag.py                # In-memory RAG pipeline
│   │   ├── similarity.py         # Cosine similarity & vector comparison
│   │   ├── summarization.py      # Log & document summarization
│   │   └── tool_calling.py       # Safe bounded tool selection & execution
│   ├── providers/                # Extensible LLM & Embedding providers
│   │   ├── base.py               # Abstract base classes & response types
│   │   ├── embeddings.py         # Ollama embeddings (nomic-embed-text)
│   │   └── ollama.py             # Ollama chat/generation inference provider
│   ├── schemas/                  # Strongly-typed Pydantic schemas
│   │   ├── classification.py
│   │   ├── extraction.py
│   │   ├── generation.py
│   │   ├── rag.py
│   │   ├── summarization.py
│   │   └── tool_calling.py
│   └── utilities/                # Production infrastructure utilities
│       ├── cost.py               # Model pricing & USD token cost estimation
│       ├── logging.py            # Structured JSON application logger
│       └── token_usage.py        # Token usage aggregation & tracking
├── results/                      # Evaluation and benchmarking outputs
│   └── evaluation_results.json   # Exported benchmark run telemetry
├── sample_data/                  # Sample incident logs and operational knowledge docs
│   ├── knowledge/                # Markdown documents for RAG
│   ├── incident_log.txt          # Raw incident timeline log
│   └── support_ticket.txt        # Sample enterprise support ticket
├── tests/                        # Comprehensive test suite (68 unit tests)
│   ├── test_classification.py
│   ├── test_cost_edge_cases.py   # Cost calculations, precision & zero-token edge cases
│   ├── test_deterministic.py     # Deterministic regex extraction tests
│   ├── test_extraction.py
│   ├── test_generation.py
│   ├── test_llm_extraction.py    # LLM extraction schemas & prompt formatting
│   ├── test_logging.py           # Structured logger behavior
│   ├── test_provider_failures.py # Resilience to timeouts, malformed JSON & API errors
│   ├── test_rag.py
│   ├── test_similarity.py
│   ├── test_summarization.py
│   ├── test_token_usage.py       # Telemetry aggregation tests
│   └── test_tool_calling.py
├── .env.example                  # Environment configuration template
├── compare_extraction.py         # Deterministic vs. LLM benchmarking tool
├── evaluate_patterns.py          # Dataset-driven pattern evaluation harness
├── evaluation_dataset.json       # Labeled golden test cases across all patterns
├── main.py                       # Unified CLI runner for all patterns & diagnostics
├── classify_cli.py               # Dedicated CLI: Classification
├── extract_cli.py                # Dedicated CLI: Extraction
├── generate_cli.py               # Dedicated CLI: Generation
├── rag_cli.py                    # Dedicated CLI: Retrieval-Augmented Generation
├── semantic_search.py            # Dedicated CLI: Vector semantic search
├── similarity_cli.py             # Dedicated CLI: Embedding similarity
├── summarize_cli.py              # Dedicated CLI: Log summarization
├── tools_cli.py                  # Dedicated CLI: Bounded tool calling
└── README.md
```

---

## ⚡ Key Engineering Principles

- **Zero Hallucination through Schemas**: All generative outputs are validated using Pydantic models. Malformed model outputs raise typed exceptions (`ClassificationError`, `RagError`, `ToolCallError`, `LLMExtractionError`) rather than silently corrupting data.
- **Deterministic vs. Stochastic Tradeoffs**: When an operation can be solved with deterministic logic (e.g., regex IP extraction taking $<0.05\text{ms}$ with zero API cost), deterministic code takes precedence over expensive LLM inference.
- **Financial & Performance Observability**: Every execution records execution telemetry including `latency_ms`, `input_tokens`, and `output_tokens`, along with real-time dollar cost estimation using configurable model pricing tiers (`Decimal` precision).
- **Resilience Against Model Failures**: Built-in markdown fence sanitization, JSON salvage logic, and comprehensive failure test coverage protect against empty payloads, hallucinated schemas, or timeout faults.
- **Centralized Configuration**: Settings are parsed and typed through `app/config.py` using `pydantic-settings` / `python-dotenv`.
- **Systematic Evaluation**: The `evaluate_patterns.py` suite runs pattern implementations against a standardized `evaluation_dataset.json`, publishing structured metrics directly into `results/evaluation_results.json`.

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
Copy `.env.example` to create your local `.env` file:
```bash
cp .env.example .env
```

Default contents:
```env
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen3.5:4b"
OLLAMA_EMBEDDING_MODEL="nomic-embed-text:latest"
OLLAMA_TIMEOUT="300.0"

# Optional hosted provider
# OPENAI_API_KEY=""
# OPENAI_MODEL="gpt-4o-mini"
```

---

## 💻 CLI Usage Examples

### Unified CLI (`main.py`)

The unified CLI provides a single entry point for all patterns with integrated telemetry and cost estimation:

```bash
# 1. Summarization
python main.py summarize sample_data/incident_log.txt

# 2. Classification (ticket or error)
python main.py classify ticket "I was double charged on my invoice #9821."
python main.py classify error "OperationalError: connection to server at 'db.internal' failed: timeout expired"

# 3. Extraction (log or software)
python main.py extract log "Failed login attempt from 192.168.1.100 on port 22"
python main.py extract software "Installed python version 3.11.4 on Ubuntu 22.04"

# 4. Constrained Generation
python main.py generate "How do I diagnose high memory consumption in redis?"

# 5. Semantic Similarity
python main.py similarity "Restart database" "Reload PostgreSQL service"

# 6. Retrieval-Augmented Generation (RAG)
python main.py rag "How do I configure the failover cluster?" --top-k 2

# 7. Safe Bounded Tool Calling
python main.py tools "Check the status of the web service"

# 8. Deterministic vs LLM Comparison
python main.py compare "Connection failed to 10.0.0.42:8080 after timeout"
```

---

### Dedicated Pattern CLIs

Each pattern also features a dedicated standalone CLI for isolated inspection:

```bash
python classify_cli.py ticket "Credit card declined during renewal"
python extract_cli.py "User Alice Smith (alice@example.com) requested reset on server-02."
python similarity_cli.py "Restart postgres database" "Reload database service"
python semantic_search.py "How do I fix container network resolution issues?"
python summarize_cli.py sample_data/incident_log.txt
python rag_cli.py "How do I restart the nginx service?"
python tools_cli.py "Convert 100 degrees Celsius to Fahrenheit"
```

---

### Extraction Comparison Benchmark (`compare_extraction.py`)

Benchmark deterministic regex extraction against local LLM extraction to measure latency, accuracy, and dollar cost tradeoffs:

```bash
python compare_extraction.py
```

Sample benchmark output:
```json
[
  {
    "approach": "deterministic_regex",
    "extracted_value": "192.168.1.25",
    "correct": true,
    "latency_ms": 0.04,
    "input_tokens": null,
    "output_tokens": null,
    "estimated_cost_usd": 0.0
  },
  {
    "approach": "llm_extraction",
    "extracted_value": "192.168.1.25",
    "correct": true,
    "latency_ms": 1420.35,
    "input_tokens": 84,
    "output_tokens": 28,
    "estimated_cost_usd": 0.00003192
  }
]
```

---

### Golden Dataset Evaluation Suite (`evaluate_patterns.py`)

Run the full evaluation pipeline against `evaluation_dataset.json` to inspect accuracy, latency, and token consumption across all patterns:

```bash
python evaluate_patterns.py
```

Results are saved to `results/evaluation_results.json`.

---

## 🧪 Running Tests

A comprehensive suite of **68 unit tests** validates all schemas, patterns, cost calculations, deterministic extractors, and failure recovery edge cases:

```bash
pytest -v
```

Test coverage includes:
- **Pattern Validation**: Classification, extraction, generation, RAG, similarity, summarization, and bounded tool execution.
- **Deterministic Heuristics**: Regex IP extraction, edge cases, IPv4 validation.
- **Provider Resilience**: API timeouts, HTTP connection drops, malformed JSON, empty responses.
- **Telemetry & Utilities**: Token usage counters, structured logging, Decimal price calculations, and zero-token cost rounding.

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
