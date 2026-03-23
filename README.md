# Agentic RAG System — Instance 1 (RAGAS Evaluation)

A fully agentic Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph, using PostgreSQL with pgvector as the vector store and Ollama for local model inference. Evaluation is performed using the RAGAS framework with `llama3.2` as the judge LLM.

---

## System Overview

The system is split into two sub-systems:

**Sub-System 1 — Agentic RAG Pipeline**
Ingests a text dataset, embeds it into a pgvector store, and answers questions using a small LLM with four agentic behaviours managed via a LangGraph state machine.

**Sub-System 2 — Evaluation**
Scores each generated answer against the expected answer using the RAGAS framework, evaluated by a fixed judge LLM (`llama3.2`).

---

## Models

| Component | Model | Size | Context |
|---|---|---|---|
| LLM | `lukaspetrik/gemma3-tools:1b` | 581MB | 128K |
| Embedding | `jina/jina-embeddings-v2-small-en` | 49MB | 4K |
| Judge LLM | `llama3.2` | — | — |

> All models are served locally via Ollama. No internet connection required at inference time.

---

## Agentic Behaviours

The LangGraph state machine implements four autonomous behaviours:

1. **Query Rewriting** — Rewrites ambiguous or underspecified queries before retrieval to improve relevance.
2. **Retrieval Retries** — If retrieved context scores below a relevance threshold, the query is reformulated and retrieval is retried (max 2 retries).
3. **Multi-hop Decomposition** — Complex questions are broken into 2–3 sub-queries, each retrieved independently, with context aggregated before answer generation.
4. **Fallback Handling** — If context remains insufficient after retries, the system explicitly abstains rather than hallucinating an answer.

---

## Evaluation Metrics

Each response is scored by `llama3.2` on three metrics (0.0 – 1.0):

| Metric | Description |
|---|---|
| Answer Relevancy | How directly the generated answer addresses the question |
| Faithfulness | Whether the answer is grounded in retrieved context (no hallucination) |
| Contextual Precision | How relevant the retrieved context is to the question |

**Framework:** `ragas-llamaindex`

---

## Project Structure

```
.
├── main.py                  # Orchestrator — runs the full pipeline end to end
├── config.py                # Configuration dataclass, parsed from models.txt
├── models.txt               # Model + DB + path configuration (parameterized)
├── ingest.py                # Loads dataset, chunks, embeds, stores in pgvector
├── rag_agent.py             # LangGraph state machine — agentic RAG pipeline
├── run_pipeline.py          # Loads Q&A files, runs RAG for each question
├── evaluate.py              # RAGAS evaluation with llama3.2 judge
├── setup_db.sh              # PostgreSQL + pgvector table initialisation script
├── requirements.txt         # Python dependencies
├── dataset1.txt             # Source knowledge base (mental health in India)
├── Questions.txt            # Evaluation questions (numbered list)
├── Answers.txt              # Expected answers (numbered list)
├── output_rag.json          # Final output with answers + evaluation scores
└── results_intermediate.json # Intermediate RAG results (before evaluation)
```

---

## Setup

### Prerequisites

- Docker with Ollama installed and running
- PostgreSQL with the `pgvector` extension enabled
- Python 3.10+
- Node.js (optional, not required for core pipeline)

### 1. Configure credentials and paths

Open `models.txt` and replace all placeholder values:

```
DB_HOST=YOUR_DB_HOST
DB_NAME=YOUR_DB_NAME
DB_ROLE=YOUR_DB_ROLE
DB_USER=YOUR_DB_USER
DATASET_PATH=YOUR_DATASET_PATH
QUESTIONS_PATH=YOUR_QUESTIONS_PATH
ANSWERS_PATH=YOUR_ANSWERS_PATH
```

Also update the defaults in `config.py` to match.

> ⚠️ Never commit real credentials. Keep your local copy of these files separate from what you push to git.

### 2. Set up the database

```bash
export DB_PASSWORD=your_password_here
bash setup_db.sh
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This will:
1. Ingest `dataset1.txt` into pgvector
2. Run the agentic RAG pipeline over all questions
3. Evaluate results with RAGAS
4. Write output to `output_rag.json`

---

## Output Format

Each entry in `output_rag.json` contains:

```json
{
  "Question": "...",
  "Generated Answer": "...",
  "Expected Answer": "...",
  "Retrieved Context": ["...", "..."],
  "Evaluation scores": {
    "answer_relevancy": 0.8,
    "faithfulness": 0.7,
    "contextual_precision": 0.9
  },
  "Evaluation framework used": "ragas-llamaindex",
  "<small_llm> name": "lukaspetrik/gemma3-tools:1b",
  "<embedding_model> name": "jina/jina-embeddings-v2-small-en"
}
```

---

## Resuming from Intermediate Results

If the RAG pipeline has already run and `results_intermediate.json` exists, `main.py` will skip ingestion and pipeline execution and go straight to evaluation. Delete this file to force a full re-run.

---

## Multi-Instance Support

The system is fully parameterized for multi-instance runs. Each instance can use a different LLM and embedding model combination by editing `models.txt`. The judge LLM (`llama3.2`) is fixed across all instances.

---

## Dataset

The knowledge base (`dataset1.txt`) is an article on the mental health crisis in urban India, covering topics including the treatment gap, stigma, infrastructure gaps, digital mental health platforms, and relevant policy and legislation.
