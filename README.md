# Agentic RAG System — Multi-Instance Evaluation Framework

An Agentic Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph, running on Docker (Ubuntu) with PostgreSQL/pgvector as the vector store and Ollama for fully local model inference. The project is structured as two parallel instances of the same pipeline, each using a different evaluation framework, enabling direct comparison of evaluation methodologies on identical RAG output.

---

## Repository Structure

```
RAG/
├── RAG 1 - RAGAS/          # Instance 1 — evaluated with RAGAS framework
└── RAG 2 - Prometheus/     # Instance 2 — evaluated with Prometheus framework
```

Both instances share the same agentic pipeline architecture, dataset, questions, and LLM. Only the evaluation sub-system differs.

---

## Dataset

The knowledge base used across both instances is a long-form article on the **mental health crisis in urban India**. The article covers:

- **Epidemiology** — prevalence data from the National Mental Health Survey (NMHS) 2015-16, conducted by NIMHANS. Approximately 150 million Indians need active mental health intervention at any given time, yet fewer than 30 million ever receive care.
- **The Treatment Gap** — defined as the difference between those who need care and those who receive it. Stands at nearly 80% for common mental disorders (depression, anxiety) and over 70% for severe conditions (schizophrenia, bipolar disorder).
- **Urban Mental Health Burden** — how rapid urbanisation, nuclear family structures, competitive employment markets, and housing insecurity create a uniquely urban psychological risk profile distinct from rural areas.
- **Stigma** — the primary barrier to care, operating at three levels: individual (internalised shame, avg. 5–12 year delay in help-seeking), family/community (marriage prospects, employment discrimination), and institutional (GPs misattributing symptoms).
- **Gender Disparities** — women's symptoms frequently misattributed to hormonal or domestic causes, leading to underdiagnosis of conditions like bipolar disorder and psychosis.
- **Infrastructure Gaps** — India has approximately 0.3 psychiatrists per 100,000 population vs a global average of 1.7. Public hospitals see psychiatrists managing 80–150 patients per outpatient session.
- **Digital Mental Health** — platforms like iCall, YourDOST, InnerHour, and Wysa, their limitations in reaching the urban poor, and the TELE-MANAS initiative (launched 2022) as a free, multilingual, phone-call-based alternative.
- **Policy & Legislation** — the Mental Healthcare Act 2017 (decriminalisation of attempted suicide, right to mental healthcare, advance directives), the National Mental Health Policy 2014, and the NMHP/DMHP programmes.

This dataset was chosen for its density of verifiable factual claims, mixed question types (factual recall, inference, case-based reasoning, true/false), and real-world relevance — making it a strong testbed for evaluating RAG retrieval quality and answer faithfulness.

**Potential research directions:**
- Expanding the dataset with NMHS 2015-16 raw reports, NIMHANS publications, or WHO South-East Asia mental health data for a richer retrieval corpus
- Testing cross-lingual retrieval using Hindi or regional language documents
- Benchmarking RAG performance on policy vs. epidemiological vs. case-based questions separately

---

## System Architecture

Both instances share the following architecture:

### Sub-System 1 — Agentic RAG Pipeline

Built with LangChain + LangGraph. A state machine manages four autonomous behaviours:

| Behaviour | Description |
|---|---|
| Query Rewriting | Rewrites ambiguous queries before retrieval for better semantic match |
| Retrieval Retries | Retries with a reformulated query if context scores below relevance threshold |
| Multi-hop Decomposition | Breaks complex questions into sub-queries, aggregates retrieved context |
| Fallback Handling | Abstains explicitly rather than hallucinating when context is insufficient |

### Sub-System 2 — Evaluation

A fixed judge LLM (`llama3.2:3b`) scores each generated answer on three metrics:

| Metric | Description |
|---|---|
| Answer Relevancy | How directly the answer addresses the question (0.0–1.0) |
| Faithfulness | Whether the answer is grounded in retrieved context — no hallucination (0.0–1.0) |
| Contextual Precision | How relevant the retrieved context is to the question (0.0–1.0) |

---

## Instances

### RAG 1 — RAGAS Evaluation (`RAG 1 - RAGAS/`)

| Component | Detail |
|---|---|
| LLM | `lukaspetrik/gemma3-tools:1b` — 581MB, 128K context |
| Embedding | `jina/jina-embeddings-v2-small-en` — 49MB |
| Judge LLM | `llama3.2:3b` |
| Evaluation Framework | `ragas-llamaindex` |

RAGAS evaluates outputs by prompting the judge LLM to return direct JSON scores (0.0–1.0) for all three metrics in a single call.

### RAG 2 — Prometheus Evaluation (`RAG 2 - Prometheus/`)

| Component | Detail |
|---|---|
| LLM | `lukaspetrik/gemma3-tools:1b` — 815MB, 32K context |
| Embedding | `jina/jina-embeddings-v2-small-en` — 66MB |
| Judge LLM | `llama3.2:3b` |
| Evaluation Framework | `prometheus` |

Prometheus uses structured rubric-based absolute grading. Each metric is evaluated independently using a detailed 1–5 scoring rubric (inspired by the Prometheus 2 paper), then normalised to 0.0–1.0. This produces more interpretable, fine-grained feedback per metric compared to direct JSON scoring.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Inference | Ollama (fully local, no API calls) |
| Orchestration | LangChain + LangGraph |
| Vector Store | PostgreSQL + pgvector extension |
| Embeddings | Ollama embedding models |
| Runtime | Docker on Ubuntu |
| Language | Python 3.10+ |

---

## Setup

### Prerequisites
- Docker with Ollama installed and models pre-loaded
- PostgreSQL with `pgvector` extension enabled
- Python 3.10+

### 1. Configure credentials and paths
In each instance folder, open `models.txt` and `config.py` and replace all `YOUR_*` placeholders with your actual DB credentials and file paths.

### 2. Set up the database
```bash
export DB_PASSWORD=your_password_here
bash setup_db.sh
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
```bash
python main.py
```

---

## Output

Each instance produces an `output_rag.json` file where every entry contains:

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

## Notes

- All models run locally via Ollama — no internet connection required at inference time
- The judge LLM (`llama3.2:3b`) is fixed across both instances to ensure evaluation consistency
- If `results_intermediate.json` exists in an instance folder, `main.py` skips ingestion and pipeline execution and jumps straight to evaluation — delete it to force a full re-run
- DB credentials and local file paths are intentionally left as placeholders in this repo — never commit real credentials
