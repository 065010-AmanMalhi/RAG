# 🤖 Agentic RAG System — Multi-Instance Evaluation Framework

![PYTHON](https://img.shields.io/badge/PYTHON-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![LANGCHAIN](https://img.shields.io/badge/LANGCHAIN-0.3+-brightgreen?style=for-the-badge&logo=chainlink&logoColor=white)
![LANGGRAPH](https://img.shields.io/badge/LANGGRAPH-0.4+-orange?style=for-the-badge)
![OLLAMA](https://img.shields.io/badge/OLLAMA-LOCAL-black?style=for-the-badge)
![PGVECTOR](https://img.shields.io/badge/PGVECTOR-POSTGRES-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![DOCKER](https://img.shields.io/badge/DOCKER-UBUNTU-2496ED?style=for-the-badge&logo=docker&logoColor=white)

An Agentic Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph, running on Docker (Ubuntu) with PostgreSQL/pgvector as the vector store and Ollama for fully local model inference. The project is structured as two parallel instances of the same pipeline, each using a different evaluation framework, enabling direct comparison of evaluation methodologies on identical RAG output.

[![RAG 1 - RAGAS](https://img.shields.io/badge/📁%20RAG%201-RAGAS%20Evaluation-blue?style=for-the-badge)](./RAG%201%20-%20RAGAS)
[![RAG 2 - Prometheus](https://img.shields.io/badge/📁%20RAG%202-Prometheus%20Evaluation-orange?style=for-the-badge)](./RAG%202%20-%20Prometheus)

---

## 📁 Repository Structure

```
RAG/
├── RAG 1 - RAGAS/          # Instance 1 — evaluated with RAGAS framework
└── RAG 2 - Prometheus/     # Instance 2 — evaluated with Prometheus framework
```

Both instances share the same agentic pipeline architecture, dataset, questions, and LLM. Only the evaluation sub-system differs.

---

## 📚 Dataset

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

> **Potential research directions:**
> - Expanding the dataset with NMHS 2015-16 raw reports, NIMHANS publications, or WHO South-East Asia mental health data for a richer retrieval corpus
> - Testing cross-lingual retrieval using Hindi or regional language documents
> - Benchmarking RAG performance on policy vs. epidemiological vs. case-based questions separately

---

## 🏗️ System Architecture

Both instances share the following architecture:

### Sub-System 1 — Agentic RAG Pipeline

Built with LangChain + LangGraph. A state machine manages four autonomous behaviours:

| Behaviour | Description |
|---|---|
| 🔄 Query Rewriting | Rewrites ambiguous queries before retrieval for better semantic match |
| 🔁 Retrieval Retries | Retries with a reformulated query if context scores below relevance threshold |
| 🔀 Multi-hop Decomposition | Breaks complex questions into sub-queries, aggregates retrieved context |
| 🚫 Fallback Handling | Abstains explicitly rather than hallucinating when context is insufficient |

### Sub-System 2 — Evaluation

A fixed judge LLM (`llama3.2:3b`) scores each generated answer on three metrics:

| Metric | Description |
|---|---|
| Answer Relevancy | How directly the answer addresses the question (0.0–1.0) |
| Faithfulness | Whether the answer is grounded in retrieved context — no hallucination (0.0–1.0) |
| Contextual Precision | How relevant the retrieved context is to the question (0.0–1.0) |

---

## 📦 Instances

### RAG 1 — RAGAS Evaluation

![LLM](https://img.shields.io/badge/LLM-gemma3--tools:1b-blue?style=flat-square)
![SIZE](https://img.shields.io/badge/SIZE-581MB-green?style=flat-square)
![CONTEXT](https://img.shields.io/badge/CONTEXT-128K-orange?style=flat-square)
![EVAL](https://img.shields.io/badge/EVAL-ragas--llamaindex-purple?style=flat-square)

RAGAS evaluates outputs by prompting the judge LLM to return direct JSON scores (0.0–1.0) for all three metrics in a single call.

### RAG 2 — Prometheus Evaluation

![LLM](https://img.shields.io/badge/LLM-gemma3--tools:1b-blue?style=flat-square)
![SIZE](https://img.shields.io/badge/SIZE-815MB-green?style=flat-square)
![CONTEXT](https://img.shields.io/badge/CONTEXT-32K-orange?style=flat-square)
![EVAL](https://img.shields.io/badge/EVAL-prometheus-red?style=flat-square)

Prometheus uses structured rubric-based absolute grading. Each metric is evaluated independently using a detailed 1–5 scoring rubric (inspired by the Prometheus 2 paper), then normalised to 0.0–1.0. This produces more interpretable, fine-grained feedback per metric compared to direct JSON scoring.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM Inference | Ollama (fully local, no API calls) |
| Orchestration | LangChain + LangGraph |
| Vector Store | PostgreSQL + pgvector extension |
| Embeddings | Ollama embedding models |
| Runtime | Docker on Ubuntu |
| Language | Python 3.10+ |

---

## 🚀 Local Setup Guide

> **Supported:** Windows, Linux. Mac not tested.
> **Required:** Docker Desktop installed and running before you begin.

### Step 1 — Clone the repository

```bash
git clone https://github.com/065010-AmanMalhi/RAG.git
cd RAG
```

### Step 2 — Set up Docker (Ubuntu container with Ollama)

This system runs inside a Docker container on Ubuntu. Make sure Docker Desktop is running, then start your container. Ollama must be running inside the container and all models must be pulled before executing the pipeline.

Pull the required models inside your container:

```bash
ollama pull lukaspetrik/gemma3-tools:1b
ollama pull jina/jina-embeddings-v2-small-en
ollama pull llama3.2:3b
```

Verify all models are available:
```bash
ollama list
```

### Step 3 — Set up PostgreSQL with pgvector

Make sure PostgreSQL is running with the `pgvector` extension enabled. Then initialise the tables by running the setup script from inside your instance folder:

```bash
export DB_PASSWORD=your_password_here
bash setup_db.sh
```

### Step 4 — Set up Python environment

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv langchain
source langchain/bin/activate        # Linux
# OR
langchain\Scripts\activate           # Windows

pip install -U langchain langchain-ollama langchain-community langgraph
pip install -r requirements.txt
```

### Step 5 — Configure credentials and paths

Open `models.txt` and `config.py` in your chosen instance folder and replace all `YOUR_*` placeholders:

```
DB_HOST=YOUR_DB_HOST        → your PostgreSQL host
DB_NAME=YOUR_DB_NAME        → your database name
DB_USER=YOUR_DB_USER        → your database user
DB_PASSWORD=YOUR_DB_PASSWORD → your database password
DATASET_PATH=YOUR_DATASET_PATH   → path to dataset1.txt
QUESTIONS_PATH=YOUR_QUESTIONS_PATH → path to Questions.txt
ANSWERS_PATH=YOUR_ANSWERS_PATH   → path to Answers.txt
```

> ⚠️ Never commit your real credentials. Keep your local copy separate from what you push to git.

### Step 6 — Run the pipeline

```bash
cd "RAG 1 - RAGAS"   # or "RAG 2 - Prometheus"
python main.py
```

The pipeline will:
1. Ingest `dataset1.txt` into the pgvector store
2. Run the agentic RAG pipeline over all 13 questions
3. Evaluate results using the configured framework
4. Write final output to `output_rag.json`

> **Tip:** If `results_intermediate.json` already exists in the folder, `main.py` will skip ingestion and pipeline execution and jump straight to evaluation. Delete it to force a full re-run from scratch.

---

## 📄 Output Format

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

## 📝 Notes

- All models run locally via Ollama — no internet connection required at inference time
- The judge LLM (`llama3.2:3b`) is fixed across both instances to ensure evaluation consistency
- DB credentials and local file paths are intentionally left as placeholders in this repo — never commit real credentials
