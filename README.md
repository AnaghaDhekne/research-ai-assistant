# Research AI Assistant

A grounded Databricks research assistant that answers questions from a small curated Databricks knowledge base. This project is the first stage of a three-project progression from basic retrieval-augmented generation to semantic RAG and multi-agent orchestration.

## What it demonstrates

- Databricks Foundation Model API using `databricks-meta-llama-3-3-70b-instruct`
- Lightweight lexical/keyword retrieval over a curated knowledge base
- Grounded generation: the model is instructed to answer only from retrieved context
- Multi-turn conversation history
- MLflow tracing for the retriever, LLM call, and overall chain
- Best-effort Unity Catalog-backed trace storage with workspace-experiment fallback
- Streamlit UI hosted with Databricks Apps

## Architecture

```text
User question
     |
Keyword Retriever
     |
Databricks Knowledge Base
     |
Foundation Model
     |
Grounded Response
```

The knowledge base covers topics including Unity Catalog, MLflow, Delta Lake, Databricks SQL, Lakeflow Jobs, Vector Search, Model Serving, Auto Loader, Databricks Apps, Unity Catalog Volumes, and Photon.

## Grounding behavior

The retriever performs simple lexical matching against topic names. Retrieved entries are passed to the model as context, and the generation prompt explicitly prevents the model from answering from unsupported outside knowledge.

This is intentionally a baseline implementation. It provides a simple architecture against which the later semantic-retrieval and agentic versions can be compared.

## Project structure

```text
apps/
└── research-assistant-agent/
    ├── app.py
    ├── app.yaml
    └── requirements.txt
```

- `app.py` — retrieval, model invocation, MLflow tracing, conversation state, and Streamlit UI
- `app.yaml` — Databricks Apps launch configuration
- `requirements.txt` — application dependencies

## Project progression

**Stage 1 — Research AI Assistant (this repository)**  
Lexical retrieval + grounded generation + MLflow tracing.

**Stage 2 — [RAG Research Agent](https://github.com/AnaghaDhekne/rag-research-agent)**  
Adds semantic Vector Search, AI Gateway with model fallback, Unity Catalog inference logging, token/latency capture, and a larger Python knowledge base.

**Stage 3 — [Multi-Agent Research System](https://github.com/AnaghaDhekne/multi-agent-research-system)**  
Adds an LLM router, specialized Python and Databricks agents, parallel cross-domain retrieval, evidence reconciliation, and grounded synthesis.

## Current scope

This project intentionally uses a small in-code knowledge base and lexical retrieval. It is a learning/baseline implementation rather than a production search system; the later repositories build on these limitations explicitly.
