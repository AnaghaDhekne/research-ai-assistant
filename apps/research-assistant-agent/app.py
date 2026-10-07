import streamlit as st
import mlflow
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient

# ---------------------------------------------------------------------------
# MLflow tracing setup (Unity Catalog) — best-effort
# ---------------------------------------------------------------------------
# The app runs as a service principal that may not have access to the user's
# personal MLflow experiment.  We attempt to set up UC tracing; if it fails
# the app still works — only trace persistence is skipped.
MLFLOW_READY = False
mlflow.set_tracking_uri("databricks")

# Try the user's UC-backed experiment first; fall back to a self-created
# workspace experiment that the service principal owns.
try:
    from mlflow.entities.trace_location import UnityCatalog
    mlflow.set_experiment(
        "/Users/anaghagdhekne@gmail.com/experiment_1_uc",
        trace_location=UnityCatalog(
            catalog_name="main",
            schema_name="default",
            table_prefix="experiment_1",
        ),
    )
    mlflow.openai.autolog()
    MLFLOW_READY = True
except Exception:
    pass

if not MLFLOW_READY:
    try:
        mlflow.set_experiment("research_assistant_agent")
        mlflow.openai.autolog()
        MLFLOW_READY = True
    except Exception as e:
        MLFLOW_READY = False
        _MLFLOW_ERR = str(e)

# ---------------------------------------------------------------------------
# Databricks Foundation Model API client
# ---------------------------------------------------------------------------
w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()
MODEL = "databricks-meta-llama-3-3-70b-instruct"

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {"topic": "Unity Catalog", "content": "Unity Catalog is Databricks' unified governance solution for data and AI assets. It provides centralized access control, auditing, lineage tracking, and data discovery across all workspaces in a Databricks account."},
    {"topic": "MLflow Tracing", "content": "MLflow Tracing is a feature that records the execution flow of GenAI applications, including LLM calls, retrievers, chains, and custom functions. It captures inputs, outputs, metadata, and timing for each span, enabling debugging and evaluation."},
    {"topic": "Delta Lake", "content": "Delta Lake is an open-source storage layer that brings ACID transactions to Apache Spark and big data workloads. It provides ACID guarantees, schema enforcement, time travel, and unified batch and stream processing."},
    {"topic": "Databricks SQL", "content": "Databricks SQL (DBSQL) is a serverless data warehouse on Databricks that lets you run SQL queries on your data lake using standard SQL. It includes SQL warehouses for compute, a SQL editor, query history, dashboards, and alerts."},
    {"topic": "Lakeflow Jobs", "content": "Lakeflow Jobs (formerly Databricks Workflows) is the Databricks orchestration service for scheduling and running data pipelines, notebooks, Python scripts, and SQL queries. It supports multi-task DAGs, retries, conditional execution, and multiple trigger types."},
    {"topic": "Vector Search", "content": "Databricks Vector Search is a serverless similarity search engine that lets you store and query vector embeddings alongside your structured data. It supports real-time vector search for RAG applications and automatic syncing from Delta tables."},
    {"topic": "Model Serving", "content": "Databricks Model Serving provides serverless, real-time inference for ML and AI models. It supports Foundation Model APIs (pay-per-token), external models (OpenAI, Anthropic), and custom models registered in Unity Catalog."},
    {"topic": "Auto Loader", "content": "Auto Loader is a Databricks feature for incrementally and efficiently processing new data files as they arrive in cloud storage. It supports JSON, CSV, Parquet, and other formats, with schema inference and evolution and exactly-once processing guarantees."},
    {"topic": "Databricks Apps", "content": "Databricks Apps is a serverless application hosting platform that lets you deploy data and AI applications directly from your Databricks workspace. It supports popular frameworks like Flask, FastAPI, Streamlit, and Gradio, with built-in Databricks authentication."},
    {"topic": "MLflow", "content": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It includes experiment tracking, model registry, model deployment, and GenAI tracing for LLM observability. On Databricks, MLflow is fully managed with Unity Catalog integration."},
    {"topic": "Unity Catalog Volumes", "content": "Unity Catalog Volumes provide governed storage for non-tabular data files in Unity Catalog. Volumes live within a UC schema and let you store, manage, and access files (CSV, JSON, images, models) with the same access control and auditing as tables."},
    {"topic": "Photon", "content": "Photon is Databricks' native vectorized query engine that accelerates Spark SQL and DataFrame operations. It uses C++ to vectorize CPU-intensive workloads, providing significant performance improvements for SQL queries, aggregations, and joins without requiring code changes."},
]

# ---------------------------------------------------------------------------
# Agent functions (traced with MLflow)
# ---------------------------------------------------------------------------
@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_context(query: str) -> list[str]:
    """Retrieve relevant documents from the knowledge base."""
    query_lower = query.lower()
    retrieved = []
    for doc in KNOWLEDGE_BASE:
        if doc["topic"].lower() in query_lower or any(
            word in query_lower for word in doc["topic"].lower().split()
        ):
            retrieved.append(f"[{doc['topic']}] {doc['content']}")
    return retrieved


@mlflow.trace(span_type=SpanType.LLM)
def generate_response(query: str, context: list[str], history: str = "") -> str:
    """Generate a response using the Databricks Foundation Model API."""
    context_text = "\n\n".join(context)
    if not context:
        return "I don't have information about that topic in my knowledge base. I can help with questions about Unity Catalog, MLflow, Delta Lake, Databricks SQL, Lakeflow Jobs, Vector Search, Model Serving, Auto Loader, Databricks Apps, Unity Catalog Volumes, and Photon."
    prompt = (
        "You are a helpful research assistant. Answer the question using ONLY the context provided below. "
        "If the context does not contain information relevant to the question, say: "
        "'I don\'t have information about that topic in my knowledge base.' "
        "Do not use your own knowledge to answer."
        f"\n\nContext:\n{context_text}"
        f"{history}"
        f"\n\nQuestion: {query}\n\nAnswer:"
    )
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content


@mlflow.trace(span_type=SpanType.CHAIN)
def chat_agent(message: str, conversation_history: list[tuple[str, str]]) -> str:
    """Multi-turn chat agent with conversation memory and MLflow tracing."""
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": "streamlit_user",
            "mlflow.trace.session": "streamlit_session",
        }
    )
    context = retrieve_context(message)
    history_context = ""
    if conversation_history:
        history_lines = []
        for prev_msg, prev_resp in conversation_history[-5:]:
            history_lines.append(f"User: {prev_msg}\nAssistant: {prev_resp}")
        history_context = "\n\nPrevious conversation:\n" + "\n".join(history_lines)
    answer = generate_response(message, context, history_context)
    return answer


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Research Assistant", page_icon="🔍", layout="wide")

st.title("Research Assistant Agent")
st.markdown("Ask questions about Unity Catalog, MLflow, Delta Lake, and more.")

with st.sidebar:
    st.header("About")
    st.markdown(f"**Model:** `{MODEL}`")
    st.markdown(f"**Knowledge base:** {len(KNOWLEDGE_BASE)} topics")
    st.markdown("**Topics:**")
    for doc in KNOWLEDGE_BASE:
        st.markdown(f"- {doc['topic']}")
    if MLFLOW_READY:
        st.markdown("**Tracing:** ✅ MLflow → Unity Catalog")
    else:
        st.markdown("**Tracing:** ⚠️ Disabled (service principal lacks experiment access)")
        with st.expander("Details"):
            st.code(_MLFLOW_ERR[:500])
    st.markdown("---")
    st.caption("Built with Streamlit on Databricks Apps.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Unity Catalog, MLflow, Delta Lake…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    conversation_history = [
        (m["content"], n["content"])
        for m, n in zip(
            [m for m in st.session_state.messages if m["role"] == "user"],
            [m for m in st.session_state.messages if m["role"] == "assistant"],
        )
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = chat_agent(prompt, conversation_history)
            except Exception as e:
                reply = f"⚠️ Error: {e}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
