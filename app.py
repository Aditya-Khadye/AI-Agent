import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title="RAG Agent",
    page_icon="🔍",
    layout="wide",
)


def apply_config(llm_provider, openai_key, anthropic_key, model_name):
    """Set environment variables and reload the settings singleton."""
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["LLM_PROVIDER"] = llm_provider
    if llm_provider == "openai":
        os.environ["OPENAI_MODEL"] = model_name
    else:
        os.environ["ANTHROPIC_MODEL"] = model_name

    from rag_agent.config import reload_settings
    reload_settings()


# --- Sidebar: Configuration & Document Ingestion ---

with st.sidebar:
    st.title("RAG Agent")
    st.caption("AI-powered document Q&A")

    st.divider()

    # LLM Configuration
    st.subheader("Configuration")

    llm_provider = st.selectbox("LLM Provider", ["openai", "anthropic"])
    openai_key = st.text_input("OpenAI API Key", type="password", help="Required for embeddings (always needed)")
    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Required only if using Anthropic as LLM provider",
    )

    if llm_provider == "openai":
        model_name = st.text_input("Model", value="gpt-4o")
    else:
        model_name = st.text_input("Model", value="claude-sonnet-4-20250514")

    # Apply configuration to environment and reload settings
    apply_config(llm_provider, openai_key, anthropic_key, model_name)

    st.divider()

    # Document Upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents", type="primary", use_container_width=True):
        if not openai_key:
            st.error("OpenAI API Key is required for embeddings.")
        else:
            with st.spinner("Ingesting documents..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for uf in uploaded_files:
                            file_path = Path(tmpdir) / uf.name
                            file_path.write_bytes(uf.getvalue())

                        from rag_agent.ingest import load_documents, chunk_documents, ingest as run_ingest
                        docs = load_documents(tmpdir)
                        chunks = chunk_documents(docs)
                        run_ingest(tmpdir)

                    st.success(f"Ingested {len(docs)} document(s) ({len(chunks)} chunks)")
                    st.session_state["docs_ingested"] = True
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # Show ingestion status
    index_path = os.path.join("./vectorstore", "index.faiss")
    if os.path.exists(index_path):
        st.info("Vector store is ready.")
    else:
        st.warning("No documents ingested yet. Upload files above.")

    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        st.rerun()


# --- Main Area: Chat Interface ---

st.header("Chat with your Documents")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Validate config
    if not openai_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
        st.stop()
    if llm_provider == "anthropic" and not anthropic_key:
        st.error("Please enter your Anthropic API Key in the sidebar.")
        st.stop()
    if not os.path.exists(index_path):
        st.error("No documents ingested yet. Upload and ingest documents first.")
        st.stop()

    # Ensure settings are fresh
    apply_config(llm_provider, openai_key, anthropic_key, model_name)

    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from rag_agent.agent import build_agent

                agent = build_agent()
                result = agent.invoke({
                    "input": prompt,
                    "chat_history": st.session_state["chat_history"],
                })
                answer = result["output"]

                st.session_state["chat_history"].append(HumanMessage(content=prompt))
                st.session_state["chat_history"].append(AIMessage(content=answer))
                st.session_state["messages"].append({"role": "assistant", "content": answer})

                st.markdown(answer)
            except FileNotFoundError:
                st.error("No documents ingested yet. Upload and ingest documents first.")
            except Exception as e:
                st.error(f"Error: {e}")
