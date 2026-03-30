# AI-Agent

A RAG (Retrieval-Augmented Generation) agent that ingests documents and answers questions using LangChain, FAISS, and your choice of OpenAI or Anthropic LLMs.

## Features

- **Document ingestion** - Load PDF, TXT, and Markdown files into a local FAISS vector store
- **Intelligent retrieval** - Semantic search across your document library
- **Agent with tools** - Search documents, list available files, and summarize content
- **Multi-turn chat** - Interactive conversation with memory of prior exchanges
- **Dual LLM support** - Use OpenAI (GPT-4o) or Anthropic (Claude) as the reasoning model
- **Streamlit web UI** - Browser-based chat interface for showcasing and demos

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```
LLM_PROVIDER=openai          # or "anthropic"
OPENAI_API_KEY=sk-...        # Required (used for embeddings)
ANTHROPIC_API_KEY=sk-ant-... # Required if using Anthropic LLM
```

### 3. Ingest Documents

```bash
rag-agent ingest ./data/           # Ingest all docs in a directory
rag-agent ingest ./report.pdf      # Ingest a single file
```

### 4. Ask Questions

```bash
rag-agent ask "What are the key findings in the report?"
```

### 5. Interactive Chat

```bash
rag-agent chat
```

## Web UI (Streamlit)

Launch the web interface for a visual demo:

```bash
streamlit run app.py
```

This opens a browser with:
- Sidebar for API key configuration, model selection, and document upload
- Chat interface with multi-turn conversation
- One-click document ingestion

To deploy on **Streamlit Community Cloud**, push to GitHub and connect at [share.streamlit.io](https://share.streamlit.io). Add your API keys in the Streamlit secrets management.

## Project Structure

```
rag_agent/
  config.py      - Settings from environment variables
  ingest.py      - Document loading, chunking, and FAISS indexing
  retriever.py   - Vector store retrieval interface
  llm.py         - LLM factory (OpenAI / Anthropic)
  tools.py       - Agent tools: search, document info, summarize
  agent.py       - Agent construction with LangChain
  cli.py         - CLI entry point (click + rich)
app.py           - Streamlit web UI
data/            - Default directory for source documents
vectorstore/     - Persisted FAISS index
tests/           - Unit tests
```

## Supported File Types

| Extension | Loader |
|-----------|--------|
| `.pdf`    | PyPDFLoader |
| `.txt`    | TextLoader |
| `.md`     | TextLoader |

## Configuration

All settings can be configured via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider (`openai` or `anthropic`) |
| `OPENAI_API_KEY` | - | OpenAI API key (always required for embeddings) |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (required if using Anthropic) |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHUNK_SIZE` | `1000` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DOCS_DIR` | `./data` | Default documents directory |
| `VECTORSTORE_DIR` | `./vectorstore` | FAISS index storage directory |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
