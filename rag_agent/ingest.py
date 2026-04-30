import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_agent.config as config

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_documents(path: str) -> list[Document]:
    """Load all supported documents from a file or directory."""
    p = Path(path)
    files: list[Path] = []

    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
        else:
            raise ValueError(
                f"Unsupported file type: {p.suffix}. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
    elif p.is_dir():
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(p.rglob(f"*{ext}"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not files:
        raise ValueError(f"No supported documents found in {path}")

    docs: list[Document] = []
    for f in sorted(files):
        loader_cls = LOADERS[f.suffix.lower()]
        loader = loader_cls(str(f))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = str(f)
        docs.extend(loaded)

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.settings.chunk_size,
        chunk_overlap=config.settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_embeddings() -> Embeddings:
    """Create the embeddings model. Uses OpenAI if key is available, otherwise HuggingFace."""
    if config.settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.settings.embedding_model,
            api_key=config.settings.openai_api_key,
        )

    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def ingest(path: str) -> FAISS:
    """Full ingestion pipeline: load -> chunk -> embed -> store.

    If a vector store already exists, new documents are added to it.
    """
    docs = load_documents(path)
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()

    index_path = os.path.join(config.settings.vectorstore_dir, "index.faiss")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            config.settings.vectorstore_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(config.settings.vectorstore_dir)
    return vectorstore
