import os

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from rag_agent.config import settings
from rag_agent.ingest import get_embeddings


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    """Load the persisted FAISS index and return a retriever."""
    index_path = os.path.join(settings.vectorstore_dir, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "No vector store found. Run 'rag-agent ingest <path>' first "
            "to ingest documents."
        )

    vectorstore = FAISS.load_local(
        settings.vectorstore_dir,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_vectorstore() -> FAISS:
    """Load and return the FAISS vector store directly."""
    index_path = os.path.join(settings.vectorstore_dir, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "No vector store found. Run 'rag-agent ingest <path>' first."
        )

    return FAISS.load_local(
        settings.vectorstore_dir,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
