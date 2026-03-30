import tempfile
from unittest.mock import patch, MagicMock

import pytest


def test_get_retriever_no_index():
    """Test that get_retriever raises when no index exists."""
    from rag_agent.retriever import get_retriever

    with patch("rag_agent.retriever.settings") as mock_settings:
        mock_settings.vectorstore_dir = "/nonexistent/path"
        with pytest.raises(FileNotFoundError, match="No vector store found"):
            get_retriever()


def test_get_vectorstore_no_index():
    """Test that get_vectorstore raises when no index exists."""
    from rag_agent.retriever import get_vectorstore

    with patch("rag_agent.retriever.settings") as mock_settings:
        mock_settings.vectorstore_dir = "/nonexistent/path"
        with pytest.raises(FileNotFoundError, match="No vector store found"):
            get_vectorstore()
