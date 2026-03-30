import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag_agent.ingest import load_documents, chunk_documents


def test_load_txt_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("This is a test document with some content for testing.")
        f.flush()
        docs = load_documents(f.name)

    assert len(docs) >= 1
    assert "test document" in docs[0].page_content


def test_load_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            Path(tmpdir, f"doc{i}.txt").write_text(f"Document {i} content.")

        docs = load_documents(tmpdir)
        assert len(docs) == 3


def test_load_unsupported_file():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"unsupported")
        f.flush()
        try:
            load_documents(f.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported" in str(e)


def test_chunk_documents():
    from langchain_core.documents import Document

    docs = [Document(page_content="word " * 500, metadata={"source": "test.txt"})]

    with patch("rag_agent.ingest.settings") as mock_settings:
        mock_settings.chunk_size = 100
        mock_settings.chunk_overlap = 20
        chunks = chunk_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"
