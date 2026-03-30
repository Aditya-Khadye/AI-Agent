from langchain_core.tools import tool

from rag_agent.llm import get_llm
from rag_agent.retriever import get_retriever, get_vectorstore


@tool
def search_documents(query: str) -> str:
    """Search ingested documents for information relevant to the query.

    Use this tool to find specific information from the document library.
    Returns the most relevant passages along with their source metadata.
    """
    retriever = get_retriever(k=4)
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found for this query."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        location = f"{source}" + (f" (page {page})" if page else "")
        results.append(f"[{i}] Source: {location}\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


@tool
def retrieve_document_info(filename: str = "") -> str:
    """List available documents in the knowledge base.

    Optionally filter by filename substring. Use this to understand
    what documents are available before searching.
    """
    try:
        vectorstore = get_vectorstore()
    except FileNotFoundError:
        return "No documents have been ingested yet."

    sources: set[str] = set()
    for doc_id in vectorstore.docstore._dict:
        doc = vectorstore.docstore._dict[doc_id]
        source = doc.metadata.get("source", "unknown")
        if not filename or filename.lower() in source.lower():
            sources.add(source)

    if not sources:
        return f"No documents found matching '{filename}'." if filename else "No documents ingested."

    source_list = "\n".join(f"- {s}" for s in sorted(sources))
    return f"Available documents ({len(sources)}):\n{source_list}"


@tool
def summarize_text(text: str) -> str:
    """Summarize a long piece of text into concise key points.

    Use this after retrieving documents to condense the information
    into a shorter, more digestible format.
    """
    llm = get_llm()
    response = llm.invoke(
        f"Summarize the following text into concise key points:\n\n{text}"
    )
    return response.content
