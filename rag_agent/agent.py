from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag_agent.llm import get_llm
from rag_agent.tools import retrieve_document_info, search_documents, summarize_text

SYSTEM_PROMPT = (
    "You are a knowledgeable AI assistant with access to a document library. "
    "Use your tools to search for relevant information before answering questions. "
    "Always cite which document and section your information comes from. "
    "If you cannot find relevant information in the documents, say so honestly "
    "rather than making up an answer."
)


def build_agent() -> AgentExecutor:
    """Build and return the RAG agent executor."""
    llm = get_llm()
    tools = [search_documents, retrieve_document_info, summarize_text]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
