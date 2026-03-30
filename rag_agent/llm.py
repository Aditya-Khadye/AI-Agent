from langchain_core.language_models import BaseChatModel

from rag_agent.config import settings


def get_llm() -> BaseChatModel:
    """Create and return the configured LLM instance."""
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
        )
