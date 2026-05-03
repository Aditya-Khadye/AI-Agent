from langchain_core.language_models import BaseChatModel

import rag_agent.config as config


def get_llm() -> BaseChatModel:
    """Create and return the configured LLM instance."""
    s = config.settings
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.openai_model,
            api_key=s.openai_api_key,
        )
    elif s.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=s.google_model,
            google_api_key=s.google_api_key,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=s.anthropic_model,
            api_key=s.anthropic_api_key,
        )
