from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = Field(default="openai", pattern="^(openai|anthropic|google)$")
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    google_model: str = "gemini-2.0-flash"
    embedding_model: str = "text-embedding-3-small"
    docs_dir: str = "./data"
    vectorstore_dir: str = "./vectorstore"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def validate_api_keys(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'. "
                "Set it in your .env file or environment."
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'. "
                "Set it in your .env file or environment."
            )
        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required when LLM_PROVIDER is 'google'. "
                "Set it in your .env file or environment."
            )


settings = Settings()


def reload_settings() -> Settings:
    """Reload settings from current environment variables."""
    global settings
    settings = Settings()
    return settings
