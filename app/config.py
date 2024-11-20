from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


# Configuration Management
class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(
        None,
    )
    COHERE_API_KEY: Optional[str] = Field(
        None,
    )

    # Weaviate Configuration
    WEAVIATE_URL: str = Field("http://weaviate:8080")

    # Security
    API_KEY: str = Field(
        "no_key",
    )
    SECURITY_ENABLED: bool = Field(
        False,
    )

    # LLM Configuration
    LLM_PROVIDER: str = Field(
        "openai",  # Options: 'openai', 'cohere'
    )
    OPENAI_MODEL: str = Field(
        "gpt-3.5-turbo",
    )
    COHERE_MODEL: str = Field(
        "command-xlarge-nightly",
    )
    SYSTEM_PROMPT: str = Field(
        "You are a helpful assistant.",
    )

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = Field(
        "local",  # Options: 'openai', 'cohere', 'local'
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        "text-embedding-ada-002",
    )
    COHERE_EMBEDDING_MODEL: str = Field(
        "embed-english-light-v2.0",
    )

    model_config = ConfigDict(env_file=".env")


settings = Settings()
