from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configuration Management
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    WEAVIATE_URL: str = "http://weaviate:8080"
    API_KEY: str = Field("no_key")
    SECURITY_ENABLED: bool = Field(False)

    model_config = ConfigDict(env_file=".env")


settings = Settings()
