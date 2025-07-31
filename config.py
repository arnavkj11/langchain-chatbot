"""
Environment Configuration Module
Handles loading and validation of environment variables securely.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Configuration class that handles environment variables securely."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    
    # SerpAPI Configuration
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    
    # LangChain Configuration
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "langchain-chatbot")
    
    # Application Configuration
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # FastAPI Configuration
    HOST: str = os.getenv("HOST", "localhost")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    
    # CORS Configuration
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required environment variables are set."""
        required_vars = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("SERPAPI_API_KEY", cls.SERPAPI_API_KEY),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value or var_value.startswith("your_") or var_value.startswith("sk-your_"):
                missing_vars.append(var_name)
        
        if missing_vars:
            logger.error(f"Missing or invalid environment variables: {', '.join(missing_vars)}")
            logger.error("Please check your .env file and ensure all API keys are properly set.")
            return False
        
        logger.info("Environment configuration validated successfully.")
        return True
    
    @classmethod
    def get_safe_config(cls) -> dict:
        """Return a dictionary of non-sensitive configuration values for logging."""
        return {
            "OPENAI_MODEL": cls.OPENAI_MODEL,
            "OPENAI_TEMPERATURE": cls.OPENAI_TEMPERATURE,
            "OPENAI_MAX_TOKENS": cls.OPENAI_MAX_TOKENS,
            "LANGCHAIN_TRACING_V2": cls.LANGCHAIN_TRACING_V2,
            "LANGCHAIN_PROJECT": cls.LANGCHAIN_PROJECT,
            "DEBUG": cls.DEBUG,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "HOST": cls.HOST,
            "PORT": cls.PORT,
            "FRONTEND_URL": cls.FRONTEND_URL,
            "OPENAI_API_KEY": "***" if cls.OPENAI_API_KEY else "NOT_SET",
            "SERPAPI_API_KEY": "***" if cls.SERPAPI_API_KEY else "NOT_SET",
            "LANGCHAIN_API_KEY": "***" if cls.LANGCHAIN_API_KEY else "NOT_SET",
        }

# Create a global config instance
config = Config()

# Validate configuration on import
if __name__ == "__main__":
    print("Configuration validation:")
    print(f"Valid: {config.validate_config()}")
    print("Safe config:")
    for key, value in config.get_safe_config().items():
        print(f"  {key}: {value}")
