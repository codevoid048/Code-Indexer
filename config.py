import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config(BaseSettings):
    """Configuration settings for the Code Indexer."""
    
    # API Settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = False
    
    # Groq API Settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "llama-3.1-70b-versatile"
    
    # Index Settings
    index_dir: str = "index_data"
    cache_size: int = 1000
    
    # File Processing Settings
    include_patterns: List[str] = [
        "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
        "*.java", "*.go", "*.rs", "*.cpp", "*.c", "*.h", "*.hpp",
        "*.php", "*.rb", "*.swift", "*.kt", "*.scala"
    ]
    
    exclude_patterns: List[str] = [
        "node_modules/*", "__pycache__/*", ".git/*", "*.pyc", 
        "*.pyo", "*.so", "*.dll", ".venv/*", "venv/*",
        "build/*", "dist/*", "*.egg-info/*", ".pytest_cache/*",
        "coverage/*", ".coverage", "*.log", "*.tmp"
    ]
    
    # Search Settings
    max_search_results: int = 100
    default_search_limit: int = 20
    include_context_by_default: bool = True
    
    # File Watcher Settings
    debounce_delay: float = 1.0  # seconds
    max_file_size: int = 1024 * 1024  # 1MB
    
    # Logging Settings
    log_level: str = "INFO"
    log_file: str = "code_indexer.log"
    
    class Config:
        env_prefix = "CODE_INDEXER_"
        env_file = ".env"
        case_sensitive = False


# Global config instance
config = Config()


# Environment variable helpers
def get_groq_api_key() -> str:
    """Get Groq API key from environment or config."""
    return config.groq_api_key or os.getenv("GROQ_API_KEY", "")


def get_index_directory() -> str:
    """Get index directory path."""
    return os.path.abspath(config.index_dir)


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    extensions = []
    for pattern in config.include_patterns:
        if pattern.startswith("*."):
            extensions.append(pattern[2:])  # Remove '*.'
    return extensions


def is_file_supported(file_path: str) -> bool:
    """Check if a file is supported based on its extension."""
    from pathlib import Path
    
    file_ext = Path(file_path).suffix.lower()
    if not file_ext:
        return False
    
    # Remove leading dot and check
    ext = file_ext[1:]  # Remove '.'
    return ext in get_supported_extensions()


# Development/Debug settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"