"""
Shared configuration and helpers for all lab steps
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# LangSmith Configuration
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-lab")
LANGSMITH_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.txt"
RAGAS_REPORT_PATH = DATA_DIR / "ragas_report.json"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)


def get_llm():
    """Create and return a ChatOpenAI instance"""
    from langchain_openai import ChatOpenAI
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        temperature=0.7,
    )


def get_embeddings():
    """Create and return an OpenAIEmbeddings instance"""
    from langchain_openai import OpenAIEmbeddings
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_EMBEDDING_MODEL,
    )


def setup_langsmith():
    """Set LangSmith environment variables"""
    if not LANGSMITH_API_KEY:
        print("⚠️  LANGSMITH_API_KEY not set - tracing will be disabled")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
        print(f"✅ LangSmith tracing enabled for project: {LANGSMITH_PROJECT}")


def get_langsmith_client():
    """Create and return a LangSmith Client"""
    from langsmith import Client
    
    if not LANGSMITH_API_KEY:
        raise ValueError("LANGSMITH_API_KEY not set in .env file")
    
    return Client(api_key=LANGSMITH_API_KEY)


if __name__ == "__main__":
    print("=== Lab Configuration Check ===")
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"LangSmith project:  {LANGSMITH_PROJECT}")
    print(f"OpenAI model:       {OPENAI_MODEL}")
    print(f"Embedding model:    {OPENAI_EMBEDDING_MODEL}")
    print(f"Knowledge base:     {KNOWLEDGE_BASE_PATH}")
    print(f"RAGAS report:       {RAGAS_REPORT_PATH}")
    print()
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set")
    else:
        print("✅ OPENAI_API_KEY loaded")
    
    if not LANGSMITH_API_KEY:
        print("⚠️  LANGSMITH_API_KEY not set (tracing disabled)")
    else:
        print("✅ LANGSMITH_API_KEY loaded")
    
    print("\n✅ Configuration verified!")
