from dotenv import load_dotenv
import os

# Load environment variables once
load_dotenv()


class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    #Gemini settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    # OPENAI settings - for future 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    # OLLAMA settings - for future
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
    EMBEDDER_PROVIDER = os.getenv("EMBEDDER_PROVIDER", "bge")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    
    @staticmethod
    def validate():
        if not Settings.GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY in environment variables")