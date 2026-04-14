from llama_index.llms.google_genai import GoogleGenAI
from src.utils.config import Settings

# later you can add OpenAI / Ollama imports


def get_llm(provider: str | None = None, model: str | None = None):
    provider = provider or Settings.LLM_PROVIDER

    if provider == "gemini":
        if not Settings.GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY in environment variables")

        return GoogleGenAI(
            model=model or Settings.GEMINI_MODEL,
            api_key=Settings.GEMINI_API_KEY
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")