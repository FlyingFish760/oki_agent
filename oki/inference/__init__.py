"""推理引擎层：默认走 Ollama。"""

from .ollama_client import OllamaClient

__all__ = ["OllamaClient"]
