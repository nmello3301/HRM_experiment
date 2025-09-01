import os
import time
from typing import List, Optional, Dict, Any
import requests

class LLMClient:
    """
    Unified interface for local (Ollama) or OpenAI-compatible endpoints.
    Usage:
        llm = LLMClient(backend="ollama", model="llama3.1:8b-instruct-q4_K_M")
        out = llm.generate("You are a helpful assistant. Say 'hello'.")
    """
    def __init__(self, backend: str = "ollama", model: Optional[str] = None, temperature: float = 0.2):
        self.backend = backend
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        if backend == "ollama":
            self.base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif backend == "openai":
            self.base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("backend must be 'ollama' or 'openai'")

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        if self.backend == "ollama":
            url = f"{self.base}/api/generate"
            payload = {
                "model": self.model,
                "prompt": (system + "\n\n" + prompt) if system else prompt,
                "options": {"temperature": self.temperature},
                "stream": False,
            }
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        else:  # openai-compatible chat
            url = f"{self.base}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
