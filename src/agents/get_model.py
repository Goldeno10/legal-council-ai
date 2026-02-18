import os
from typing import Union
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek


def get_model(
    model=None,
    format=None,
    temperature: Union[int, float] = 0.7,
    structured: bool = False
):
    if os.getenv("USE_LOCAL_AI") == "true":
        ollama = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"),
            model="deepseek-v3.1:671b-cloud", # "deepseek-v3.2:cloud",
            temperature=temperature,
            format=format if format else None,
        )

        return ollama
    
    else:
        deepseek = ChatDeepSeek(
            model=model or "deepseek-reasoner",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            temperature=temperature,
        )
        if structured:
            pass
        return deepseek
