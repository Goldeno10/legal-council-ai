import os
from typing import Union
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

def get_model(model=None, temperature: Union[int, float]=0):
    """
    Switch between Local Ollama and Cloud DeepSeek based on ENV.
    """
    if os.getenv("USE_LOCAL_AI") == "true":
        # Google-Standard: Use a high-context local model like Llama 3.1 or 3.2
        return ChatOllama(
            base_url= os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"),
            model="deepseek-v3.2:cloud",
            # api_key=os.getenv("OLLAMA_API_KEY"),
            # model=model or "llama3.2", 
            temperature=temperature,
            format="json" # Local models often need explicit JSON formatting
        )
    else:
        return ChatDeepSeek(
            model=model or "deepseek-reasoner", 
            api_key=os.getenv("DEEPSEEK_API_KEY", ""), # type: ignore
            temperature=temperature
        )
