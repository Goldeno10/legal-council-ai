import os
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

def get_model(model=None, temperature=0):
    """
    Switch between Local Ollama and Cloud DeepSeek based on ENV.
    """
    if os.getenv("USE_LOCAL_AI") == "true":
        # Google-Standard: Use a high-context local model like Llama 3.1 or 3.2
        return ChatOllama(
            model=model or "llama3.2", 
            temperature=temperature,
            format="json" # Local models often need explicit JSON formatting
        )
    else:
        return ChatDeepSeek(
            model=model or "deepseek-reasoner", 
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature
        )
