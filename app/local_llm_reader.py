# local_llm_loader.py

from langchain_community.llms import LlamaCpp
import os

# Cache to avoid reloading models multiple times
model_cache = {}

def load_model(model_name):
    """
    Load a specific GGUF model using langchain_community.llms.LlamaCpp with hardware optimization.
    """
    print(f"[INFO] Loading model: {model_name}")

    # Model configurations
    model_configs = {
        "mistral": {
            "path": "models/mistral.gguf",
            "temperature": 0.6,
            "n_ctx": 4096,
            "max_tokens": 1024,
            "n_gpu_layers": 32,
            "n_threads": 8,
            "n_batch": 512,
        },
        "tinyllama": {
            "path": "models/tinyllama.gguf",
            "temperature": 0.6,
            "n_ctx": 2048,
            "max_tokens": 512,
            "n_gpu_layers": 25,
            "n_threads": 6,
            "n_batch": 512,
        },
        "qwen": {
            "path": "models/qwen1.8b.gguf",
            "temperature": 0.6,
            "n_ctx": 2048,
            "max_tokens": 512,
            "n_gpu_layers": 25,
            "n_threads": 6,
            "n_batch": 512,
        }
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    config = model_configs[model_name]
    model_path = config["path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the LLM with hardware-aware settings
    llm = LlamaCpp(
        model_path=model_path,
        temperature=config["temperature"],
        n_ctx=config["n_ctx"],
        max_tokens=config["max_tokens"],
        n_batch=config["n_batch"],
        n_threads=config["n_threads"],
        n_gpu_layers=config["n_gpu_layers"],
        verbose=False,
        use_mlock=True,
        use_mmap=True
    )

    print(f"[INFO] Model '{model_name}' loaded successfully.")
    return llm


def get_llm(model_name: str):
    """
    Retrieve a cached model instance, or load it if not already cached.
    """
    if model_name not in model_cache:
        model_cache[model_name] = load_model(model_name)
    return model_cache[model_name]


# Example usage:
# from local_llm_loader import get_llm

# llm = get_llm("tinyllama")  # or "mistral", "tinyllama", "qwen"
# print(llm)  # This will print the loaded model instance
# Note: Ensure that the model files are in the specified paths and that the required libraries are installed.
