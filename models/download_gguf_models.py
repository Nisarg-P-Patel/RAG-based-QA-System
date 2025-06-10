import os
import requests

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

GGUF_MODELS = {
    "mistral": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "local_name": "mistral.gguf"
    },
    "tinyllama": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "local_name": "tinyllama.gguf"
    }
}

def download_with_requests(name, config):
    local_path = os.path.join(MODEL_DIR, config["local_name"])
    if os.path.exists(local_path):
        print(f"[SKIP] {name} already exists at {local_path}")
        return

    print(f"[INFO] Downloading {name} from {config['url']} ...")
    try:
        response = requests.get(config["url"], stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"[✅] {name} downloaded successfully to {local_path}")
    except requests.exceptions.RequestException as e:
        print(f"[❌ ERROR] Failed to download {name}: {e}")

def main():
    print("=== Downloading GGUF Models ===")
    for name, config in GGUF_MODELS.items():
        download_with_requests(name, config)
    print("=== GGUF Model Download Complete ===")

if __name__ == "__main__":
    main()
