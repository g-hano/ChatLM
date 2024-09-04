import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_NAME = "nomic-ai/nomic-embed-text-v1.5"
TEMPERATURE = 0.0
CONTEXT_WINDOW = 32_000
TOP_K = 2
CHUNK_SIZE = 512
CHUNK_OVERLAP = 10
SYSTEM_PROMPT = """[INST] You are a helpful assistant that answers
user questions using the documents provided.
Your answer MUST be in markdown format without
any prefixes like 'assistant:' [/INST]"""
running_on = "cuda" if torch.cuda.is_available() else "cpu"
print(f"---Running on {running_on}---")
DEVICE = "auto"

# Flask app configuration
UPLOAD_FOLDER = 'uploads'
