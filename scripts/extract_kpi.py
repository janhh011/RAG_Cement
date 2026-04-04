import json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
import ollama
from pydantic import Basemodel

#Define the schema for the reponse
class KpiBasic(Basemodel):
    value: float | None = None
    unit: str
    page_reference: str|None = None
    quote: str

#Path definitions and model selection
METADATA_PATH = Path("configs/kpi_metadata.json")
CHUNKS_DIR = Path("data/chunks")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL = "xyz"

def call_ollama_extraction(prompt):
    """
    Via Ollama, this function queries the LLM based on the prompt given in the parameter that was passed
    """
    try:
        reponse = ollama.chat(
            model = LLM_MODEL,
            messages=[{"role":"user", "content":prompt}],
            format = "json",
            options = {"temperature" : 0}
        )
        return KpiBasic.model_validate_json(reponse.message.content)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
