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
    

def main():
    #Since the vector database is PDF specific, vector storage is ephemeral (in-memory vector database)
    embedding_function = HuggingFaceEmbeddingFunction(model_name=EMBED_MODEL_ID)
    vector_storage = chromadb.EphemeralClient()

    if not METADATA_PATH.exists():
        print("The metadata path does not exist")
        return
    
    #load metadata json document
    with METADATA_PATH.open("f", encoding="utf-8") as f:
        metadata = json.load(f)

    #list all chunks (never between two companies)
    chunk_files = list(CHUNKS_DIR.glob("*_chunks.json"))
    if not chunk_files:
        print("Error: There are no chunks to process.")
        return
    
    #loop over chunk files
    for chunk_file in chunk_files:
        company_id = chunk_file.stem.replace("_chunks", "")
        print(f"Processing report from {company_id}")

        collection_name=f"idx_{company_id}"
        collection = vector_storage.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        with chunk_file.open("r", encoding="utf-8") as f:
            report_chunks=json.load(f)

        """
        Syntax for collection.add: 
        collection.add(
        ids=["id1", "id2", "id3"],
        documents=["lorem ipsum...", "doc2", "doc3"],
        metadatas=[{"chapter": 3, "verse": 16}, {"chapter": 3, "verse": 5}, {"chapter": 29, "verse": 11}],
        )
        """

        collection.add(
            ids=[c["id"]for c in report_chunks],
            documents=[c["text"]for c in report_chunks],
            metadatas=[c["metadata"]for c in report_chunks]
        )

        extraction_results = {}

        #loop over every kpi within the company
        for kpi_key, kpi_info in metadata["KPIs"].items():
            print(f"Extracting {kpi_info["name"]}")

            query_results=collection.query(
                query_texts=[kpi_info["search_string"]],
                n_results=10
            )

            context_text ="\n---\n".join(query_results["documents"][0])

            prompt=f""" 
            Extract the following KPI from the sustainability report context provided below.
            
            KPI Target: {kpi_info['name']}
            Technical Definition: {kpi_info['definition']}
            Target Unit: {kpi_info['unit']}
            Calculation Logic: {kpi_info.get('calculation_logic', 'Direct extraction')}
            
            Glossary Reference:
            {json.dumps(metadata.get('glossary', {}), indent=2)}

            Sustainability Report Context:
            {context_text}
            """

            results = call_ollama_extraction(prompt)
            if results:
                extraction_results[kpi_key]=results.model_dump()

        
        output_path = RESULTS_DIR / f"{company_id}_extracted.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(extraction_results, f, indent=2)

        print("Success. Data exported to {output_path}")


if __name__ == "__main__":
    main()