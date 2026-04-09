import json
from pathlib import Path
import chromadb

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama
from pydantic import BaseModel

from sentence_transformers import CrossEncoder

#Define the schema for the response
class KpiBasic(BaseModel):
    value: float
    unit: str
    page_reference: int
    quote: str

#Path definitions and model selection
METADATA_PATH = Path("configs/kpi_metadata.json")
CHUNKS_DIR = Path("data/chunks")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_ID = "qwen3-embedding:8b"
LLM_MODEL = "qwen2.5:14b"
BATCH_SIZE = 10 #for embedding
N_RESULTS = 40 #for reranking
FINAL_N_RESULTS = 5

def call_ollama_extraction(prompt):
    """
    Via Ollama, this function queries the LLM based on the prompt given in the parameter that was passed
    """
    try:
        response = ollama.chat(
            model = LLM_MODEL,
            messages=[{"role":"user", "content":prompt}],
            format=KpiBasic.model_json_schema(),
            options = {"temperature" : 0}
        )
        return KpiBasic.model_validate_json(response["message"]["content"])
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def main():
    #Initialize reranker model
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')


    #Since the vector database is PDF specific, vector storage is ephemeral (in-memory vector database)

    #fix this:
    embedding_function = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL_ID,
        url="http://localhost:11434/api/embeddings"
    )
    vector_storage = chromadb.EphemeralClient()

    if not METADATA_PATH.exists():
        print("The metadata path does not exist")
        return
    
    #load metadata json document
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    #list all chunks (never between two companies)
    chunk_files = list(CHUNKS_DIR.glob("*_chunks.json"))
    if not chunk_files:
        print("Error: There are no chunks to process.")
        return
    
    #loop over chunk files
    for chunk_file in chunk_files:
        company_id = chunk_file.stem.replace("_chunks", "")

        output_path = RESULTS_DIR / f"{company_id}_extracted.json"
        if output_path.exists():
            print(f"Skipping {company_id} (results already exist at {output_path})")
            continue
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

        for i in range(0, len(report_chunks), BATCH_SIZE):
            batch = report_chunks[i : i + BATCH_SIZE]
            
            # Clean metadata to remove empty lists (ChromaDB requirement)
            cleaned_metadatas = []
            for meta in [c["metadata"] for c in batch]:
                cleaned = {k: v for k, v in meta.items() if not (isinstance(v, list) and len(v) == 0)}
                cleaned_metadatas.append(cleaned)
            
            collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=cleaned_metadatas
            )
            print(f"Embedded batch {i//BATCH_SIZE + 1}/{(len(report_chunks)-1)//BATCH_SIZE + 1}")

        extraction_results = {}

        #loop over every kpi within the company
        for kpi_key, kpi_info in metadata["KPIs"].items():
            print(f"Extracting {kpi_info["name"]}")

            #Broad retrieval pre reranker
            query_results=collection.query(
                query_texts=[kpi_info["search_string"]],
                n_results=N_RESULTS
            )

            #reranker logic
            candidates = query_results["documents"][0]
            metadatas = query_results["metadatas"][0]
            
            #Pairs for reranker: [(query, chunk1), (query, chunk2), ...]
            pairs = [[kpi_info["search_string"], doc] for doc in candidates]
            scores = reranker.predict(pairs)
            
            # Sort by score descending and take top K
            ranked_indices = scores.argsort()[::-1][:FINAL_N_RESULTS]
            
            reranked_docs = [candidates[i] for i in ranked_indices]
            reranked_metas = [metadatas[i] for i in ranked_indices]

            context_segments = []
            for doc, meta in zip(reranked_docs, reranked_metas):
                context_segments.append(f"--- SOURCE: Page {meta.get('page_numbers')} ---\n{doc}")
            context_text = "\n\n".join(context_segments)

            prompt = f""" 
            You are a strict ESG auditor. Extract the EXACT value for {kpi_info['name']} from the following MARKDOWN context for the year 2024.
            Either a KPI is found, then fill in value, unit, page_reference, quote. Or if the KPI is not found, value, unit, page_reference, quote must all return null.

            ### AUDIT CONSTRAINTS:
            1. MARKDOWN TABLE LOGIC: If the data is in a table, identify the correct column (Year/Category) and row (KPI Name).
            2. UNIT RIGIDITY: The value MUST be in '{kpi_info['unit']}'. If the table lists {kpi_info['name']} in a different unit, return 'value': null.
            3. HIERARCHY: Only extract company-wide/group-level data. Ignore regional or subsidiary-specific values.
            4. ADMIT DEFEAT: If the exact KPI is missing or ambiguous in the context, return 'value': null, 'unit': null, 'page_reference':null, 'quote': null.

            ### KPI SPECIFICATION:
            - Name: {kpi_info['name']}
            - Definition: {kpi_info['definition']}
            {f"- Logic: {kpi_info['calculation_logic']}" if 'calculation_logic' in kpi_info else ""}

            ### CONTEXT (MARKDOWN TABLES & TEXT):
            {context_text}
            """

            results = call_ollama_extraction(prompt)
            if results:
                extraction_results[kpi_key]=results.model_dump()

        
        output_path = RESULTS_DIR / f"{company_id}_extracted.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(extraction_results, f, indent=2)

        print(f"Success. Data exported to {output_path}")

        vector_storage.delete_collection(name=collection_name)


if __name__ == "__main__":
    main()