import json
from pathlib import Path
from docling_core.types.doc import DoclingDocument
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

#This script loads the extracted JSON documents, applies the layout-aware HybridChunker from Docling, and then saves the chunks
#1.Path definition, 2.Select correct tokenizer for specific model, 3.Configure Hybridchunker, 4. Process data with chunker 
#To Do: select model and define model_id + max_tokens

EMBED_MODEL_ID = "xyz"
MAX_TOKENS = 0

def main():
    #(1)
    processed_dir = Path("data/processed")
    output_dir = Path("data/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    #(2)
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        tokenizer = HuggingFaceTokenizer(tokenizer = hf_tokenizer, max_tokens = MAX_TOKENS)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

    #(3)
    #merge_peers=True combines small sections to maximize LLM context
    #repeat_table_header=True makes sure table column names are appended to split rows
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens = MAX_TOKENS,
        merge_peers=True,
        repeat_table_header = True
    )

    #(4)
    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        print(f"No processed JSON files found in {processed_dir}.")

    for json_path in json_files:
        print(f"Processing {json_path.name}")

        if f"{json_path.stem}.json".exists():
            print(f"Target file {json_path.stem} already exists. Skipping...")
            continue

        try:
            doc = DoclingDocument.load_from_json(json_path)
            
            chunk_iter = chunker.chunk(doc)
            processed_chunks = []
            
            for i, chunk in enumerate(chunk_iter):
                text_for_llm = chunker.contextualize(chunk)

                processed_chunks.append({
                    "id": f"{json_path.stem}_{i}",
                    "text": text_for_llm,
                    "metadata": {
                        "source": json_path.name,
                        "headings": chunk.meta.export_json_dict().get("headings", []),
                        "page_numbers": [prov.page_no for prov in chunk.meta.doc_items if hasattr(prov, 'page_no')]
                    }
                })

                print(f"Success. Created {len(processed_chunks)} chunks for {json_path.stem}.")

        except Exception as e:
            print(f"Error {e}")