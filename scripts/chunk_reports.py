import json
from pathlib import Path
from docling_core.types.doc import DoclingDocument
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

#This script loads the extracted JSON documents, applies the layout-aware HybridChunker from Docling, and then saves the chunks
#1.Path definition, 2.Select correct tokenizer for specific model, 3.Configure Hybridchunker, 4. Process data with chunker 
#To Do: select model and define model_id + max_tokens

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MAX_MODEL_TOKENS = 32768
MAX_CHUNKING_TOKENS = 1024 #model max tokens is 32768, but lowering for meaningful vectors

def main():
    #(1)
    processed_dir = Path("data/processed")
    output_dir = Path("data/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    #(2)
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens = MAX_MODEL_TOKENS)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    #(3)
    #merge_peers=True combines small sections to maximize LLM context
    #repeat_table_header=True makes sure table column names are appended to split rows
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens = MAX_CHUNKING_TOKENS,
        merge_peers=True,
        repeat_table_header = True
    )

    #(4)
    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        print(f"No processed JSON files found in {processed_dir}.")

    for json_path in json_files:
        print(f"Processing {json_path.name}")

        output_path = output_dir / f"{json_path.stem}_chunks.json"
        if output_path.exists():
            print(f"Skipping {json_path.name} (already chunked)")
            continue

        try:
            doc = DoclingDocument.load_from_json(json_path)
            
            chunk_iter = chunker.chunk(doc)
            processed_chunks = []
            
            for i, chunk in enumerate(chunk_iter):
                #Inlcude heading path into text for llm
                headings = chunk.meta.export_json_dict().get("headings", [])
                heading_path = " > ".join(headings)
                text_for_llm = f"CONTEXT PATH: {heading_path}\n{chunker.contextualize(chunk)}"

                page_set = {prov.page_no for item in chunk.meta.doc_items for prov in item.prov if hasattr(prov, "page_no")}

                processed_chunks.append({
                    "id": f"{json_path.stem}_{i}",
                    "text": text_for_llm,
                    "metadata": {
                        "source": json_path.name,
                        "headings": chunk.meta.export_json_dict().get("headings", []),
                        "page_numbers": sorted(list(page_set))
                    }
                })


            with output_path.open("w", encoding="utf-8") as f:
                json.dump(processed_chunks, f, indent=2)
                
            print(f"Success: Created {len(processed_chunks)} chunks for {json_path.stem}")

        except Exception as e:
            print(f"Error processing {json_path.name}: {e}")


if __name__ == "__main__":
    main()