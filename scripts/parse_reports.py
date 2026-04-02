import json
from pathlib import Path
from docling.document_converter import DocumentConverter

#This script looks for sustainability reports in data/raw , converts with Docling into a JSON format to be later converted into DoclingDocument, and then pasts the results into data/output.
#1.Path definition, 2.Iterate reports and convert with Docling (skip if target JSON/Markdown already exists)

def main():
    #(1)
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Docling...")
    converter = DocumentConverter()

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}. Please add reports.")
        return False

    print(f"Found {len(pdf_files)} reports. Starting conversion...")
    print(f"File list: {pdf_files}")

    #(2)
    for pdf_path in pdf_files:
        output_path = output_dir/f"{pdf_path.stem}.json"
        if output_path.exists():
            print(f"Skipping {pdf_path.name} (already converted)")
            continue
        try:
            print(f"Converting: {pdf_path.stem}...")

            result = converter.convert(pdf_path)

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result.document.export_to_dict(), f)
                
            print(f"Success: Saved to {output_path.name}")


        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
            



if __name__ == "__main__":
    main()
