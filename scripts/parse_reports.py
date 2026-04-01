import os
from pathlib import Path
from docling.document_converter import DocumentConverter

#This is a test run for the Docling converter. It looks for sustainability reports in reports/raw , converts with Docling, and then pasts the results into reports/output.

def main():
    input_dir = Path("reports/raw")
    output_dir = Path("reports/output")

    print("Initializing Docling...")
    converter = DocumentConverter()

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}. Please add reports.")
        return False

    print(f"Found {len(pdf_files)} reports. Starting conversion...")
    print(f"File list: {pdf_files}")

    for pdf_path in pdf_files:
        output_path = output_dir/f"{pdf_path.stem}.md"
        if output_path.exists():
            print(f"Skipping {pdf_path.name} (already converted)")
            continue
        try:
            print(f"Converting: {pdf_path.stem}...")

            result = converter.convert(pdf_path)

            with output_path.open("w", encoding="utf-8") as f:
                f.write(result.document.export_to_markdown())
                
            print(f"Success: Saved to {output_path.name}")


        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
            



if __name__ == "__main__":
    main()
