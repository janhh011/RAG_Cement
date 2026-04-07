import json
from pathlib import Path
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import(
    PdfPipelineOptions, TableStructureOptions, TableFormerMode, granite_picture_description)
import gc #garbage collection
from pypdf import PdfReader, PdfWriter #splitting pdf for better memory usage


#This script looks for sustainability reports in data/raw , converts with Docling into a JSON format to be later converted into DoclingDocument, and then pasts the results into data/output.
#1.Path definition, 2.Iterate reports and convert with Docling (skip if target JSON/Markdown already exists)


def main():
    #(1)
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    #Change: PDFPipelineOptions() (specifically designed for procesisng PDFs and image-rich documents)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    #Picture description with Doclings PDF_PipelineOptions()
    pipeline_options.do_picture_description = True
    #for now: using pre defined granite-vision-3.1-2b-preview from IBM
    pipeline_options.picture_description_options = (granite_picture_description)
    pipeline_options.picture_description_options.prompt = (
    "Describe the image as clearly as possible. In the process of doing so, name every KPI that is mentioned in the picture and all necessary information to classify that KPI. Be accurate."
    )
    pipeline_options.images_scale = 0.6
    pipeline_options.generate_picture_images = False

    #For scanned reports:
    #pipeline_options.do_ocr = True

    print("Initializing Docling...")
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options = pipeline_options)}
        )

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

            del result
            gc.collect

        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
            gc.collect



if __name__ == "__main__":
    main()
