import json
from pathlib import Path
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import(
    PdfPipelineOptions, TableStructureOptions, TableFormerMode, granite_picture_description)

""" import logging
import traceback
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("docling") """

#This script looks for sustainability reports in data/raw , converts with Docling into a JSON format to be later converted into DoclingDocument, and then pasts the results into data/output.
#1.Path definition, 2.Iterate reports and convert with Docling (skip if target JSON/Markdown already exists)

def main():
    #(1)
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    temp_dir = Path("data/temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    #Change: PDFPipelineOptions() (specifically designed for procesisng PDFs and image-rich documents)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    #Picture description with Doclings PDF_PipelineOptions() --> currently turned OFF because memory allocation of >24GB for vision model crashes Python
    pipeline_options.do_picture_description = False
    #for now: using pre defined granite-vision-3.1-2b-preview from IBM
    pipeline_options.picture_description_options = (granite_picture_description)
    pipeline_options.picture_description_options.prompt = (
    "Describe the image as clearly as possible. In the process of doing so, name every KPI that is mentioned in the picture and all necessary information to classify that KPI. Be accurate."
    )
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_picture_images = False

    #For scanned reports:
    #pipeline_options.do_ocr = True

    print("Initializing Docling...")
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options = pipeline_options)}
        )

    pdf_files = [
        f for f in input_dir.glob("*.pdf") 
        if not (output_dir / f"{f.stem}.json").exists()
    ]

    if not pdf_files:
        print(f"No new PDFs found for processing.")
        return

    print(f"Found {len(pdf_files)} reports. Starting conversion...")
    print(f"File list: {pdf_files}")

    #(2)
    conv_results = converter.convert_all(pdf_files)

    for result in conv_results:
        source_name = result.input.file.name
        
        if result.document is None:
            print(f"Failed to convert: {source_name}")
            continue

        try:
            # Export JSON
            json_output = output_dir / f"{Path(source_name).stem}.json"
            with json_output.open("w", encoding="utf-8") as f:
                json.dump(result.document.export_to_dict(), f)

            # Export Markdown (Optimized for LLM context)
            md_output = output_dir / f"{Path(source_name).stem}.md"
            with md_output.open("w", encoding="utf-8") as f:
                f.write(result.document.export_to_markdown())

            print(f"Successfully processed: {source_name}")

        except Exception as e:
            print(f"Error saving {source_name}: {e}")

if __name__ == "__main__":
    main()
