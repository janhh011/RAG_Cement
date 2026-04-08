import json
from pathlib import Path
from io import BytesIO
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import(
    PdfPipelineOptions, TableStructureOptions, TableFormerMode, granite_picture_description)
import gc #garbage collection
from pypdf import PdfReader, PdfWriter #splitting pdf for better memory usage
from docling.datamodel.document import DoclingDocument

""" import logging
import traceback
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("docling") """

#This script looks for sustainability reports in data/raw , converts with Docling into a JSON format to be later converted into DoclingDocument, and then pasts the results into data/output.
#1.Path definition, 2.Iterate reports and convert with Docling (skip if target JSON/Markdown already exists)

BATCH_SIZE = 10

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
    pipeline_options.images_scale = 0.3
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

            #Batching the PDF
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            docs_to_merge = []

            for start_page in range(0, total_pages, BATCH_SIZE):
                end_page = min(start_page + BATCH_SIZE, total_pages)
                print(f"Converting pages {start_page +1} to {end_page}...")

                writer = PdfWriter()
                for page_idx in range(start_page, end_page):
                    writer.add_page(reader.pages[page_idx])

                batch_tmp = temp_dir / f"temp_p{start_page}.pdf"
                with open(batch_tmp, "wb") as f:
                    writer.write(f)

                #Convert partial PDF
                result = converter.convert(batch_tmp)

                # Merge into master document
                docs_to_merge.append(result.document)
                
                # Cleanup batch
                batch_tmp.unlink()
                del result
                gc.collect()

            #save full document
            if docs_to_merge:
                print("Merging all pages into final document...")
                full_doc = DoclingDocument.concatenate(docs=docs_to_merge)

                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(full_doc.export_to_dict(), f)

                md_output_path = output_dir / f"{pdf_path.stem}.md"
                with md_output_path.open("w", encoding="utf-8") as f:
                    f.write(full_doc.export_to_markdown())
                
                print(f"Success: {output_path.name} and {md_output_path.name}")

        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
        finally:
            # Clear master doc from memory for next file
            docs_to_merge = []
            gc.collect()


if __name__ == "__main__":
    main()
