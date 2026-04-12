# RAG for Cement Industry KPI Extraction

This project is part of my master’s thesis at Technical University Berlin and University College London. It implements a Retrieval-Augmented Generation (RAG) system to automate the extraction of Key Performance Indicators (KPIs) from cement industry sustainability reports.

## System Architecture
The system uses **local components only** (local embeddings and LLMs) to ensure development remains cost-effective and flexible during the research phase. Depending on final accuracy results, the local models may be swapped for cloud-based API models in future iterations.

## Pipeline Structure
The core processing steps are located in the `scripts/` directory:

* **`parse_reports.py`**: Uses docling to parse PDF reports into structured data.
* **`chunking_reports.py`**: Handles the segmentation of parsed reports.
* **`extract_kpi.py`**: A local LLM extracts data points based on the definitions in **`configs/kpi_metadata.json`**.

## Data and Results
Preliminary extraction results are available in **`data/*_extracted.json`**. Current test cases include:
* **Heidelberg Materials**: 2024 Aggregated Report (~300 pages).
* **Schwenk**: 2021 Sustainability Report.

## Optimization
Optimization potentials and refinements are tracked in the **Issues** tab. These are prioritized based on performance comparisons against a manual ground truth dataset.