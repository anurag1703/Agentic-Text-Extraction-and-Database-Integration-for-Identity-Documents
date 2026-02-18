# Agentic Text Extraction and Database Integration for Identity Documents

End-to-end PDF document processing pipeline: preprocessing → classification → vision extraction → validation → storage.

## Overview
- Purpose: Process ID-style PDFs and extract structured data through a configurable pipeline.
- Entry point: `tools/orchestrator.py` — a CLI wrapper that runs the full pipeline for a single PDF.

## Prerequisites
- Python 3.11 or newer
- System dependency: `poppler` (required by `pdf2image`)
  - Windows: `install poppler` or download a build and add its `bin` to `PATH`.
- Optional: GPU and CUDA drivers for faster `torch` operations.

Model files: `models/Id_Classifier.pt` should be present in the repository.

## Install and setup
1. Create and activate a virtual environment.

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (cmd):
```cmd
python -m venv .venv
.\.venv\Scripts\activate
```



2. Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `pdf2image` complains about missing `poppler`, add `poppler/bin` to your `PATH` or set `POPPLER_PATH` environment variable.

## Configuration
- Inspect `config/agent_config.py` to tune timeouts and endpoints.
- Ollama extraction defaults to `http://localhost:11434`. If you do not run Ollama locally, skip extraction with `--no-extract`.
- DB: uses `aiosqlite`. Provide `--db-path` to store results or use `--no-store` to skip storage.

## Running the pipeline
`tools/orchestrator.py` is the recommended entrypoint. It orchestrates preprocessing, classification, extraction, validation, and storage.

Basic run (single PDF):
```bash
set SAMPLE_ORCH_PDF= path of your PDF.
python tools/orchestrator.py
```


## Outputs
- Preprocessed images: `outputs/preprocessed/` (or folder passed to `--save-images`)
- Pipeline JSON: `outputs/runs/` (or folder passed to `--save-json`)
- Database file: path passed with `--db-path`

## Troubleshooting
- `pdf2image` errors: install `poppler` and ensure `PATH` or `POPPLER_PATH` points to the `poppler/bin` folder.
- Extraction fails: Ollama service may be down — start Ollama or run with `--no-extract`.
- Slow / OOM on large PDFs: reduce pages via `--max-pages` or run on a machine with more memory/GPU.
- Missing model file: ensure `models/Id_Classifier.pt` exists or pass `--model <path>`.
- Database errors: provide writable path with `--db-path` or use `--no-store` during troubleshooting.

## Where to look when editing or debugging
- Preprocessing: `tools/document_preprocessor.py`
- Classification: `tools/document_classifier.py`
- Vision extraction: `tools/vision_extractor.py`
- Validation: `tools/data_validator.py`
- Database: `tools/db_engine.py`
- Orchestration / CLI: `tools/orchestrator.py` (recommended entrypoint)

## Path to update in Scripts.
- Document_Preprocessor: poppler path in the PreprocessorConfig Class
- Document_Classifier: model path in the ClassifierConfig Class

---
Generated: Feburary 2026
