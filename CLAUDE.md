# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning experimentation repository that combines:
1. **Traditional ML**: Linear regression implementation with scikit-learn
2. **LLM Integration**: Llama 3.1 model integration via Ollama for document Q&A and custom persona-based interactions
3. **Vector Database**: ChromaDB for semantic search and document retrieval
4. **RAG Pipeline**: LlamaIndex-based retrieval-augmented generation for PDF document processing

## Environment Setup

The project uses a virtual environment (.venv). Activate/deactivate:
```bash
# Activate
source .venv/bin/activate

# Deactivate
deactivate
```

## Installation

Install the package in development mode:
```bash
pip install -e .
```

This uses the custom `get_requirements()` function in setup.py that filters out `-e .` from requirements.txt.

## Key Dependencies

- **ML/Data**: pandas, numpy, scikit-learn
- **LLM/Embeddings**: llama-index, llama-index-llms-huggingface, sentence-transformers, transformers
- **Vector DB**: ChromaDB (via chromadb package)
- **Document Processing**: pypdf, langchain, langchain-core, langchain-community
- **Acceleration**: einops, accelerate

## Project Structure

```
src/
├── linear_regression/         # Traditional ML module
│   ├── datapreprocessing.py  # Synthetic data generation
│   ├── model_training.py     # Model training and evaluation
│   └── main.py              # Linear regression pipeline
└── database/
    └── c_db.py              # ChromaDB integration example

notebooks/                    # Jupyter notebooks for experimentation
├── notebook.ipynb           # Main RAG pipeline with LlamaIndex + Ollama
└── documnets.ipynb         # Additional experiments

documents/                   # PDF documents for RAG (e.g., hdfs.pdf)
```

## Running Components

### Linear Regression Pipeline
```bash
cd src/linear_regression
python main.py
```
This runs the complete pipeline: data preparation → training → evaluation, outputting the MSE.

### ChromaDB Example
```bash
python src/database/c_db.py
```
Demonstrates basic ChromaDB operations: collection creation, document insertion, and similarity search.

### RAG Document Q&A (Notebook)
The primary LLM workflow is in `notebooks/notebook.ipynb`:
1. Loads documents from `./documents/` directory
2. Uses HuggingFace embeddings (BAAI/bge-small-en-v1.5)
3. Creates VectorStoreIndex via LlamaIndex
4. Queries via Ollama-hosted Llama 3.1 model (requires Ollama running on localhost:11434)

### Custom Ollama Model (Health Coach)
The `Modelfile` defines a custom Llama 3.1 persona named "David" (personal health trainer).

Create and run the custom model:
```bash
ollama create david -f ./Modelfile
ollama run david
```

## Architecture Notes

### Linear Regression Module
- **datapreprocessing.py**: Generates synthetic linear data with noise using numpy
- **model_training.py**: Provides `train_linear_regression()` and `eval_model()` functions
- **main.py**: Orchestrates the pipeline; imports use relative imports (works when run from src/linear_regression/)

### RAG Pipeline (LlamaIndex)
The notebook implements a complete RAG system:
1. **Document Loading**: `SimpleDirectoryReader` loads PDFs from `documents/`
2. **Embeddings**: HuggingFace embeddings configured via `Settings.embed_model`
3. **LLM**: Ollama client pointing to locally-hosted Llama 3.1 (120s timeout)
4. **Prompt Template**: Uses Llama 3.1 chat template format with system/user/assistant tags
5. **Query Engine**: Standard LlamaIndex query engine with vector retrieval

The system prompt in the notebook configures the model as a Q&A assistant focused on accuracy based on provided context.

### ChromaDB Integration
Basic example in `c_db.py` shows:
- Client initialization
- Collection creation
- Document addition with IDs
- Query with automatic embedding and similarity search

## Development Notes

- The repository uses both `setup.py` and `pyproject.toml` (setup.py is actively used)
- Python 3.12 is the target version (per pyproject.toml)
- LLM operations require Ollama running locally with llama3.1 model available
- The notebooks expect documents in a `documents/` subdirectory relative to the notebook location
