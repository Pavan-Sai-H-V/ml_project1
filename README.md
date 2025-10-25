# Machine Learning Projects Repository

A collection of machine learning experiments ranging from traditional ML to advanced neural architectures with external memory.

## Projects

### 1. Linear Regression (`src/linear_regression/`)
Traditional machine learning pipeline with scikit-learn:
- Synthetic data generation with noise
- Model training and evaluation
- MSE-based performance metrics

**Run:**
```bash
cd src/linear_regression
python main.py
```

### 2. RAG with Llama 3.1 (`notebooks/notebook.ipynb`)
Retrieval-Augmented Generation system for document Q&A:
- LlamaIndex + Ollama integration
- ChromaDB vector storage
- HuggingFace embeddings (BAAI/bge-small-en-v1.5)
- PDF document processing

**Requirements:** Ollama running locally with llama3.1 model

### 3. Differentiable Neural Computer (DNC) (`src/dnc/`)
**NEW**: Advanced neural architecture with external memory for sequence processing:

#### What is a DNC?
A Differentiable Neural Computer combines neural networks with external memory, similar to how a CPU accesses RAM. It features:
- **External Memory Matrix**: Explicit storage (like computer RAM)
- **Content-Based Addressing**: Find memory by similarity
- **Temporal Linkage**: Track write order for sequential access
- **Dynamic Allocation**: Smart memory management
- **Multiple Read Heads**: Parallel memory operations

#### Key Capabilities
✓ Variable-length sequence memorization
✓ Associative recall (key-value lookup)
✓ Interpretable memory states
✓ End-to-end differentiable training

#### Demo Tasks
The [dnc_demo.ipynb](notebooks/dnc_demo.ipynb) notebook demonstrates:

1. **Copy Task**: Memorize and reproduce random sequences
   - Tests memory storage and sequential recall
   - Visualizes memory usage over time

2. **Associative Recall**: Store key-value pairs and retrieve by key
   - Tests content-based addressing
   - Shows read attention patterns

#### Run the Demo
```bash
jupyter notebook notebooks/dnc_demo.ipynb
```

The notebook includes:
- Complete DNC training on both tasks
- Memory state visualizations (usage, attention weights, temporal links)
- Performance analysis and comparisons

#### Architecture Details
See [src/dnc/README.md](src/dnc/README.md) for:
- Mathematical formulation
- Implementation details
- API reference
- Performance tips

### 4. ChromaDB Integration (`src/database/c_db.py`)
Basic vector database operations:
- Collection management
- Document embedding and storage
- Similarity search

**Run:**
```bash
python src/database/c_db.py
```

### 5. Custom Llama Model (Ollama)
Persona-based LLM with custom system prompt:

**Creating and Running:**
```bash
ollama create david -f ./Modelfile
ollama run david
```

**Modelfile Configuration:**
- **Model Source:** llama3.1
- **Temperature:** 1 (balanced creativity)
- **System Prompt:** Personal health trainer persona named David

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

This uses the custom `get_requirements()` function in `setup.py` that filters out `-e .` from requirements.txt.

## Project Structure

```
mlprojects/
├── src/
│   ├── linear_regression/    # Traditional ML
│   ├── database/             # ChromaDB examples
│   └── dnc/                  # Differentiable Neural Computer
│       ├── __init__.py
│       ├── dnc_model.py      # Main DNC implementation
│       ├── memory_operations.py  # Memory addressing & temporal linkage
│       └── README.md         # Detailed DNC documentation
├── notebooks/
│   ├── notebook.ipynb        # RAG pipeline with Llama
│   └── dnc_demo.ipynb        # DNC demonstrations
├── documents/                # PDFs for RAG
├── Modelfile                 # Custom Ollama model config
├── setup.py                  # Package setup
└── requirements.txt          # Dependencies
```

## Key Dependencies

- **ML/Data**: pandas, numpy, scikit-learn
- **Deep Learning**: torch, torchvision
- **LLM/Embeddings**: llama-index, transformers, sentence-transformers
- **Vector DB**: chromadb
- **Visualization**: matplotlib, seaborn
- **Document Processing**: pypdf, langchain

## Development Notes

- Python 3.12 target version
- Uses both `setup.py` and `pyproject.toml`
- LLM operations require Ollama running on localhost:11434
- DNC training requires GPU for best performance (CPU supported)

## References

### DNC
- Paper: [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101) (Graves et al., Nature 2016)
- DeepMind: [Differentiable Neural Computers Blog](https://deepmind.google/discover/blog/differentiable-neural-computers/)

### RAG/LLM
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama](https://ollama.ai/)
