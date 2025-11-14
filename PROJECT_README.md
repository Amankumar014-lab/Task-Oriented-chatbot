# 🔧 Task-Oriented RAG Chatbot - Setup Guide

This directory has been enhanced with a **Retrieval Augmented Generation (RAG)** chatbot to help users perform technical tasks like device repair and troubleshooting.

## 🎯 New Features Added

- **Task-Oriented RAG Chatbot** with Mistral 7B Instruct
- **FAISS Vector Search** with SentenceTransformers embeddings
- **Interactive Gradio UI** with source citations
- **Free-Tier Compatible** with 4-bit quantization

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Use Pre-built Index (Fastest - 2 minutes)

**Recommended for first-time users and demos**

```bash
# 1. Extract pre-built index
# (faiss_index.bin and documents.pkl already in project root)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch chatbot
python app.py
```

Open `http://localhost:7860` in your browser!

### Option 2: Build from Scratch (20-30 minutes)

**Only needed if rebuilding index or updating dataset**

```bash
# 1. Extract dataset
unzip dataset_jsons.zip
# This creates: MyFixit-Dataset-master/jsons/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build index (one-time)
python data_processor.py    # Process JSON guides
python embeddings.py         # Create FAISS index

# 4. Launch chatbot
python app.py
```

This will:

- Load 31,601 JSON repair guides from extracted folder
- Extract and chunk text into 100-200 word segments
- Create embeddings using SentenceTransformers
- Build FAISS index for similarity search
- Save index files: `faiss_index.bin` and `documents.pkl`

## 💡 Advanced Usage

### Local Deployment

```bash
# Standard launch
python app.py

# With custom configuration
# Edit app.py to adjust:
# - use_simple_model=True (for CPU-only environments)
# - share=True (for public Gradio link)
```

### Google Colab Deployment

**Fastest way to deploy with GPU acceleration**

1. **Open notebook**: Upload `colab_quickstart.ipynb` to Google Colab
2. **Upload files**: When prompted, upload these files:
   - `app.py`
   - `data_processor.py`
   - `embeddings.py`
   - `retriever.py`
   - `llm_handler.py`
   - `requirements.txt`
   - `rag_index.zip` (pre-built index)
   - `dataset_jsons.zip` (if rebuilding index)
3. **Run cells**: Execute all cells in sequence
4. **Get public link**: Gradio generates a public URL valid for 72 hours

**Colab Environment:**

- T4 GPU (free tier)
- ~13GB RAM
- ~4GB model memory with 4-bit quantization
- 6-12 second response time

**Note:** The notebook handles all setup automatically - no manual installation needed!

## 📁 Project Files

```
genai_latest/
├── data_processor.py       # Document loading and chunking
├── embeddings.py           # Embedding creation and FAISS index
├── retriever.py            # Document retrieval engine
├── llm_handler.py          # LLM inference (Mistral 7B)
├── app.py                  # Gradio web interface (main entry point)
├── colab_quickstart.ipynb  # Google Colab deployment notebook
├── requirements.txt        # Python dependencies
├── rag_index.zip           # Pre-built FAISS index + documents
├── dataset_jsons.zip       # Compressed MyFixit dataset
├── PROJECT_README.md       # User guide (this file)
├── CLEANUP_SUMMARY.txt     # File organization history
└── cloud_deployment/       # Copy of core files for easy upload
    ├── app.py
    ├── data_processor.py
    ├── embeddings.py
    ├── retriever.py
    ├── llm_handler.py
    ├── requirements.txt
    └── colab_quickstart.ipynb
```

## 🏗️ Architecture

```
User Query → Retriever (FAISS) → LLM (Mistral 7B) → Response + Sources
```

1. **Data Processor**: Loads and chunks JSON repair guides
2. **Embeddings**: Creates vectors with SentenceTransformers
3. **Retriever**: Searches FAISS index for relevant chunks
4. **LLM**: Generates task-oriented responses with Mistral 7B
5. **Gradio UI**: Interactive chatbot interface

## 🎓 Example Queries

- "How do I replace an iPhone screen?"
- "My laptop won't turn on. What should I check?"
- "Steps to replace a phone battery"
- "Fix broken headphone jack"

## 🌐 Deployment

### Google Colab (Recommended)

Upload `colab_quickstart.ipynb` to Google Colab and run all cells. The notebook will:

- Install dependencies automatically
- Extract pre-built index or build from scratch
- Launch Gradio with public shareable link

### HuggingFace Spaces

1. Create new Gradio Space
2. Upload all Python files from `cloud_deployment/` folder
3. Upload `rag_index.zip` and extract in Space
4. Set `app.py` as entry point
5. Auto-deploys!

### Local

```bash
# Standard local deployment
python app.py

# With public URL (share link)
# Edit app.py: demo.launch(share=True)
python app.py
```

## ⚙️ Configuration

### Adjust Chunk Size

Edit `data_processor.py`:

```python
processor = DataProcessor(chunk_size=150)  # 100-200 words
```

### Adjust Retrieval Count

Edit `retriever.py`:

```python
retriever = Retriever(top_k=5)  # Top-k documents
```

### LLM Settings

Edit `llm_handler.py`:

```python
llm = LLMHandler(
    load_in_4bit=True,      # Memory-efficient
    max_new_tokens=512,     # Response length
    temperature=0.7         # Creativity
)
```

## 🔧 Troubleshooting

### Out of Memory?

Edit `app.py` to use simplified model:

```python
# Change in app.py:
demo = create_interface(use_simple_model=True)
```

Or reduce batch size in `embeddings.py`:

```python
batch_size = 16  # Reduce from 32
```

### CUDA Not Available?

System auto-detects and uses CPU. For GPU acceleration:

```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### FAISS Index Missing?

Extract the pre-built index:

```bash
# Index files should be in project root:
# - faiss_index.bin
# - documents.pkl

# If missing, extract from rag_index.zip or rebuild:
python data_processor.py
python embeddings.py
```

## 📊 Technical Stack

- **LLM**: Mistral 7B Instruct (4-bit quantized)
- **Embeddings**: all-MiniLM-L6-v2 (384-dim)
- **Vector DB**: FAISS (cosine similarity)
- **Framework**: HuggingFace Transformers, bitsandbytes
- **UI**: Gradio
- **Dataset**: MyFixit (iFixit repair guides)

## 📚 Full Documentation

See individual module docstrings and inline comments for detailed API documentation.

## 🙏 Credits

- **Original Dataset**: [MyFixit Dataset](https://github.com/microsoft/MyFixit-Dataset)
- **LLM**: Mistral AI
- **Embeddings**: SentenceTransformers
- **Vector Search**: Facebook AI (FAISS)

---

## Original MyFixit Dataset README

For information about the original MyFixit dataset, see below or refer to the [original repository](https://github.com/microsoft/MyFixit-Dataset).

---

# MyFixit Dataset

This repository contains the MyFixit dataset collected from [iFixit](https://www.ifixit.com) website.

**31,601 repair manuals** across 15 device categories with annotated steps.

For details, refer to the [LREC 2020 paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.260.pdf).
