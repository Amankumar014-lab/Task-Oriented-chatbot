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

### 1. Check Dependencies
```bash
python main.py --check-deps
```

### 2. Build Pipeline (Process data, create embeddings)
```bash
python main.py --build
```

This will:
- Load JSON repair guides from `jsons/` folder
- Extract and chunk text into 100-200 word segments
- Create embeddings using SentenceTransformers
- Build FAISS index for similarity search
- Save index files: `faiss_index.bin` and `documents.pkl`

### 3. Launch Chatbot
```bash
python main.py --launch
```

Open `http://localhost:7860` in your browser!

### All-in-One Command
```bash
python main.py --build --launch
```

## 💡 Usage Options

```bash
# Create public link (for Colab/sharing)
python main.py --launch --share

# Use simplified model (faster, CPU-friendly)
python main.py --launch --simple

# Custom port
python main.py --launch --port 8080

# Test all components
python main.py --test
```

## 📁 New Files

```
MyFixit-Dataset-master/
├── data_processor.py       # Document loading and chunking
├── embeddings.py           # Embedding creation and FAISS
├── retriever.py            # Document retrieval
├── llm_handler.py          # LLM inference (Mistral 7B)
├── app.py                  # Gradio web interface
├── main.py                 # Orchestration script
├── requirements.txt        # Python dependencies
└── PROJECT_README.md       # Full documentation (this file)
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

### Google Colab
```python
!python main.py --build --launch --share
```

### HuggingFace Spaces
1. Create new Gradio Space
2. Upload all Python files + requirements.txt
3. Upload `jsons/` folder
4. Auto-deploys!

### Local
```bash
python main.py --launch
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
```bash
# Use simplified model
python main.py --launch --simple
```

### CUDA Not Available?
System auto-detects and uses CPU. For GPU:
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### FAISS Index Missing?
```bash
# Build first
python main.py --build
```

## 📊 Technical Stack

- **LLM**: Mistral 7B Instruct (4-bit quantized)
- **Embeddings**: all-MiniLM-L6-v2 (384-dim)
- **Vector DB**: FAISS (cosine similarity)
- **Framework**: LangChain, HuggingFace Transformers
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
