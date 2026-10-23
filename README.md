# RAG System and AI Assistant Agent

This repository contains two small LangChain projects:

1. **Document Q&A** (`project1_rag`) — a retrieval-augmented generation (RAG) prototype that searches Transformer notes and asks an LLM to answer from the retrieved passages.
2. **Tool-using assistant** (`project2_agent`) — a command-line chat agent that can do arithmetic, report the current date and time, and look up sample student grades.

## Requirements

- Python 3.10 or later
- A Groq API key to run either assistant

## Setup

From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root and add your Groq API key:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` private. It is ignored by Git; do not commit API keys.

## Run the tool-using assistant

From the repository root:

```powershell
python project2_agent\main.py
```

Enter a message at the `You:` prompt. The agent can use the tools in `project2_agent\tools.py`. Type `exit` or `quit` to stop.

## Document Q&A (RAG)

The RAG prototype uses a hybrid retriever (Chroma similarity search and BM25) over text chunks, then asks Groq to answer with citations to the retrieved passages. Its sample source document is in `project1_rag\data\transformers_notes.txt`.

**Before running the RAG scripts on another machine**, update the hard-coded source-document and embedding-model cache paths in `project1_rag\ingest.py` to locations available on that machine. The ingestion script also deletes and rebuilds its Chroma collection, so run it only when you intend to re-index:

```powershell
python project1_rag\ingest.py
python project1_rag\ask.py
```

The current `ask.py` contains a sample question in the script; change it there to ask something else.

## Project layout

```text
project1_rag/
  data/                 Sample Transformer notes
  ask.py                Sample RAG question
  config.py             Shared RAG settings
  ingest.py             Document loading and Chroma indexing
  rag_chain.py          Retrieval and answer-generation chain
project2_agent/
  agent.py              Agent and tool registration
  grades_data.py        Sample student grades
  main.py               Interactive command-line chat loop
  tools.py              Arithmetic, date/time, and grade lookup tools
requirements.txt        Python dependencies
```
