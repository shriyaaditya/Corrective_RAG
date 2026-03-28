# CRAG 

A practical local implementation of **Corrective Retrieval-Augmented Generation (CRAG)** with:

- automatic document ingestion from a folder
- local vector search (FAISS + sentence-transformers)
- Groq-based evaluator/rewriter/generator
- optional web fallback (DuckDuckGo + Wikipedia)

This project is designed to feel simple in daily use: drop files into a folder, ask questions, get grounded answers.

---

## What This Project Does

1. Watches a documents folder (default: `docs/`)
2. Parses supported files into text chunks
3. Embeds chunks and stores them in a persistent vector DB (default: `vector_db/`)
4. Retrieves relevant chunks for a question
5. Evaluates retrieval quality and chooses one action:
   - `CORRECT`: use internal knowledge
   - `INCORRECT`: use web search
   - `AMBIGUOUS`: combine both
6. Generates final answer with Groq model

---

## Folder Roles

- `docs/`  
  Your **input folder**. Add PDFs/DOCX/XLSX/TXT/CSV/MD/etc here.

- `vector_db/`  
  Your **persistent memory** (FAISS index + metadata).  
  This is reused across runs so previous knowledge stays.

---

## Supported Document Formats

- `.pdf`
- `.docx`, `.doc` (best effort)
- `.xlsx`, `.xls`
- `.csv`
- `.txt`, `.md`, `.rst`, `.json`, `.html`, `.htm`

---

## Quick Start

## 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## 3) Add your Groq API key

Create/update `.env` in project root:

```env
GROQ_API_KEY=your_key_here
```

## 4) Run

Interactive:

```powershell
python main.py
```

Single question:

```powershell
python main.py --query "What does this dataset say about churn?"
```

Show indexed sources:

```powershell
python main.py --list-sources
```

Custom folders:

```powershell
python main.py --docs-dir my_docs --db-dir my_vector_db
```

---

## Day-to-Day Workflow

1. Keep `python main.py` running.
2. Drop new files into `docs/`.
3. Watcher ingests automatically.
4. Ask questions in terminal.

Notes:
- unchanged files are skipped quietly
- changed files (same name, different content) are re-indexed

---

## Commands in Interactive Mode

- `:sources` -> list indexed documents
- `:help` -> show help
- `exit` / `quit` / `q` -> stop app

---

## Configuration

Main knobs are in `config.py`:

- model names for evaluator/rewriter/generator
- retrieval thresholds
- top-k values
- generation temperature and max tokens

---

## Troubleshooting

- **`No module named 'faiss'`**  
  Install dependencies in the same Python environment you use to run `main.py`.

- **`No module named 'pypdf'` / doc parser libs missing**  
  Run `pip install -r requirements.txt` again in active venv.

- **No docs found**  
  Make sure files are in the folder used by `--docs-dir` (default `docs/`).

- **No good local answers yet**  
  Start by adding more domain documents; until then pipeline may rely on web fallback.

---

## Tech Stack

- Groq API (`groq`)
- DuckDuckGo search (`duckduckgo-search`)
- Wikipedia API (`wikipedia-api`)
- Local embeddings (`sentence-transformers`)
- Local ANN index (`faiss-cpu`)
- File watching (`watchdog`)
- File parsing (`pypdf`, `python-docx`, `openpyxl`)

---

## Why This Setup

- Works locally
- Keeps your own document knowledge persistent
- Uses mostly free/open tooling
- Modular enough to swap retriever/evaluator components later

