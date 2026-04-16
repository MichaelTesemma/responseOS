# Response OS PoC Backend + Dashboard

Minimal FastAPI proof-of-concept for PDF-based question answering.

## Features

- Upload PDF document
- Extract text and chunk into embeddings
- Retrieve top relevant chunks for a question
- Query Ollama for grounded answers with citations

## Run

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Start the app:

```bash
uvicorn app.main:app --reload
```

3. Open `http://127.0.0.1:8000`

## Notes

- Data is stored only in memory.
- The app uses the Ollama CLI to generate answers.
- Upload a PDF first, then ask a question.
