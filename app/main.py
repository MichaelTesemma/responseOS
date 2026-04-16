from __future__ import annotations

import os
import subprocess
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

MAX_FILE_SIZE = 30 * 1024 * 1024
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
MAX_CHUNKS = 100
EMBED_BATCH_SIZE = 8

model = SentenceTransformer("all-MiniLM-L6-v2")
documents: List[dict] = []
uploaded_files: List[dict] = []
next_file_id = 1


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"

def clean_text(text: str) -> str:
    lines = []
    previous_line = None
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned == previous_line:
            continue
        previous_line = cleaned
        lines.append(cleaned)
    return " ".join(lines)


def extract_text_from_pdf(file: UploadFile) -> str:
    if hasattr(file.file, "seek"):
        file.file.seek(0)
    reader = PdfReader(file.file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    raw_text = "\n\n".join(pages).strip()
    return clean_text(raw_text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words) and len(chunks) < MAX_CHUNKS:
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= 50:
            chunks.append(chunk[:1000].strip())
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def index_document(text: str, file_id: int) -> None:
    chunks = chunk_text(text)
    if not chunks:
        return
    embeddings = embed_texts(chunks)
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        documents.append({
            "text": chunk,
            "embedding": emb,
            "id": len(documents) + 1,
            "chunk": idx,
            "file_id": file_id,
        })


def retrieve_top_chunks(question: str, top_k: int = 3) -> List[dict]:
    if not documents:
        return []
    question_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    embeddings = np.stack([doc["embedding"] for doc in documents])
    scores = embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]


def format_context(chunks: List[dict]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"[{index}] {chunk['text']}")
    return "\n\n".join(lines)


def call_ollama(prompt: str) -> str:
    try:
        completed = subprocess.run(
            ["ollama", "run", "smollm2:1.7b", prompt],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return completed.stdout.strip()
    except FileNotFoundError:
        return "Ollama CLI not found. Please install Ollama and ensure it is on PATH."
    except subprocess.CalledProcessError as exc:
        error_text = (exc.stderr or exc.stdout or "Unexpected Ollama error.").strip()
        return f"Ollama error: {error_text}"
    except subprocess.TimeoutExpired:
        return "Ollama request timed out."


@app.get("/")
async def index(request: Request, uploaded: bool = False):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": None,
            "show_upload": bool(documents),
            "uploaded": uploaded,
            "uploaded_files": uploaded_files,
        },
    )


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "answer": "Please upload a PDF file.",
                "show_upload": bool(documents),
                "uploaded_files": uploaded_files,
            },
        )

    size = 0
    if hasattr(file.file, "seek"):
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)

    if size > MAX_FILE_SIZE:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "answer": "PDF must be 30 MB or smaller.",
                "show_upload": bool(documents),
                "uploaded_files": uploaded_files,
            },
        )

    text = extract_text_from_pdf(file)
    if not text:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "answer": "Could not extract text from the uploaded PDF.",
                "show_upload": bool(documents),
                "uploaded_files": uploaded_files,
            },
        )

    global next_file_id
    file_id = next_file_id
    next_file_id += 1
    index_document(text, file_id)

    uploaded_files.append({
        "id": file_id,
        "filename": file.filename,
        "size": format_size(size),
    })
    return RedirectResponse(url="/?uploaded=1", status_code=303)


@app.post("/remove")
async def remove(request: Request, file_id: int = Form(...)):
    global documents, uploaded_files
    documents = [doc for doc in documents if doc.get("file_id") != file_id]
    uploaded_files = [item for item in uploaded_files if item["id"] != file_id]
    return RedirectResponse(url="/", status_code=303)


@app.post("/ask")
async def ask(request: Request, question: str = Form(...)):
    if not documents:
        answer = "Upload a PDF before asking a question."
        return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "show_upload": False})

    top_chunks = retrieve_top_chunks(question)
    context_text = format_context(top_chunks)
    prompt = (
        "You are a security compliance assistant.\n\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not in the context, say: \"I don't have enough information.\"\n"
        "Cite sources like [1], [2].\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{question}\n"
    )
    answer = call_ollama(prompt)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": answer,
            "show_upload": True,
            "uploaded_files": uploaded_files,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
