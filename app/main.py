from __future__ import annotations

import os
import subprocess
from pathlib import Path
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

model = SentenceTransformer("all-MiniLM-L6-v2")
documents: List[dict] = []


def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def index_document(text: str) -> None:
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
            ["ollama", "run", "llama3.2:latest", prompt],
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
            },
        )

    index_document(text)
    return RedirectResponse(url="/?uploaded=1", status_code=303)


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
    return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "show_upload": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
