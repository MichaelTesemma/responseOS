# PoC Backend + Dashboard (FastAPI) — Agent Instructions

## 🎯 Objective

Build a **minimal proof-of-concept backend** with a basic dashboard that demonstrates:

1. Upload a document (PDF)
2. Ask a question
3. Get an answer grounded in the document with simple citations

This is NOT an MVP. Do not add unnecessary complexity.

---

## 🧱 Tech Stack

* FastAPI
* Jinja2 (for templates)
* sentence-transformers (embeddings)
* FAISS (or in-memory similarity search)
* PyPDF (or equivalent PDF parser)
* Ollama (for LLM)

---

## 📁 Project Structure

```
app/
├── main.py
├── templates/
│   └── index.html
├── static/
│   └── style.css (optional)
```

---

## 🧠 Global In-Memory Storage

Create a global variable:

```python
documents = []
```

Each item:

```python
{
    "text": "chunk text",
    "embedding": [vector]
}
```

---

## ⚙️ Core Components

### 1. PDF Parsing

* Extract raw text from uploaded PDF
* Keep it simple (no OCR needed)

---

### 2. Chunking

* Split text into chunks of ~300–500 words
* Add small overlap between chunks

---

### 3. Embeddings

* Use sentence-transformers
* Generate embedding for each chunk
* Store in memory

---

### 4. Retrieval

When a question is asked:

1. Embed the question
2. Compute similarity with all stored chunks
3. Return top 3 most relevant chunks

---

### 5. Generation (Ollama)

Send prompt:

```
You are a security compliance assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't have enough information."

Cite sources like [1], [2].

Context:
{top_chunks}

Question:
{user_question}
```

---

## 🔌 API Endpoints

### 1. GET /

* Render `index.html`
* Pass `answer=None`

---

### 2. POST /upload

* Accept file upload
* Parse PDF
* Chunk text
* Generate embeddings
* Store in `documents`
* Redirect back to `/`

---

### 3. POST /ask

* Accept question from form
* Embed question
* Retrieve top 3 chunks
* Call Ollama
* Return rendered `index.html` with answer

---

## 🖥️ Frontend (index.html)

### Requirements

Single page with:

#### 1. Upload Form

```
<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <button type="submit">Upload</button>
</form>
```

---

#### 2. Question Form

```
<form action="/ask" method="post">
  <input type="text" name="question" placeholder="Ask a question..." required>
  <button type="submit">Ask</button>
</form>
```

---

#### 3. Answer Display

```
{% if answer %}
  <div>
    <h3>Answer:</h3>
    <p>{{ answer }}</p>
  </div>
{% endif %}
```

---

## ⚡ Behavior Rules

* Do NOT persist data (memory only)
* Do NOT add authentication
* Do NOT add background workers
* Do NOT add databases
* Do NOT add agents

---

## ✅ Success Criteria

The system works if:

1. A user uploads a PDF
2. Asks a question
3. Receives a grounded answer using document content

Example:

Question:
"Do you encrypt data at rest?"

Answer:
"Yes, data is encrypted at rest using AES-256 [1]."

---

## 🚫 Explicit Non-Goals

* No production readiness
* No scalability
* No optimization
* No advanced UI
* No multi-user support

---

## 🧠 Implementation Priority

1. Upload → parse → chunk → embed
2. Question → retrieve → generate
3. Display answer

Do not move forward until each step works.

---

## 🏁 Final Instruction

Favor **simplicity over correctness** and **working code over perfect architecture**.

This is a learning and validation system, not a product.
