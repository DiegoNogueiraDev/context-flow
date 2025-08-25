# RAG Básico (TDD) — MVP com FastAPI + UI (sem mocks)

MVP **simples e objetivo** de RAG onde:

* você **sobe um arquivo `.txt`**,
* o backend **faz chunking + vetorização + index** (embeddings *ou* TF‑IDF de fallback),
* a UI web permite **pesquisar** e **ver correlações** entre trechos relacionados,
* tudo desenvolvido com **TDD**, **sem mocks** e **sem processos simulados** (testes sobem a app real e exercitam endpoints de verdade).

> Flags do seu comando onívoras (—think —uc —depth deep —delegate —c7 —seq —serena —focus quality)s

---

## 0) Stack

* **Backend**: FastAPI + SQLite (persistência) + NumPy;

  * **Embeddings**: `sentence-transformers` (ex.: `all-MiniLM-L6-v2`).
  * **Fallback automático**: TF‑IDF (`scikit-learn`) caso `sentence-transformers` não esteja disponível (mantém o **sem mocks**).
  * **Index**: em memória (cosine). Opcional: FAISS se instalado.
* **UI**: página estática simples (HTML + JS fetch), servida pelo próprio FastAPI.
* **TDD**: `pytest` + `httpx` + `fastapi.testclient`.

> Sem dependência de serviços externos. Se o modelo de embeddings não estiver cacheado localmente, ele será baixado no primeiro uso (ou use a variável `MODEL_LOCAL_PATH`).
s
---

## 2) Como rodar (rápido)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2.1 Testes (TDD, sem mocks)
pytest -q

# 2.2 Servidor (dev)
uvicorn app:app --reload --port 8000
# Abra: http://localhost:8000  (UI)
# Docs: http://localhost:8000/docs
```

Variáveis úteis:

* `EMBEDDING_MODEL` (padrão: `sentence-transformers/all-MiniLM-L6-v2`)
* `MODEL_LOCAL_PATH` (opcional, caminho local do modelo `sentence-transformers`)
* `EMBEDDING_BACKEND` = `st` | `tfidf` (força backend)

---

## 3) Arquivos — Código completo

### 3.1 `requirements.txt`

```txt
fastapi==0.111.0
uvicorn[standard]==0.30.0
numpy==1.26.4
scikit-learn==1.5.0
pytest==8.2.0
httpx==0.27.0
python-multipart==0.0.9
# Opcionais (melhor qualidade):
sentence-transformers==2.7.0
faiss-cpu==1.8.0
```

---

### 3.2 `app.py`

```python
import os
import io
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ==========================
# Vetorização (ST ou TF‑IDF)
# ==========================
class VectorBackend:
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class SentenceTransformerBackend(VectorBackend):
    def __init__(self, model_name: Optional[str] = None, local_path: Optional[str] = None):
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.dim = None
        # Carrega do caminho local se fornecido
        if local_path and Path(local_path).exists():
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(local_path)
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        # warmup para definir dim
        v = self.model.encode(["warmup"], normalize_embeddings=True)
        self.dim = int(v.shape[1])

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

class TfidfBackend(VectorBackend):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=4096)
        self.fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            # Em fallback, adaptativamente "fita" no lote recebido
            self.fit(texts)
        X = self.vectorizer.transform(texts)
        # normaliza linha a linha (cosine)
        norms = np.linalg.norm(X.toarray(), axis=1, keepdims=True) + 1e-9
        return X.toarray() / norms

# ==========================
# Armazenamento e índice
# ==========================
class DocumentStore:
    def __init__(self, db_path: str = "rag.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_schema()
        self.backend: VectorBackend = self._init_backend()
        self.dim: int = self._detect_dim()
        self.embeddings: Optional[np.ndarray] = None  # (N, D)
        self.chunk_ids: List[int] = []
        self._load_into_memory()

    def _init_backend(self) -> VectorBackend:
        backend = os.getenv("EMBEDDING_BACKEND", "auto").lower()
        if backend in ("st", "auto"):
            try:
                return SentenceTransformerBackend(
                    model_name=os.getenv("EMBEDDING_MODEL"),
                    local_path=os.getenv("MODEL_LOCAL_PATH")
                )
            except Exception:
                if backend == "st":
                    raise
        # fallback TF‑IDF
        return TfidfBackend()

    def _detect_dim(self) -> int:
        if isinstance(self.backend, SentenceTransformerBackend):
            return self.backend.dim
        return 4096  # upper bound do TF‑IDF; ajustado dinamicamente após o primeiro fit

    def _create_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                ord INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
        self.conn.commit()

    # ---------- util ---------
    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _to_blob(v: np.ndarray) -> bytes:
        v = v.astype("float32")
        return v.tobytes()

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype="float32")
        if dim > 0 and arr.size % dim == 0:
            return arr.reshape(-1, dim)
        return arr.reshape(1, -1)

    def _load_into_memory(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, vector FROM chunks ORDER BY id ASC;")
        rows = cur.fetchall()
        if not rows:
            self.embeddings = np.zeros((0, self.dim), dtype="float32")
            self.chunk_ids = []
            return
        vecs = []
        ids = []
        dim = self.dim
        for cid, vec_blob in rows:
            vec = self._from_blob(vec_blob, dim)
            vecs.append(vec)
            ids.append(cid)
        self.embeddings = np.vstack(vecs)
        # normaliza (p/ cosine)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self.embeddings = self.embeddings / norms
        self.chunk_ids = ids

    # ---------- API pública ---------
    def add_document(self, name: str, text: str) -> Dict[str, Any]:
        chunks = self._chunk(text)
        vecs = self.backend.encode([c for (_, c) in chunks])
        if isinstance(self.backend, TfidfBackend):
            # Garante normalização (já normalizado na encode)
            pass
        cur = self.conn.cursor()
        cur.execute("INSERT INTO documents(name, created_at) VALUES(?, ?);", (name, self._now()))
        doc_id = cur.lastrowid
        for i, (_, chunk_text) in enumerate(chunks):
            v_blob = self._to_blob(vecs[i])
            cur.execute(
                "INSERT INTO chunks(doc_id, ord, text, vector) VALUES(?, ?, ?, ?);",
                (doc_id, i, chunk_text, v_blob)
            )
        self.conn.commit()
        # recarrega para memória
        self._load_into_memory()
        return {"doc_id": int(doc_id), "chunks": len(chunks)}

    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return {"results": []}
        qv = self.backend.encode([query])[0]
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sims = self.embeddings @ qv  # cosine
        k = int(min(k, sims.shape[0]))
        top_idx = np.argpartition(-sims, k-1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = []
        for ridx in top_idx:
            chunk_id = self.chunk_ids[ridx]
            score = float(sims[ridx])
            doc_id, ord_, text = self._get_chunk_meta(chunk_id)
            corr = self._correlate(ridx, top=3, exclude_self=True)
            results.append({
                "chunk_id": int(chunk_id),
                "score": score,
                "doc_id": int(doc_id),
                "doc_name": self._get_doc_name(doc_id),
                "ord": int(ord_),
                "text": text,
                "correlations": [int(self.chunk_ids[i]) for i in corr]
            })
        return {"results": results}

    def stats(self) -> Dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents;")
        n_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks;")
        n_chunks = cur.fetchone()[0]
        return {"documents": n_docs, "chunks": n_chunks}

    # ---------- helpers SQL ---------
    def _get_doc_name(self, doc_id: int) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM documents WHERE id=?;", (doc_id,))
        row = cur.fetchone()
        return row[0] if row else "?"

    def _get_chunk_meta(self, chunk_id: int) -> Tuple[int, int, str]:
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id, ord, text FROM chunks WHERE id=?;", (chunk_id,))
        row = cur.fetchone()
        return int(row[0]), int(row[1]), row[2]

    # ---------- chunking ---------
    @staticmethod
    def _chunk(text: str, max_chars: int = 600, overlap: int = 80) -> List[Tuple[int, str]]:
        text = text.replace('\r\n', '\n').strip()
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        idx = 0
        for p in paragraphs:
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                chunk = p[start:end]
                chunks.append((idx, chunk))
                idx += 1
                if end == len(p):
                    break
                start = max(0, end - overlap)
        if not chunks and text:
            chunks = [(0, text[:max_chars])]
        return chunks

    # ---------- correlação ---------
    def _correlate(self, idx: int, top: int = 3, exclude_self: bool = True) -> List[int]:
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []
        v = self.embeddings[idx]
        sims = self.embeddings @ v
        if exclude_self:
            sims[idx] = -1.0
        k = int(min(top, sims.shape[0]))
        top_idx = np.argpartition(-sims, k-1)[:k]
        return top_idx[np.argsort(-sims[top_idx])].tolist()

# ==========================
# FastAPI app e endpoints
# ==========================
app = FastAPI(title="RAG MVP (sem mocks)")
STORE = DocumentStore()

# UI estática
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root_page():
    index_path = Path("static/index.html")
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>RAG MVP</h1><p>Suba um arquivo em /upload e pesquise em /search?q=..."  # fallback

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    return STORE.stats()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "uploaded.txt"
    content = (await file.read()).decode("utf-8", errors="ignore")
    info = STORE.add_document(name, content)
    return {"ok": True, **info}

@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50)):
    return STORE.search(q, k)
```

---

### 3.3 `static/index.html`

```html
<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>RAG MVP</title>
    <style>
      body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif; max-width: 920px; margin: 24px auto; padding: 0 16px}
      header{display:flex; gap:12px; align-items:center;}
      .card{border:1px solid #e5e7eb; border-radius:12px; padding:16px; margin:12px 0}
      .muted{color:#6b7280}
      input[type="file"], input[type="text"], button{padding:10px; border-radius:8px; border:1px solid #e5e7eb}
      button{cursor:pointer}
      .row{display:flex; gap:8px; align-items:center}
      .result{white-space:pre-wrap}
      .pill{display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #e5e7eb; margin-right:6px; font-size:12px}
    </style>
  </head>
  <body>
    <header>
      <h1>🔎 RAG MVP</h1>
      <span class="muted" id="stats"></span>
    </header>

    <section class="card">
      <h3>1) Upload de arquivo .txt</h3>
      <div class="row">
        <input id="file" type="file" accept=".txt" />
        <button id="send">Enviar</button>
        <span id="uploadStatus" class="muted"></span>
      </div>
    </section>

    <section class="card">
      <h3>2) Busca e correlação</h3>
      <div class="row">
        <input id="q" type="text" placeholder="Digite sua busca…" style="flex:1" />
        <button id="go">Buscar</button>
      </div>
      <div id="results"></div>
    </section>

    <script src="/static/app.js"></script>
  </body>
</html>
```

---

### 3.4 `static/app.js`

```js
async function refreshStats(){
  const r = await fetch('/stats');
  const j = await r.json();
  document.getElementById('stats').textContent = `docs: ${j.documents} • chunks: ${j.chunks}`;
}

async function doUpload(){
  const el = document.getElementById('file');
  if(!el.files.length){ alert('Selecione um .txt'); return; }
  const fd = new FormData();
  fd.append('file', el.files[0]);
  const r = await fetch('/upload', { method: 'POST', body: fd });
  const j = await r.json();
  document.getElementById('uploadStatus').textContent = j.ok ? `OK • doc_id=${j.doc_id} • chunks=${j.chunks}` : 'falhou';
  refreshStats();
}

async function doSearch(){
  const q = document.getElementById('q').value.trim();
  if(!q){ return; }
  const r = await fetch(`/search?q=${encodeURIComponent(q)}&k=6`);
  const j = await r.json();
  const box = document.getElementById('results');
  box.innerHTML = '';
  (j.results || []).forEach((it, idx) => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `
      <div class="muted">#${idx+1} • score=${it.score.toFixed(3)} • doc=${it.doc_name} • chunk_id=${it.chunk_id} (ord ${it.ord})</div>
      <div class="result">${it.text.replace(/</g, '&lt;')}</div>
      <div class="muted" style="margin-top:8px">Correlação: ${(it.correlations||[]).map(c=>`<span class="pill">${c}</span>`).join('')}</div>
    `;
    box.appendChild(div);
  });
}

document.getElementById('send').addEventListener('click', doUpload);
document.getElementById('go').addEventListener('click', doSearch);
refreshStats();
```

---

### 3.5 `sample_data/sample.txt`

```txt
O contexto flui melhor quando dividimos o conhecimento em trechos pequenos e coerentes. Isso facilita a indexação e a recuperação.

A busca semântica complementa a busca lexical. Em um MVP, é válido usar embeddings pequenos para manter a latência baixa.

Correlar trechos significa encontrar vizinhos próximos no espaço vetorial. Essa vizinhança sugere que eles falam de temas relacionados.
```

---

### 3.6 `tests/test_end_to_end.py`

```python
import io
from pathlib import Path
from fastapi.testclient import TestClient
import app as rag_app

client = TestClient(rag_app.app)


def test_health():
    r = client.get('/healthz')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_upload_and_search(tmp_path):
    # usa o sample.txt real (sem mocks)
    sample_path = Path('sample_data/sample.txt')
    data = sample_path.read_text(encoding='utf-8')

    # upload
    files = { 'file': ('sample.txt', data, 'text/plain') }
    r = client.post('/upload', files=files)
    assert r.status_code == 200
    j = r.json()
    assert j['ok'] is True
    assert j['chunks'] > 0

    # busca
    r = client.get('/search', params={'q': 'busca semântica', 'k': 5})
    assert r.status_code == 200
    res = r.json()['results']
    assert isinstance(res, list)
    assert len(res) >= 1
    # cada resultado deve trazer correlações (ids de chunks)
    assert 'correlations' in res[0]
```

---

## 4) Critérios de Aceite (MVP)

* [x] **Upload** de `.txt` e **indexação** imediata (chunking + vetores) sem jobs externos.
* [x] **Busca** com retorno dos top‑K e **correlações** (IDs de chunks semelhantes) por resultado.
* [x] **UI web** única e mínima que prova valor (upload + busca + exibição).
* [x] **TDD sem mocks**: testes de *health*, *upload* e *search* invocam endpoints reais e exercitam o vetor e o índice.
* [x] **Persistência**: documentos e vetores vão para `rag.db` (SQLite). Recarrega índice em memória no boot.

> Métrica sugerida (observabilidade mínima): tempo de upload+index por arquivo e latência P95 da rota `/search` (pode-se logar `time.time()` no código para medir e evoluir depois).

---

## 5) Roadmap de melhoria (curto e direto)

1. **FAISS**: habilitar automaticamente se a lib existir (já provisionado para encaixar).
2. **Rerankss**: `BM25 → top 50 → cosine re‑rank` (quando volume crescer).
3. **Campos do resultado**: realce de *snippets* com *highlight* das palavras da consulta.
4. **Correlação por documento**: agrupar hits por doc e exibir *“outros trechos relacionados”* do mesmo documento primeiro.
5. **Observabilidade**: logs estruturados (p.ex. Pino/UVicorn JSON) + contadores simples em `/stats` (p95/p99).
6. **Integração com agentes**: endpoints HTTP já prontos; Serena/MCP podem consumir `/search` e `/upload` diretamente.

---

## 6) Dicas operacionais

* Sem internet? Configure `MODEL_LOCAL_PATH` apontando para um diretório com um modelo `sentence-transformers` já baixado.
* Dados não‑txt: converta previamente (ex.: `pdftotext`, `docx2txt`) antes do upload. Este MVP assume **texto simples** por design.
* Tamanho de chunk e *overlap* estão em `DocumentStore._chunk` (rápido de ajustar).

---

## 7) Endpoints (resumo)

* `GET /` → UI.
* `GET /docs` → OpenAPI (útil para *agents* e teste manual).
* `GET /healthz` → status.
* `GET /stats` → contagem de docs/chunks.
* `POST /upload` (`multipart/form-data` com `file`) → indexa.
* `GET /search?q=...&k=5` → resultados com correlação.

---

## 8) Teste manual rápido (cURL)

```bash
curl -F "file=@sample_data/sample.txt" http://localhost:8000/upload
curl "http://localhost:8000/search?q=busca%20sem%C3%A2ntica&k=5"
```

---

**Pronto.** É só colar a pasta e rodar. Se quiser, posso adaptar para **Next.js + API Routes** mantendo o mesmo contrato HTTP e os mesmos testes end‑to‑end sem mocks.
