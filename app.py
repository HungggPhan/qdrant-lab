"""Dịch vụ upload PDF vào Qdrant.

Chạy (đúng 1 process để model chỉ nạp một lần):
    uvicorn app:app --host 0.0.0.0 --port 8000

POST /upload  (form: folder, files...) -> 202, {batch_id, jobs, rejected}  (field cũ `file` vẫn nhận)
GET  /batches/{id}                  -> trạng thái cả lô: counts + từng job
GET  /jobs/{id}                     -> processing | done | failed
GET  /                              -> trang upload thử (nhiều file, kéo thả)
GET  /chat                          -> giao diện chat
POST /chat  (json: question, top_k) -> NDJSON stream: sources -> token... -> done
"""
import json
import logging
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ingest import DEFAULT_COLLECTION, DOCUMENTS_DIR, MODEL_NAME, Ingestor
from rag import DEFAULT_MIN_SCORE, MODELS_DIR, NO_ANSWER, Answerer, LlamaServer, Retriever, cited_numbers

log = logging.getLogger("uvicorn.error")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024  # mỗi file
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "20"))  # mỗi request
MAX_BATCH_BYTES = int(os.getenv("MAX_BATCH_MB", "500")) * 1024 * 1024  # tổng mỗi request
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
EMBED_MODEL = os.getenv("EMBED_MODEL", MODEL_NAME)
JOBS_DB = Path(__file__).parent / "jobs.db"
CHAT_MODEL = Path(os.getenv("CHAT_MODEL", MODELS_DIR / "gemma-2b-it.Q4_K_M.gguf"))
LLM_CTX = int(os.getenv("LLM_CTX", "6144"))
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "99"))  # 99 = toàn bộ model lên GPU; 0 = chạy CPU
CHAT_PAGE = Path(__file__).parent / "chat.html"
MIN_SCORE = float(os.getenv("MIN_SCORE", DEFAULT_MIN_SCORE))
CHAT_DB = Path(__file__).parent / "chat.db"
HISTORY_TURNS = 2  # số lượt hỏi-đáp gần nhất đưa cho LLM để hiểu câu hỏi nối tiếp
SHORT_FOLLOWUP_WORDS = 6  # câu hỏi ngắn hơn ngưỡng này thì ghép thêm câu hỏi trước vào truy vấn tìm kiếm

FOLDER_RE = re.compile(r"[\w\-]{1,100}")  # chữ/số (kể cả tiếng Việt), gạch dưới, gạch ngang; không có / \ .

jobs_queue: "queue.Queue[str]" = queue.Queue()
state: dict = {}


# ---------------------------------------------------------------------------
# Trạng thái job (SQLite, sống sót qua restart)
# ---------------------------------------------------------------------------
def db():
    return closing(sqlite3.connect(JOBS_DB, timeout=30))


def chat_db():
    return closing(sqlite3.connect(CHAT_DB, timeout=30))


def init_chat_db():
    with chat_db() as conn, conn:
        conn.executescript(
            """CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT, created_at REAL, updated_at REAL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT NOT NULL,
                question TEXT, answer TEXT, sources TEXT, cited TEXT, created_at REAL);
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);"""
        )


def init_db():
    with db() as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, folder TEXT, filename TEXT, status TEXT,
                chunks INTEGER, error TEXT, created_at REAL, updated_at REAL)"""
        )
        # cột tiến độ (thêm sau nên migrate cho jobs.db cũ)
        have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        for col, typ in (("stage", "TEXT"), ("progress_done", "INTEGER"), ("progress_total", "INTEGER"), ("batch_id", "TEXT")):
            if col not in have:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")


def create_job(folder: str, filename: str, batch_id: str | None = None) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    with db() as conn, conn:
        conn.execute(
            "INSERT INTO jobs (id, folder, filename, status, created_at, updated_at, batch_id) VALUES (?, ?, ?, 'processing', ?, ?, ?)",
            (job_id, folder, filename, now, now, batch_id),
        )
    return job_id


def update_job(job_id: str, status: str, chunks: int | None = None, error: str | None = None):
    with db() as conn, conn:
        conn.execute(
            "UPDATE jobs SET status=?, chunks=?, error=?, updated_at=? WHERE id=?",
            (status, chunks, error, time.time(), job_id),
        )


def set_progress(job_id: str, stage: str, done: int, total: int):
    with db() as conn, conn:
        conn.execute(
            "UPDATE jobs SET stage=?, progress_done=?, progress_total=?, updated_at=? WHERE id=?",
            (stage, done, total, time.time(), job_id),
        )


def get_job(job_id: str) -> dict | None:
    with db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Worker nền: xử lý lần lượt từng job (embedding ăn CPU nên chạy tuần tự)
# ---------------------------------------------------------------------------
def notify(job: dict):
    """Gửi thông báo khi job xong/lỗi. TODO: email / webhook / websocket."""
    log.info("job %s (%s/%s): %s", job["id"], job["folder"], job["filename"], job["status"])


def process_job(job_id: str):
    """Xử lý một job `processing`: nạp file vào Qdrant rồi ghi `done`/`failed`.

    Không ném exception ra ngoài (lỗi của một job không được làm chết worker).
    """
    job = get_job(job_id)
    if job is None or job["status"] != "processing":
        return  # job đã bị xoá hoặc đã xử lý xong (ví dụ id bị đưa vào queue hai lần)
    try:
        pdf = DOCUMENTS_DIR / job["folder"] / job["filename"]
        if not pdf.is_file():
            raise FileNotFoundError(f"Không thấy file {job['folder']}/{job['filename']}")
        chunks = state["ingestor"].ingest_file(
            pdf, job["folder"], progress=lambda stage, done, total: set_progress(job_id, stage, done, total)
        )
        update_job(job_id, "done", chunks=chunks)
    except Exception as e:  # noqa: BLE001
        log.exception("job %s lỗi", job_id)
        update_job(job_id, "failed", error=f"{type(e).__name__}: {e}")


def enqueue_pending() -> int:
    """Đưa các job còn `processing` trong DB vào queue (job dở dang khi server tắt đột ngột)."""
    with db() as conn:
        pending = [r[0] for r in conn.execute("SELECT id FROM jobs WHERE status='processing' ORDER BY created_at")]
    for job_id in pending:
        jobs_queue.put(job_id)
    return len(pending)


def worker():
    """Vòng lặp lấy job từ queue và xử lý tuần tự. `None` trong queue = tín hiệu dừng."""
    while True:
        job_id = jobs_queue.get()
        try:
            if job_id is None:
                return
            process_job(job_id)
            job = get_job(job_id)
            if job:
                notify(job)
        except Exception:  # noqa: BLE001 - kể cả lỗi DB/notify cũng không được giết worker
            log.exception("worker gặp lỗi với job %s", job_id)
        finally:
            jobs_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_chat_db()
    state["ingestor"] = Ingestor(EMBED_MODEL, QDRANT_URL, COLLECTION)  # nạp model đúng một lần
    ing: Ingestor = state["ingestor"]
    state["retriever"] = Retriever(ing.model, ing.client, COLLECTION, "query: " if "e5" in EMBED_MODEL.lower() else "")
    llm_server = LlamaServer(CHAT_MODEL, n_ctx=LLM_CTX, gpu_layers=LLM_GPU_LAYERS)
    try:
        llm_server.start()  # llama-server (CUDA) làm tiến trình con, nạp model lên GPU
        state["answerer"] = Answerer(llm_server.base_url, n_ctx=LLM_CTX)
    except Exception:  # noqa: BLE001 - lỗi LLM chỉ tắt chat, upload vẫn chạy
        log.exception("không khởi động được LLM %s, /chat sẽ trả 503", CHAT_MODEL)
    thread = threading.Thread(target=worker, name="ingest-worker")
    thread.start()
    n = enqueue_pending()
    if n:
        log.info("đưa %d job dở dang vào queue", n)
    yield
    jobs_queue.put(None)  # job đang chạy được làm xong, sau đó worker thoát
    thread.join()
    llm_server.stop()


app = FastAPI(title="Qdrant PDF ingest", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def safe_filename(name: str | None) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name or "").name).strip(" .")
    if not name.lower().endswith(".pdf") or len(name) <= 4:
        raise HTTPException(400, "Chỉ nhận file .pdf")
    return name


async def stage_upload(file: UploadFile, target_dir: Path) -> tuple[Path, int]:
    """Ghi một file upload vào file `.part` trong target_dir, trả về (đường dẫn .part, số byte).

    Chưa đụng tới file thật: việc đổi tên làm sau khi cả lô đã qua kiểm tra. Ném HTTPException nếu file
    không nhận được; mọi lỗi đều xoá file `.part` đang ghi dở.
    """
    tmp = target_dir / f".{uuid.uuid4().hex}.part"
    size = 0
    try:
        head = b""
        while len(head) < len(b"%PDF-"):  # đọc đủ phần đầu rồi mới kiểm tra, không phụ thuộc kích thước lần đọc
            part = await file.read(len(b"%PDF-") - len(head))
            if not part:
                break
            head += part
        if not head:
            raise HTTPException(400, "File rỗng")
        if not head.startswith(b"%PDF-"):
            raise HTTPException(400, "File không phải PDF hợp lệ")
        with open(tmp, "wb") as out:
            chunk = head
            while chunk:
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File vượt quá {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
                out.write(chunk)
                chunk = await file.read(1024 * 1024)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp, size


@app.post("/upload", status_code=202)
async def upload(
    folder: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),  # tên field cũ (một file), vẫn nhận để không phá client cũ
):
    folder = folder.strip()
    if not FOLDER_RE.fullmatch(folder):
        raise HTTPException(400, "Tên folder chỉ gồm chữ, số, gạch ngang và gạch dưới (tối đa 100 ký tự)")
    uploads = ([file] if file is not None else []) + files
    if len(uploads) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(413, f"Tối đa {MAX_FILES_PER_UPLOAD} file mỗi lần upload")

    target_dir = DOCUMENTS_DIR / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex
    staged: list[tuple[str, Path]] = []  # (tên file, file .part) chưa đổi tên
    accepted, rejected, seen, total = [], [], set(), 0
    try:
        for up in uploads:
            try:
                filename = safe_filename(up.filename)
                if filename.casefold() in seen:  # Windows không phân biệt hoa/thường
                    raise HTTPException(400, "File trùng tên với file trước đó trong lô")
                tmp, size = await stage_upload(up, target_dir)
            except HTTPException as e:  # lỗi của một file không làm hỏng cả lô
                rejected.append({"filename": up.filename, "reason": e.detail})
                continue
            except OSError as e:
                rejected.append({"filename": up.filename, "reason": f"Không ghi được file: {e}"})
                continue
            seen.add(filename.casefold())
            staged.append((filename, tmp))
            total += size
            if total > MAX_BATCH_BYTES:
                raise HTTPException(413, f"Tổng dung lượng vượt quá {MAX_BATCH_BYTES // 1024 // 1024} MB")

        # cả lô đã qua kiểm tra: lúc này mới ghi đè file thật
        for filename, tmp in staged:
            try:
                os.replace(tmp, target_dir / filename)
            except PermissionError:
                rejected.append({"filename": filename, "reason": "File cùng tên đang được xử lý, thử lại sau"})
            except OSError as e:
                rejected.append({"filename": filename, "reason": f"Không ghi được file: {e}"})
            else:
                accepted.append(filename)
    finally:
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)  # lô bị từ chối hoặc lỗi bất ngờ: chỉ xoá file tạm, file cũ còn nguyên
    if not accepted:
        raise HTTPException(400, {"message": "Không có file nào hợp lệ", "rejected": rejected})

    jobs = []
    for filename in accepted:
        job_id = create_job(folder, filename, batch_id)
        jobs_queue.put(job_id)
        jobs.append({"job_id": job_id, "filename": filename, "status": "processing"})
    return {"batch_id": batch_id, "folder": folder, "jobs": jobs, "rejected": rejected}


def public_job(job: dict) -> dict:
    """Dạng job trả cho client: gộp ba cột tiến độ thành `progress`."""
    job = dict(job)
    stage, done, total = job.pop("stage"), job.pop("progress_done"), job.pop("progress_total")
    job["progress"] = None if stage is None else {
        "stage": stage,  # reading | chunking | embedding | saving
        "done": done,
        "total": total,
        "percent": round(100 * done / total) if total else 0,
    }
    return job


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy job")
    return public_job(job)


@app.get("/batches/{batch_id}")
def batch_status(batch_id: str):
    with db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM jobs WHERE batch_id=? ORDER BY rowid", (batch_id,)).fetchall()
    if not rows:
        raise HTTPException(404, "Không tìm thấy lô upload")
    jobs = [public_job(dict(r)) for r in rows]
    counts = {s: sum(j["status"] == s for j in jobs) for s in ("processing", "done", "failed")}
    return {"batch_id": batch_id, "folder": jobs[0]["folder"], "counts": counts, "jobs": jobs}


# ---------------------------------------------------------------------------
# Chat (RAG)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(4, ge=1, le=5)
    folder: str | None = None
    conversation_id: str | None = None  # bỏ trống = mở cuộc trò chuyện mới


def load_history(conv_id: str, limit: int) -> list[tuple[str, str]]:
    with chat_db() as conn:
        rows = conn.execute(
            "SELECT question, answer FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT ?", (conv_id, limit)
        ).fetchall()
    return rows[::-1]


def save_turn(conv_id: str, question: str, answer: str, sources: list, cited: list):
    now = time.time()
    with chat_db() as conn, conn:
        conn.execute(
            "INSERT INTO messages (conv_id, question, answer, sources, cited, created_at) VALUES (?,?,?,?,?,?)",
            (conv_id, question, answer, json.dumps(sources, ensure_ascii=False), json.dumps(cited), now),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))


def ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.post("/chat")
def chat(req: ChatRequest):
    answerer: Answerer | None = state.get("answerer")
    if answerer is None:
        raise HTTPException(503, "Chưa nạp được mô hình ngôn ngữ (xem log server)")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Câu hỏi rỗng")

    conv_id = req.conversation_id
    history: list[tuple[str, str]] = []
    if conv_id:
        with chat_db() as conn:
            if conn.execute("SELECT 1 FROM conversations WHERE id=?", (conv_id,)).fetchone() is None:
                raise HTTPException(404, "Không tìm thấy cuộc trò chuyện")
        history = load_history(conv_id, HISTORY_TURNS)
    else:
        conv_id, now = uuid.uuid4().hex, time.time()
        with chat_db() as conn, conn:
            conn.execute("INSERT INTO conversations VALUES (?,?,?,?)", (conv_id, question[:60], now, now))

    # câu hỏi nối tiếp thường ngắn ("còn Java thì sao?") -> ghép câu hỏi trước để tìm đúng chủ đề
    search_q = question
    if history and len(question.split()) < SHORT_FOLLOWUP_WORDS:
        search_q = f"{history[-1][0]} {question}"
    hits = state["retriever"].search(search_q, top_k=req.top_k, folder=req.folder, min_score=MIN_SCORE)

    def events():
        yield ndjson({"type": "conversation", "id": conv_id})
        # nguồn lấy từ metadata thật của chunk đã đưa cho model, đánh số [1], [2]... khớp với prompt
        yield ndjson({"type": "sources", "sources": [{"n": i, **h} for i, h in enumerate(hits, start=1)]})
        if not hits:  # không đủ bằng chứng -> không gọi LLM
            yield ndjson({"type": "token", "text": NO_ANSWER})
            save_turn(conv_id, question, NO_ANSWER, [], [])
            yield ndjson({"type": "done", "cited": []})
            return
        answer = ""
        try:
            for piece in answerer.stream(question, hits, history):
                answer += piece
                yield ndjson({"type": "token", "text": piece})
        except Exception as e:  # noqa: BLE001
            log.exception("lỗi sinh câu trả lời")
            yield ndjson({"type": "error", "message": f"{type(e).__name__}: {e}"})
            return
        cited = cited_numbers(answer, len(hits))
        save_turn(conv_id, question, answer, [{"n": i, **h} for i, h in enumerate(hits, start=1)], cited)
        yield ndjson({"type": "done", "cited": cited})

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/conversations")
def list_conversations():
    with chat_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 100").fetchall()
    return [dict(r) for r in rows]


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    with chat_db() as conn:
        conn.row_factory = sqlite3.Row
        conv = conn.execute("SELECT id, title FROM conversations WHERE id=?", (conv_id,)).fetchone()
        if conv is None:
            raise HTTPException(404, "Không tìm thấy cuộc trò chuyện")
        rows = conn.execute(
            "SELECT question, answer, sources, cited FROM messages WHERE conv_id=? ORDER BY id", (conv_id,)
        ).fetchall()
    messages = [
        {"question": r["question"], "answer": r["answer"], "sources": json.loads(r["sources"]), "cited": json.loads(r["cited"])}
        for r in rows
    ]
    return {**dict(conv), "messages": messages}


@app.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: str):
    with chat_db() as conn, conn:
        conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return FileResponse(CHAT_PAGE, media_type="text/html; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><meta charset="utf-8"><title>Upload PDF</title>
<style>body{font:16px system-ui;max-width:640px;margin:2rem auto;padding:0 1rem}
input,button{font:inherit}#drop{display:block;margin:.5rem 0;padding:1.2rem;border:2px dashed #999;border-radius:8px;text-align:center;cursor:pointer}
#drop.over{background:#eef6ff;border-color:#06c}#files{display:none}
ul{list-style:none;padding:0}li{padding:.3rem 0;border-bottom:1px solid #eee;display:flex;gap:.6rem;justify-content:space-between}
.done{color:#080}.failed{color:#b00}#sum{font-weight:600;margin:.5rem 0}</style>
<h1>Upload PDF</h1>
<form id="f"><input name="folder" placeholder="Tên folder" required>
<label id="drop" for="files">Kéo thả PDF vào đây hoặc bấm để chọn (nhiều file)<br><small id="picked"></small></label>
<input id="files" type="file" accept="application/pdf" multiple>
<button>Upload</button></form>
<div id="sum"></div><ul id="list"></ul>
<script>
const $ = id => document.getElementById(id);
const STAGES = {reading: 'Đọc PDF', chunking: 'Chia chunk', embedding: 'Tạo embedding', saving: 'Ghi vào Qdrant'};
const KEY = 'lastBatch';
let timer = null;

function esc(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
function describe(j) {
  if (j.status === 'done') return ['done', 'Xong: ' + j.chunks + ' chunk'];
  if (j.status === 'failed') return ['failed', 'Thất bại: ' + j.error];
  const p = j.progress;
  return ['', p ? STAGES[p.stage] + ' ' + p.done + '/' + p.total + ' (' + p.percent + '%)' : 'Đang chờ tới lượt xử lý...'];
}
function render(jobs, rejected, counts) {
  $('list').innerHTML = jobs.map(j => { const [c, t] = describe(j);
    return '<li><span>' + esc(j.filename) + '</span><span class="' + c + '">' + esc(t) + '</span></li>'; }).join('')
    + rejected.map(x => '<li><span>' + esc(x.filename || '?') + '</span><span class="failed">Bị loại: ' + esc(x.reason) + '</span></li>').join('');
  if (counts) $('sum').textContent = counts.done + '/' + jobs.length + ' xong, ' + counts.failed + ' lỗi, ' + counts.processing + ' đang xử lý'
    + (rejected.length ? ', ' + rejected.length + ' bị loại' : '');
}
function watch(batchId, rejected) {
  clearInterval(timer);
  const tick = async () => {
    const r = await fetch('/batches/' + batchId);
    if (!r.ok) { clearInterval(timer); try { localStorage.removeItem(KEY); } catch (e) {} return; }
    const b = await r.json();
    render(b.jobs, rejected, b.counts);
    if (b.counts.processing === 0) { clearInterval(timer); try { localStorage.removeItem(KEY); } catch (e) {} }
  };
  tick(); timer = setInterval(tick, 1000);
}
function setFiles(list) {
  const dt = new DataTransfer();
  for (const f of list) if (f.name.toLowerCase().endsWith('.pdf')) dt.items.add(f);
  $('files').files = dt.files;
  $('picked').textContent = dt.files.length ? 'Đã chọn ' + dt.files.length + ' file' : '';
}
$('files').onchange = e => setFiles(e.target.files);
const drop = $('drop');
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); setFiles(e.dataTransfer.files); };

$('f').onsubmit = async e => {
  e.preventDefault();
  const files = $('files').files;
  if (!files.length) { $('sum').textContent = 'Chưa chọn file PDF nào'; return; }
  const fd = new FormData();
  fd.append('folder', e.target.folder.value);
  for (const f of files) fd.append('files', f);
  $('sum').textContent = 'Đang upload ' + files.length + ' file...'; $('list').innerHTML = '';
  const r = await fetch('/upload', {method: 'POST', body: fd});
  const j = await r.json();
  if (!r.ok) {
    const d = j.detail;
    $('sum').textContent = 'Lỗi: ' + (typeof d === 'string' ? d : (d && d.message) || r.status);
    if (d && d.rejected) render([], d.rejected, null);
    return;
  }
  try { localStorage.setItem(KEY, JSON.stringify({id: j.batch_id, rejected: j.rejected})); } catch (e) {}
  watch(j.batch_id, j.rejected);
};

try { const last = JSON.parse(localStorage.getItem(KEY)); if (last) watch(last.id, last.rejected || []); } catch (e) {}
</script>"""
