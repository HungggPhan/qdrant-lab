"""RAG: truy xuất chunk từ Qdrant rồi để LLM cục bộ (GGUF trong ./models) trả lời dựa trên context.

Dùng chung model embedding và QdrantClient của `Ingestor` (không nạp bge-m3 lần hai).
"""
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from qdrant_client import models

MODELS_DIR = Path(__file__).parent / "models"
LLAMA_SERVER_BIN = Path(os.getenv("LLAMA_SERVER_BIN", Path(__file__).parent / "bin" / "llama.cpp" / "llama-server.exe"))
LLM_PORT = int(os.getenv("LLM_PORT", "8081"))
NO_ANSWER = "Chưa tìm thấy thông tin phù hợp trong tài liệu."

# Cosine của bge-m3: chunk liên quan đo được >= ~0.60, câu ngoài phạm vi <= ~0.46. Dưới ngưỡng này coi như không có bằng chứng.
DEFAULT_MIN_SCORE = 0.50

PROMPT = """Bạn là trợ lý hỏi đáp trên tài liệu.

NGỮ CẢNH:
{context}

{history}CÂU HỎI: {question}

Hãy trả lời CHỈ dựa trên NGỮ CẢNH ở trên, không dùng kiến thức bên ngoài, bằng tiếng Việt, ngắn gọn.
BẮT BUỘC ghi số nguồn trong ngoặc vuông ngay sau mỗi câu lấy từ ngữ cảnh, ví dụ: "Kế thừa cho phép lớp con dùng lại thuộc tính của lớp cha [1]. Lớp con có thể ghi đè phương thức [2]." Chỉ dùng các số có trong ngữ cảnh.
Nếu ngữ cảnh không chứa câu trả lời, chỉ trả lời đúng một câu: {no_answer}
"""


class Retriever:
    def __init__(self, model, client, collection: str, prefix_query: str = ""):
        self.model = model
        self.client = client
        self.collection = collection
        self.prefix_query = prefix_query  # họ e5 cần "query: "; bge-m3 không cần

    def search(self, question: str, top_k: int = 4, folder: str | None = None,
               min_score: float = DEFAULT_MIN_SCORE) -> list[dict]:
        vector = self.model.encode(self.prefix_query + question, normalize_embeddings=True).tolist()
        flt = None
        if folder:
            flt = models.Filter(must=[models.FieldCondition(key="folder", match=models.MatchValue(value=folder))])
        result = self.client.query_points(
            collection_name=self.collection, query=vector, query_filter=flt,
            limit=top_k, with_payload=True,
        )
        hits = []
        for p in result.points:
            if p.score < min_score:
                continue
            hits.append({
                "id": p.id, "score": round(p.score, 4), "text": p.payload["text"],
                "source": p.payload["source"], "folder": p.payload["folder"],
                "page": p.payload["page"], "chunk_index": p.payload["chunk_index"],
            })
        return hits


class LlamaServer:
    """Chạy `llama-server` (llama.cpp bản CUDA) làm tiến trình con; dùng lại nếu đã có server ở cổng đó."""

    def __init__(self, model_path: Path, binary: Path = LLAMA_SERVER_BIN, port: int = LLM_PORT,
                 n_ctx: int = 6144, gpu_layers: int = 99):
        self.model_path, self.binary, self.port, self.n_ctx, self.gpu_layers = model_path, binary, port, n_ctx, gpu_layers
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                return r.status == 200
        except OSError:
            return False

    def start(self, timeout: float = 120):
        if self.healthy():
            return  # đã có server (tự chạy tay hoặc từ lần trước)
        if not self.binary.is_file():
            raise FileNotFoundError(f"Không thấy {self.binary} (tải llama.cpp bản CUDA, xem CLAUDE.md)")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Không thấy model {self.model_path}")
        log_file = open(Path(__file__).parent / "llama-server.log", "wb")
        self.proc = subprocess.Popen(
            [str(self.binary), "-m", str(self.model_path), "-c", str(self.n_ctx), "-ngl", str(self.gpu_layers),
             "-np", "1",  # 1 slot: nếu không, ngữ cảnh bị chia cho nhiều slot song song
             "--host", "127.0.0.1", "--port", str(self.port)],
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server thoát với mã {self.proc.returncode} (xem llama-server.log)")
            if self.healthy():
                return
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server không sẵn sàng sau %ds" % timeout)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Answerer:
    """Gọi llama-server qua HTTP. Gemma-it không có vai system nên toàn bộ chỉ dẫn nằm trong lượt user."""

    def __init__(self, base_url: str, n_ctx: int = 6144, max_answer_tokens: int = 512):
        self.base_url = base_url
        self.n_ctx = n_ctx
        self.max_answer_tokens = max_answer_tokens

    def _post(self, path: str, body: dict, timeout: float = 30):
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(req, timeout=timeout)

    def _tokens(self, text: str) -> list[int]:
        with self._post("/tokenize", {"content": text, "add_special": False}) as r:
            return json.load(r)["tokens"]

    def _trim(self, text: str, max_tokens: int) -> str:
        ids = self._tokens(text)
        if len(ids) <= max_tokens:
            return text
        with self._post("/detokenize", {"tokens": ids[:max_tokens]}) as r:
            return json.load(r)["content"].strip() + " …"

    def build_prompt(self, question: str, hits: list[dict], history: list[tuple[str, str]] | None = None) -> str:
        # chia đều ngân sách token cho các chunk, chừa chỗ cho chỉ dẫn + câu hỏi + câu trả lời
        budget = self.n_ctx - self.max_answer_tokens - 400 - len(self._tokens(question))
        if history:
            budget -= 400  # chỗ cho lịch sử
        per_chunk = max(200, budget // max(1, len(hits)))
        blocks = [
            f"[{i}] (tệp: {h['source']}, trang {h['page']})\n{self._trim(h['text'], per_chunk)}"
            for i, h in enumerate(hits, start=1)
        ]
        past = ""
        if history:  # chỉ để hiểu câu hỏi nối tiếp (đại từ, "còn ... thì sao"), không phải bằng chứng
            lines = [f"Hỏi: {q}\nĐáp: {self._trim(a, 120)}" for q, a in history]
            past = "LỊCH SỬ HỘI THOẠI (chỉ để hiểu câu hỏi, KHÔNG dùng làm nguồn thông tin):\n" + "\n".join(lines) + "\n\n"
        return PROMPT.format(no_answer=NO_ANSWER, context="\n\n".join(blocks), question=question, history=past)

    def stream(self, question: str, hits: list[dict], history: list[tuple[str, str]] | None = None):
        """Sinh từng đoạn text của câu trả lời."""
        messages = [{"role": "user", "content": self.build_prompt(question, hits, history)}]
        body = {"messages": messages, "max_tokens": self.max_answer_tokens, "temperature": 0.1, "stream": True}
        with self._post("/v1/chat/completions", body, timeout=300) as r:  # đóng kết nối = server dừng sinh
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                choices = json.loads(line[5:])["choices"]
                delta = choices[0]["delta"].get("content") if choices else None
                if delta:
                    yield delta


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    """Các số [n] model đã trích dẫn và thực sự tồn tại trong nguồn."""
    nums = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in nums if 1 <= n <= n_sources)
