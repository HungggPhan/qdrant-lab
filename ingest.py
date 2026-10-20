"""Logic nạp PDF vào Qdrant: sửa mã tiếng Việt, chia chunk, embedding, upsert.

Dùng chung cho CLI (add_document.py) và dịch vụ FastAPI (app.py).
"""
import logging
import os
import re
import unicodedata
import uuid
from pathlib import Path

import torch
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from vi_text import merge_split_syllables

logging.getLogger("pypdf").setLevel(logging.ERROR)  # tắt cảnh báo font của pypdf

DOCUMENTS_DIR = Path(__file__).parent / "documents"
MODEL_NAME = "BAAI/bge-m3"  # 8192 token, 1024 chiều, không cần tiền tố (model họ e5 thì cần "passage: " / "query: ")
DEFAULT_COLLECTION = "vi_documents"

# ---------------------------------------------------------------------------
# Sửa lỗi mã hóa tiếng Việt của PDF
# ---------------------------------------------------------------------------
# Các PDF này dùng font Việt cũ (kiểu VN-Time): pypdf trích ra ký tự Latin-1,
# trong đó dấu thanh là ký tự đứng riêng SAU nguyên âm (vd "ca\xefc" -> "các").
# Chữ hoa bằng chữ thường - 0x20.
_BASE_LOWER = {
    "\xe0": "ă", "\xe1": "â", "\xe2": "đ", "\xe3": "ê",
    "\xe4": "ô", "\xe5": "ơ", "\xe6": "ư",
    # nguyên âm đã gộp sẵn dấu: ì ỉ ĩ í ị
    "\xe7": "ì", "\xe8": "ỉ", "\xe9": "ĩ", "\xea": "í", "\xeb": "ị",
}
_TONE_LOWER = {
    "\xef": "́", "\xec": "̀", "\xed": "̉", "\xee": "̃",  # a e i o u y
    "\xfa": "́", "\xf6": "̀", "\xf8": "̉", "\xf9": "̃",  # â ê ô
    "\xf5": "́", "\xf2": "̀", "\xf3": "̉", "\xf4": "̃",  # ă
    "\xfb": "̣", "\xfc": "̣",  # nặng
}
_VI_MAP = {}
for _src, _dst in {**_BASE_LOWER, **_TONE_LOWER}.items():
    _VI_MAP[ord(_src)] = _dst
    _VI_MAP[ord(_src) - 0x20] = _dst.upper()  # đ -> Đ, ... ; dấu thanh không đổi
_VI_TABLE = _VI_MAP


def fix_vietnamese(text: str) -> str:
    text = text.translate(_VI_TABLE)
    text = re.sub(r"[ \t]+(?=[̀-̣̉])", "", text)  # pypdf hay chèn khoảng trắng trước dấu thanh
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Đọc PDF và chia chunk
# ---------------------------------------------------------------------------
def is_legacy_vietnamese(text: str) -> bool:
    """True nếu text trích từ font Việt cũ (cần `fix_vietnamese`), False nếu đã là Unicode đúng.

    PDF Unicode tiếng Việt có nhiều chữ như ạ ả ế ộ ữ ơ ư đ (U+1EA0..U+1EF9, đ, ơ, ư); font cũ chỉ cho ra
    ký tự Latin-1 nên gần như không có các chữ này. Áp `fix_vietnamese` lên text đã đúng sẽ làm hỏng dấu
    (vd "c\xe1c" -> "câc", "v\xe0" -> "vă").
    """
    modern = sum(1 for c in text if "Ạ" <= c <= "ỹ" or c in "đĐơƠưƯ")
    legacy = sum(1 for c in text if ord(c) in _VI_TABLE)  # ký tự Latin-1 mà font cũ dùng làm chữ/dấu Việt
    size = max(1, len(text))
    # tài liệu tiếng Anh (không có chữ Việt nào, cũng gần như không có ký tự font cũ) thì giữ nguyên
    return modern < 0.01 * size and legacy >= 0.01 * size


def read_pdf_pages(path: Path, on_page=None, repair_spaces=True):
    """Trả về [(số trang, text đã sửa dấu)] của một file PDF. `on_page(số trang, tổng)` báo tiến độ đọc.

    Cách mã hoá được xác định cho cả tài liệu (không theo từng trang, vì trang ít dấu dễ bị nhận nhầm).
    `repair_spaces` nối lại âm tiết bị pypdf tách bằng khoảng trắng ("c ủa" -> "của"), xem vi_text.py.
    """
    pdf_pages = PdfReader(path).pages
    raw = []
    for number, page in enumerate(pdf_pages, start=1):
        if on_page:
            on_page(number, len(pdf_pages))
        raw.append((number, page.extract_text() or ""))
    legacy = is_legacy_vietnamese("".join(t for _, t in raw))
    pages = []
    for number, text in raw:
        text = fix_vietnamese(text) if legacy else unicodedata.normalize("NFC", text)
        text = re.sub(r"[ \t]+", " ", text)
        if repair_spaces:
            text = merge_split_syllables(text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        if text:
            pages.append((number, text))
    return pages


# Các mức tách, từ thô đến mịn: đoạn -> câu -> mệnh đề (dấu phẩy) -> từ
_SPLITTERS = [r"\n\s*\n", r"(?<=[.!?;:])\s+", r"(?<=,)\s+", r"\s+"]


class Chunker:
    """Chia text thành chunk <= max_tokens (đếm bằng tokenizer của model), chỉ cắt ở ranh giới câu/mệnh đề/từ."""

    def __init__(self, tokenizer, max_tokens: int, overlap_tokens: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _pieces(self, text: str, level: int = 0):
        text = " ".join(text.split())
        if not text:
            return []
        n = self.count(text)
        if n <= self.max_tokens:
            return [(text, n)]
        if level >= len(_SPLITTERS):  # một "từ" dài hơn cả giới hạn: cắt cứng theo ký tự
            half = len(text) // 2
            return self._pieces(text[:half], level) + self._pieces(text[half:], level)
        out = []
        for part in re.split(_SPLITTERS[level], text):
            out.extend(self._pieces(part, level + 1))
        return out

    def chunk(self, text: str):
        pieces = []
        for paragraph in re.split(_SPLITTERS[0], text):
            pieces.extend(self._pieces(paragraph, 1))

        chunks, current, total = [], [], 0
        for piece, n in pieces:
            if current and total + n > self.max_tokens:
                chunks.append(" ".join(t for t, _ in current))
                # phần chồng lấp: các piece cuối của chunk trước (nguyên câu/mệnh đề/từ)
                tail, tail_n = [], 0
                for item in reversed(current):
                    if tail_n + item[1] > self.overlap_tokens or tail_n + item[1] + n > self.max_tokens:
                        break
                    tail.insert(0, item)
                    tail_n += item[1]
                current, total = tail, tail_n
            current.append((piece, n))
            total += n
        if current:
            chunks.append(" ".join(t for t, _ in current))
        return chunks



class Ingestor:
    """Giữ model + kết nối Qdrant trong bộ nhớ để nạp nhiều file mà không phải load lại."""

    def __init__(self, model_name=MODEL_NAME, qdrant_url="http://localhost:6333", collection=DEFAULT_COLLECTION,
                 max_tokens=1024, overlap_tokens=50, batch_size=16, device=None):
        self.collection = collection
        self.batch_size = batch_size
        # GPU nếu có (fp16: bge-m3 ~1,1 GB thay vì ~2,2 GB, để còn chỗ cho LLM trong 4 GB VRAM)
        device = device or os.getenv("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = SentenceTransformer(
            model_name, device=device, model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
        )
        if max_tokens > self.model.max_seq_length - 12:
            raise ValueError(
                f"max_tokens phải <= {self.model.max_seq_length - 12} (giới hạn của model là {self.model.max_seq_length})"
            )
        self.chunker = Chunker(self.model.tokenizer, max_tokens, overlap_tokens)
        # họ e5 yêu cầu tiền tố "passage: " cho tài liệu
        self.prefix = "passage: " if "e5" in model_name.lower() else ""
        self.dim = self.model.get_sentence_embedding_dimension()
        self.client = QdrantClient(url=qdrant_url)
        self.ensure_collection()

    def ensure_collection(self, recreate: bool = False):
        """Tạo collection; tạo lại nếu recreate hoặc số chiều lệch với model."""
        client = self.client
        if client.collection_exists(self.collection):
            existing = client.get_collection(self.collection).config.params.vectors.size
            if recreate or existing != self.dim:
                if existing != self.dim:
                    print(f"Collection {self.collection} có {existing} chiều, model này cho {self.dim} chiều -> tạo lại.")
                client.delete_collection(self.collection)
        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
            for field in ("folder", "source"):  # để xoá/lọc theo file nhanh
                client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    def ingest_file(self, pdf: Path, folder: str, progress=None) -> int:
        """Nạp một PDF; ghi đè dữ liệu cũ của cùng file. Trả về số chunk.

        `progress(stage, done, total)` (tuỳ chọn) được gọi khi tiến độ thay đổi; stage là
        reading (trang) | chunking (trang) | embedding (chunk) | saving (chunk).
        """
        report = progress or (lambda *_: None)
        records = []
        pages = read_pdf_pages(pdf, on_page=lambda n, total: report("reading", n, total))
        for done_pages, (page_number, page_text) in enumerate(pages, start=1):
            report("chunking", done_pages, len(pages))
            for chunk in self.chunker.chunk(page_text):
                index = len(records)
                # ID cố định theo (folder, file, trang, vị trí chunk) -> chạy lại là ghi đè, không nhân đôi
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{folder}/{pdf.name}/{page_number}/{index}"))
                records.append(
                    {
                        "id": point_id,
                        "text": chunk,
                        "source": pdf.name,
                        "folder": folder,
                        "page": page_number,
                        "chunk_index": index,
                        "language": "vi",
                    }
                )
        if not records:
            raise ValueError("PDF không có text (có thể là file scan, chưa hỗ trợ OCR)")

        embeddings = []
        for start in range(0, len(records), self.batch_size):
            report("embedding", start, len(records))
            embeddings.extend(self.model.encode(
                [self.prefix + r["text"] for r in records[start:start + self.batch_size]],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist())
        report("embedding", len(records), len(records))

        # xoá chunk cũ của file này (file mới có thể ít chunk hơn bản cũ)
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="folder", match=models.MatchValue(value=folder)),
                    models.FieldCondition(key="source", match=models.MatchValue(value=pdf.name)),
                ])
            ),
            wait=True,
        )
        for start in range(0, len(records), self.batch_size):
            report("saving", start, len(records))
            batch = zip(records[start:start + self.batch_size], embeddings[start:start + self.batch_size])
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=record["id"],
                        vector=vector,
                        payload={k: v for k, v in record.items() if k != "id"},
                    )
                    for record, vector in batch
                ],
                wait=True,
            )
        return len(records)
