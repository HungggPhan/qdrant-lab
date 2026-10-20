"""Tạo vi_syllables.txt: các âm tiết tiếng Việt thật, xuất hiện nguyên vẹn (>= --min-count lần) trong documents/.

vi_text.merge_split_syllables dùng danh sách này để quyết định các ca mơ hồ ("thi ết" -> "thiết" nhưng
"cho ảnh" giữ nguyên). Chạy lại khi thêm nhiều tài liệu tiếng Việt mới:
    python build_vocab.py
"""
import argparse
import collections
import sys
import unicodedata

from ingest import DOCUMENTS_DIR, read_pdf_pages
from vi_text import VOCAB_FILE, _TOKEN, _is_syllable


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-count", type=int, default=3)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    counts = collections.Counter()
    pdfs = sorted(DOCUMENTS_DIR.glob("*/*.pdf"))
    for pdf in pdfs:
        for _, text in read_pdf_pages(pdf, repair_spaces=False):  # text chưa nối, nếu không sẽ tự lặp
            for token in text.split():
                m = _TOKEN.fullmatch(token)
                if m:
                    counts[unicodedata.normalize("NFC", m.group(2).lower())] += 1

    words = sorted(w for w, n in counts.items() if n >= args.min_count and _is_syllable(w))
    VOCAB_FILE.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words)} âm tiết từ {len(pdfs)} file -> {VOCAB_FILE.name}")


if __name__ == "__main__":
    main()
