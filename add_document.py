"""Nạp các PDF trong documents/<folder> vào Qdrant (chạy một lần, tải model mỗi lần chạy).

Chạy:
    python add_document.py --folder thong-tin-va-dieu-do-trong-he-thong-dien
    python add_document.py --all --recreate      # xoá collection rồi nạp lại toàn bộ documents/
Muốn nạp nhiều lần mà không load lại model thì dùng dịch vụ: uvicorn app:app
"""
import argparse
import sys
from pathlib import Path

from ingest import DEFAULT_COLLECTION, DOCUMENTS_DIR, MODEL_NAME, Ingestor


def main():
    parser = argparse.ArgumentParser(description="Nạp PDF trong documents/<folder> vào Qdrant")
    parser.add_argument("--folder", help="tên thư mục con trong documents/ (hoặc đường dẫn)")
    parser.add_argument("--all", action="store_true", help="nạp mọi thư mục con của documents/")
    parser.add_argument("--device", help="cuda | cpu (mặc định: cuda nếu có, hoặc biến EMBED_DEVICE)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="số token tối đa mỗi chunk (chừa chỗ cho token đặc biệt và tiền tố)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="số token chồng lấp giữa hai chunk liền kề")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recreate", action="store_true", help="xóa collection nếu đã tồn tại rồi tạo lại")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if bool(args.folder) == args.all:
        sys.exit("Chọn đúng một trong hai: --folder <tên> hoặc --all")
    if args.all:
        folders = sorted(p for p in DOCUMENTS_DIR.iterdir() if p.is_dir() and any(p.glob("*.pdf")))
    else:
        folder = Path(args.folder)
        if not folder.is_dir():
            folder = DOCUMENTS_DIR / args.folder
        if not folder.is_dir():
            sys.exit(f"Không tìm thấy thư mục: {args.folder}")
        folders = [folder]
    if not any(any(f.glob("*.pdf")) for f in folders):
        sys.exit("Không có file PDF nào để nạp")

    try:
        ingestor = Ingestor(args.model, args.qdrant_url, args.collection,
                            args.max_tokens, args.chunk_overlap, args.batch_size, args.device)
    except ValueError as e:
        sys.exit(str(e))
    if args.recreate:
        ingestor.ensure_collection(recreate=True)

    print(f"Embedding trên: {ingestor.device}")
    total = 0
    for folder in folders:
        print(folder.name)
        for pdf in sorted(folder.glob("*.pdf")):
            try:
                count = ingestor.ingest_file(pdf, folder.name)
            except ValueError as e:  # vd PDF scan không có text: bỏ qua file đó, nạp tiếp
                print(f"  {pdf.name}: BỎ QUA ({e})")
                continue
            print(f"  {pdf.name}: {count} chunk")
            total += count

    info = ingestor.client.get_collection(args.collection)
    print("Inserted:", total, "points")
    print("Collection status:", info.status)
    print("Stored points:", info.points_count)


if __name__ == "__main__":
    main()
