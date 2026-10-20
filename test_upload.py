"""Test HTTP của /upload nhiều file và /batches. Không nạp model, không cần Qdrant:
chỉ dựng app không chạy lifespan, DB và thư mục tài liệu trỏ vào thư mục tạm."""
import pytest
from fastapi.testclient import TestClient

import app as app_module

PDF = b"%PDF-1.4\n%fake body\n"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "JOBS_DB", tmp_path / "jobs.db")
    monkeypatch.setattr(app_module, "DOCUMENTS_DIR", tmp_path / "documents")
    while not app_module.jobs_queue.empty():
        app_module.jobs_queue.get_nowait()
    app_module.init_db()
    return TestClient(app_module.app)  # không dùng `with`: bỏ qua lifespan (model, llama-server, worker)


def pdfs(*names):
    return [("files", (n, PDF, "application/pdf")) for n in names]


def test_upload_several_files_creates_one_job_each(client, tmp_path):
    r = client.post("/upload", data={"folder": "demo"}, files=pdfs("a.pdf", "b.pdf"))

    assert r.status_code == 202
    body = r.json()
    assert body["batch_id"]
    assert body["folder"] == "demo"
    assert [j["filename"] for j in body["jobs"]] == ["a.pdf", "b.pdf"]
    assert all(j["status"] == "processing" and j["job_id"] for j in body["jobs"])
    assert body["rejected"] == []
    assert (tmp_path / "documents" / "demo" / "a.pdf").read_bytes() == PDF
    assert (tmp_path / "documents" / "demo" / "b.pdf").read_bytes() == PDF


def test_bad_files_are_rejected_individually_and_good_ones_still_accepted(client, tmp_path):
    files = [
        ("files", ("good.pdf", PDF, "application/pdf")),
        ("files", ("notes.txt", b"hello", "text/plain")),
        ("files", ("empty.pdf", b"", "application/pdf")),
        ("files", ("fake.pdf", b"not a pdf at all", "application/pdf")),
    ]
    r = client.post("/upload", data={"folder": "demo"}, files=files)

    assert r.status_code == 202
    body = r.json()
    assert [j["filename"] for j in body["jobs"]] == ["good.pdf"]
    assert {x["filename"] for x in body["rejected"]} == {"notes.txt", "empty.pdf", "fake.pdf"}
    assert all(x["reason"] for x in body["rejected"])
    left = sorted(p.name for p in (tmp_path / "documents" / "demo").iterdir())
    assert left == ["good.pdf"]  # file bị loại không để lại gì trong documents/


def test_all_files_invalid_returns_400(client):
    r = client.post("/upload", data={"folder": "demo"}, files=[("files", ("a.txt", b"x", "text/plain"))])
    assert r.status_code == 400


def test_bad_folder_returns_400(client):
    r = client.post("/upload", data={"folder": "../etc"}, files=pdfs("a.pdf"))
    assert r.status_code == 400


def test_duplicate_name_in_batch_keeps_first_and_rejects_later(client):
    first, second = PDF + b"first", PDF + b"second"
    files = [("files", ("dup.pdf", first, "application/pdf")), ("files", ("dup.pdf", second, "application/pdf"))]
    body = client.post("/upload", data={"folder": "demo"}, files=files).json()

    assert [j["filename"] for j in body["jobs"]] == ["dup.pdf"]
    assert [x["filename"] for x in body["rejected"]] == ["dup.pdf"]
    assert "trùng" in body["rejected"][0]["reason"]
    assert (app_module.DOCUMENTS_DIR / "demo" / "dup.pdf").read_bytes() == first


def test_too_many_files_rejects_whole_request(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_FILES_PER_UPLOAD", 2)
    r = client.post("/upload", data={"folder": "demo"}, files=pdfs("a.pdf", "b.pdf", "c.pdf"))

    assert r.status_code == 413
    assert not (app_module.DOCUMENTS_DIR / "demo").exists() or not list((app_module.DOCUMENTS_DIR / "demo").iterdir())


def test_batch_total_size_over_limit_rejects_whole_request_and_leaves_nothing(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_BATCH_BYTES", len(PDF) * 2 - 1)
    r = client.post("/upload", data={"folder": "demo"}, files=pdfs("a.pdf", "b.pdf"))

    assert r.status_code == 413
    assert list((app_module.DOCUMENTS_DIR / "demo").iterdir()) == []
    assert app_module.jobs_queue.empty()


def test_legacy_single_file_field_still_works(client):
    r = client.post("/upload", data={"folder": "demo"}, files=[("file", ("one.pdf", PDF, "application/pdf"))])

    assert r.status_code == 202
    assert [j["filename"] for j in r.json()["jobs"]] == ["one.pdf"]


class StubIngestor:
    def ingest_file(self, pdf, folder, progress=None):
        if pdf.name == "bad.pdf":
            raise RuntimeError("boom")
        return 3


def test_batch_status_reports_each_job_and_totals(client, monkeypatch):
    monkeypatch.setitem(app_module.state, "ingestor", StubIngestor())
    body = client.post("/upload", data={"folder": "demo"}, files=pdfs("ok.pdf", "bad.pdf", "later.pdf")).json()

    before = client.get(f"/batches/{body['batch_id']}").json()
    assert before["counts"] == {"processing": 3, "done": 0, "failed": 0}

    for job in body["jobs"][:2]:  # worker xử lý hai job đầu
        app_module.process_job(job["job_id"])
    after = client.get(f"/batches/{body['batch_id']}").json()

    assert after["folder"] == "demo"
    assert after["counts"] == {"processing": 1, "done": 1, "failed": 1}
    by_name = {j["filename"]: j for j in after["jobs"]}
    assert [j["filename"] for j in after["jobs"]] == ["ok.pdf", "bad.pdf", "later.pdf"]  # đúng thứ tự chọn
    assert by_name["ok.pdf"]["status"] == "done" and by_name["ok.pdf"]["chunks"] == 3
    assert by_name["bad.pdf"]["status"] == "failed" and "boom" in by_name["bad.pdf"]["error"]
    assert by_name["later.pdf"]["status"] == "processing"


def test_unknown_batch_returns_404(client):
    assert client.get("/batches/nope").status_code == 404
