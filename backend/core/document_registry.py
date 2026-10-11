"""Persistent demo review queue; staged files never enter Chroma before approval."""
import hashlib
import json
import re
import sqlite3
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from .document_processing import read_pages, prepare_chunks, MAX_DOCUMENT_PAGES
from .security import resource_slot

_LOCK = RLock()
EDIT_ROLES = {"ROLE_LECTURER", "ROLE_ADMIN"}


class DocumentRegistry:
    def __init__(self, root, pipeline, guardrail):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pipeline, self.guardrail = pipeline, guardrail
        self.db_path = self.root / "document_registry.sqlite3"
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, course_id TEXT NOT NULL, title TEXT NOT NULL,
                stored_path TEXT NOT NULL, uploaded_by TEXT NOT NULL, uploaded_at TEXT NOT NULL,
                version INTEGER NOT NULL, roles TEXT NOT NULL, checksum TEXT NOT NULL,
                status TEXT NOT NULL, integrity TEXT NOT NULL, warnings TEXT NOT NULL,
                preview TEXT NOT NULL, chunks INTEGER NOT NULL DEFAULT 0,
                approved_by TEXT, approved_at TEXT, last_checked TEXT,
                UNIQUE(course_id, title, version))""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            if "uploader_name" not in columns:
                db.execute("ALTER TABLE documents ADD COLUMN uploader_name TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _authorize(role):
        if role not in EDIT_ROLES:
            raise PermissionError("Chỉ giảng viên và quản trị viên được quản lý tài liệu.")

    @staticmethod
    def _course(course):
        course = course.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{1,31}", course):
            raise ValueError("Mã học phần cần 2–32 chữ cái, chữ số, dấu gạch ngang hoặc gạch dưới.")
        return course

    @staticmethod
    def _hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def _path(self, stored_path):
        path = (self.root / stored_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Đường dẫn tài liệu nằm ngoài kho lưu trữ.")
        return path

    def list(self, course):
        with self._db() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM documents WHERE course_id=? ORDER BY uploaded_at DESC", (self._course(course),))]

    def get(self, doc_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise ValueError("Không tìm thấy tài liệu.")
        return dict(row)

    def _update(self, doc_id, **fields):
        permitted = {"integrity", "last_checked", "status", "chunks", "approved_by", "approved_at"}
        if not fields.keys() <= permitted:
            raise ValueError("Trường cập nhật không hợp lệ.")
        with self._db() as db:
            db.execute("UPDATE documents SET " + ", ".join(f"{key}=?" for key in fields) + " WHERE id=?",
                       (*fields.values(), doc_id))

    def _audit(self, category, user_id, doc_id, action="PASSED"):
        self.guardrail.log_security_event(user_id=user_id, event_category=category,
                                         guardrail_action=action, patterns=f"document_id={doc_id}")

    def stage(self, filename, content, course, roles, user_id, role, uploader_name=""):
        self._authorize(role)
        course = self._course(course)
        title = Path(filename.replace("\\", "/")).name
        ext = Path(title).suffix.lower()
        if not title or len(title) > 255 or ext not in {".pdf", ".docx", ".txt"}:
            raise ValueError("Chỉ nhận PDF, DOCX hoặc TXT có tên tối đa 255 ký tự.")
        if not content or len(content) > 20 * 1024 * 1024:
            raise ValueError("Mỗi tài liệu cần có nội dung và không vượt quá 20 MB.")
        if not roles or not set(roles) <= {"ROLE_STUDENT", "ROLE_LECTURER", "ROLE_ADMIN"}:
            raise ValueError("Cần chọn vai trò đọc hợp lệ.")
        if ext == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
            raise ValueError("Nội dung tệp không phải PDF.")
        doc_id = uuid.uuid4().hex
        stored = f"review/{doc_id}{ext}"
        path = self._path(stored)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, resource_slot() as acquired:
            if not acquired:
                raise RuntimeError("Hệ thống đang bận. Vui lòng thử lại sau.")
            path.write_bytes(content)
            try:
                if ext == ".docx":
                    with zipfile.ZipFile(path) as archive:
                        if "word/document.xml" not in archive.namelist() or sum(i.file_size for i in archive.infolist()) > 50 * 1024 * 1024:
                            raise ValueError("DOCX không hợp lệ hoặc vượt giới hạn giải nén 50 MB.")
                pages = read_pages(path)
                if len(pages) > MAX_DOCUMENT_PAGES or sum(len(p["text"]) for p in pages) > 2_000_000:
                    raise ValueError(f"Tài liệu vượt {MAX_DOCUMENT_PAGES} trang hoặc 2 triệu ký tự.")
                chunks = prepare_chunks(pages)
                if not chunks:
                    raise ValueError("Không trích xuất được nội dung để tra cứu. Hãy kiểm tra file hoặc OCR.")
                warnings = []
                for index, page in enumerate(pages):
                    if self.guardrail.scan_untrusted_text(page["text"]):
                        warnings.append(f"Trang/khối {index + 1}: có mẫu chỉ dẫn cần xem xét (có thể là ví dụ học thuật).")
                    if not page["text"].strip():
                        warnings.append(f"Trang/khối {index + 1}: không trích xuất được văn bản.")
                warnings.extend(sorted({m["extraction_warning"] for _, m in chunks if m.get("extraction_warning")}))
                now = datetime.now(timezone.utc).isoformat()
                with self._db() as db:
                    version = db.execute("SELECT COALESCE(MAX(version), 0)+1 FROM documents WHERE course_id=? AND title=?", (course, title)).fetchone()[0]
                    db.execute("""INSERT INTO documents
                        (id,course_id,title,stored_path,uploaded_by,uploaded_at,version,roles,checksum,status,integrity,warnings,preview)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (doc_id, course, title, stored, str(user_id), now, version,
                        json.dumps(roles), self._hash(path), "PENDING", "MATCH", json.dumps(warnings, ensure_ascii=False),
                        self.guardrail.redact_pii("\n\n".join(p["text"] for p in pages)[:6000])))
                    db.execute("UPDATE documents SET uploader_name=? WHERE id=?", (uploader_name, doc_id))
            except Exception:
                path.unlink(missing_ok=True)
                raise
        self._audit("DOCUMENT_STAGED", user_id, doc_id)
        if warnings:
            self._audit("DOCUMENT_REVIEW_WARNING", user_id, doc_id, "FLAGGED")
        return doc_id

    def check_integrity(self, doc_id, user_id, role):
        self._authorize(role)
        doc = self.get(doc_id)
        path = self._path(doc["stored_path"])
        try:
            integrity = "MISSING" if not path.is_file() else "MATCH" if self._hash(path) == doc["checksum"] else "MISMATCH"
        except OSError:
            integrity = "MISSING"
        self._update(doc_id, integrity=integrity, last_checked=datetime.now(timezone.utc).isoformat())
        self._audit("DOCUMENT_INTEGRITY_CHECK", user_id, doc_id, "PASSED" if integrity == "MATCH" else "BLOCKED")
        return integrity

    def validate_indexed(self, metadata):
        # Existing index stays usable until the lecturer explicitly migrates it.
        if metadata.get("review_status") != "APPROVED":
            return metadata.get("review_status", "LEGACY") == "LEGACY"
        with self._db() as db:
            row = db.execute("SELECT * FROM documents WHERE course_id=? AND title=? AND version=? AND status='APPROVED'",
                (metadata.get("course_id"), metadata.get("document_title"), metadata.get("document_version"))).fetchone()
        if not row or row["checksum"] != metadata.get("checksum_sha256"):
            return False
        path = self._path(row["stored_path"])
        try:
            if path.is_file() and self._hash(path) == row["checksum"]:
                return True
        except OSError:
            pass
        if row["integrity"] == "MATCH":
            self._update(row["id"], integrity="MISMATCH", last_checked=datetime.now(timezone.utc).isoformat())
            self._audit("DOCUMENT_INTEGRITY_CHECK", None, row["id"], "BLOCKED")
        return False

    def approve(self, doc_id, user_id, role, acknowledge=False):
        self._authorize(role)
        with _LOCK, resource_slot() as acquired:
            if not acquired:
                raise RuntimeError("Hệ thống đang bận. Vui lòng thử lại sau.")
            doc = self.get(doc_id)
            if doc["status"] not in {"PENDING", "APPROVED"}:
                raise ValueError("Tài liệu không ở trạng thái cho phép duyệt.")
            if not acknowledge:
                raise ValueError("Cần xác nhận đã xem nội dung và cảnh báo trước khi duyệt.")
            if self.check_integrity(doc_id, user_id, role) != "MATCH":
                raise ValueError("File thiếu hoặc hash không khớp. Hãy tải lại thành phiên bản mới.")
            count = self.pipeline.process_and_index_document(self._path(doc["stored_path"]), doc["course_id"], doc["title"],
                json.loads(doc["roles"]), expected_checksum=doc["checksum"], document_title=doc["title"],
                review_metadata={"review_status": "APPROVED", "document_version": doc["version"], "reviewed_by": str(user_id)})
            with self._db() as db:
                db.execute("UPDATE documents SET status='SUPERSEDED' WHERE course_id=? AND title=? AND status='APPROVED' AND id<>?", (doc["course_id"], doc["title"], doc_id))
            self._update(doc_id, status="APPROVED", chunks=count, approved_by=str(user_id), approved_at=datetime.now(timezone.utc).isoformat())
            self._audit("DOCUMENT_APPROVED", user_id, doc_id)
            return count

    def reject(self, doc_id, user_id, role):
        self._authorize(role)
        with _LOCK:
            if self.get(doc_id)["status"] != "PENDING":
                raise ValueError("Chỉ từ chối tài liệu đang chờ duyệt.")
            self._update(doc_id, status="REJECTED")
            self._audit("DOCUMENT_REJECTED", user_id, doc_id, "BLOCKED")

    def delete(self, doc_id, user_id, role):
        self._authorize(role)
        with _LOCK:
            doc = self.get(doc_id)
            if doc["status"] == "APPROVED" and not self.pipeline.delete_document(doc["course_id"], doc["title"]):
                raise RuntimeError("Không xóa được chỉ mục; file gốc được giữ lại.")
            self._path(doc["stored_path"]).unlink(missing_ok=True)
            self._update(doc_id, status="DELETED", integrity="MISSING")
            self._audit("DOCUMENT_DELETED", user_id, doc_id)

    def delete_course(self, course, user_id, role, confirmation):
        self._authorize(role)
        course = self._course(course)
        if confirmation.strip() != course:
            raise ValueError("Nhập chính xác mã học phần để xác nhận xóa.")
        with _LOCK:
            records = self.list(course)
            paths = [self._path(doc["stored_path"]) for doc in records]
            # Validate all legacy paths before removing the collection or any file.
            for folder in self.root.iterdir():
                if folder.is_dir() and folder.name.casefold() == course.casefold():
                    resolved = folder.resolve()
                    if not resolved.is_relative_to(self.root) or resolved == self.root:
                        raise ValueError("Thư mục học phần không nằm trong kho lưu trữ.")
                    for path in folder.iterdir():
                        if path.is_file():
                            if not path.resolve().is_relative_to(resolved):
                                raise ValueError("File gốc không nằm trong thư mục học phần.")
                            paths.append(path)
            if not self.pipeline.delete_course(course):
                raise RuntimeError("Không xóa được kho tra cứu. File và hồ sơ được giữ lại; hãy thử lại sau.")
            with self._db() as db:
                db.execute("UPDATE documents SET status='DELETED', integrity='MISSING' WHERE course_id=?", (course,))
            failed = []
            for path in set(paths):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    failed.append(path)
            self._audit("COURSE_DOCUMENTS_DELETED", user_id, course, "PARTIAL" if failed else "PASSED")
            if failed:
                raise RuntimeError("Đã xóa kho tra cứu, nhưng còn file chưa dọn được. Hãy kiểm tra quyền ghi và thử xóa môn lần nữa.")
