"""Nạp tài liệu: chuẩn bị toàn bộ vector trước khi thay dữ liệu cũ."""
import hashlib
import json
import math
import os
from pathlib import Path
import requests
import chromadb
from chromadb.errors import NotFoundError
from chromadb.api.types import EmbeddingFunction
from .document_processing import CHUNKING_VERSION, prepare_chunks, read_pages, sanitize_text


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text", url="http://127.0.0.1:11434/api/embed", task_type="search_document"):
        self.model_name = model_name
        self.url = url
        self.task_type = task_type

    def __call__(self, input):
        if not input:
            return []
        response = requests.post(self.url, json={"model": self.model_name,
            "input": [f"{self.task_type}: {text}" for text in input], "truncate": False}, timeout=120)
        response.raise_for_status()
        embeddings = response.json().get("embeddings") or []
        if len(embeddings) != len(input) or any(not vector or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in vector) for vector in embeddings):
            raise RuntimeError("Embedding rỗng, không hợp lệ hoặc sai số lượng; dữ liệu cũ được giữ lại.")
        if len({len(vector) for vector in embeddings}) != 1:
            raise RuntimeError("Các embedding không cùng số chiều.")
        if any(not any(vector) for vector in embeddings):
            raise RuntimeError("Embedding toàn số 0; không cập nhật tài liệu.")
        return embeddings


class DocumentETLPipeline:
    def __init__(self, ollama_base_url=None, chroma_host=None, chroma_port=None):
        base = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.chroma_client = chromadb.HttpClient(host=chroma_host or os.getenv("CHROMA_HOST", "127.0.0.1"),
            port=chroma_port or int(os.getenv("CHROMA_PORT", "8000")))
        self.embedding_fn = OllamaEmbeddingFunction(model_name=os.getenv("EMBED_MODEL", "nomic-embed-text"), url=f"{base}/api/embed")
        self.chunk_size = int(os.getenv("DOCUMENT_CHUNK_SIZE", "1400"))
        self.chunk_overlap = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "200"))

    @staticmethod
    def compute_sha256(file_path):
        hasher = hashlib.sha256()
        with open(file_path, "rb") as stream:
            while chunk := stream.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    sanitize_text = staticmethod(sanitize_text)

    @staticmethod
    def _collection_name(course_id):
        return f"kb_{course_id.strip().lower()}_public"

    def process_and_index_document(self, file_path, course_id, document_id, allowed_roles,
                                   expected_checksum=None, document_title=None, review_metadata=None):
        if not isinstance(allowed_roles, list) or not allowed_roles:
            raise ValueError("Cần chọn ít nhất một vai trò được phép đọc tài liệu.")
        roles = json.dumps(sorted(set(allowed_roles)))
        checksum = self.compute_sha256(file_path)
        if expected_checksum and checksum != expected_checksum:
            raise ValueError("Hash tài liệu không khớp bản đã được kiểm tra.")
        chunks = prepare_chunks(read_pages(file_path), self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("Tài liệu không có nội dung có thể lập chỉ mục.")
        if self.compute_sha256(file_path) != checksum:
            raise ValueError("File thay đổi trong lúc đọc; hãy kiểm tra lại.")
        collection = self.chroma_client.get_or_create_collection(name=self._collection_name(course_id),
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_fn.model_name})
        model = (collection.metadata or {}).get("embedding_model")
        if model and model != self.embedding_fn.model_name:
            raise ValueError("Kho môn học dùng embedding khác. Cần tạo kho mới hoặc nạp lại toàn bộ môn với cùng mô hình.")
        if not model and self.embedding_fn.model_name != "nomic-embed-text" and collection.count():
            raise ValueError("Kho cũ chưa có thông tin embedding; không thể trộn mô hình mới vào kho này.")
        old_ids = set(collection.get(where={"document_id": str(document_id)}, include=[]).get("ids") or [])
        revision = hashlib.sha256(f"{checksum}:{roles}:{CHUNKING_VERSION}:{self.chunk_size}:{self.chunk_overlap}".encode()).hexdigest()[:16]
        prefix = f"{document_id}_{revision}"
        ids, texts, metas = [], [], []
        for index, (text, page_meta) in enumerate(chunks):
            cid = f"{prefix}_chunk_{index}"
            ids.append(cid)
            path = " / ".join(dict.fromkeys(s for s in (page_meta["chapter_title"], page_meta["section_title"]) if s))
            texts.append(f"{path}\n{text}" if path else text)
            metas.append({**page_meta, "chunk_id": cid, "chunk_prefix": prefix,
                "document_id": str(document_id), "document_title": document_title or Path(file_path).name,
                "course_id": course_id.strip().upper(), "chunk_index": index,
                "allowed_roles": roles, "checksum_sha256": checksum, "embedding_model": self.embedding_fn.model_name,
                **(review_metadata or {})})
        batch_size = 20
        embeddings = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(self.embedding_fn(texts[start:start + batch_size]))
        if len({len(vector) for vector in embeddings}) != 1:
            raise RuntimeError("Số chiều embedding thay đổi giữa các mẻ; không cập nhật tài liệu.")
        if self.compute_sha256(file_path) != checksum:
            raise ValueError("File thay đổi trong lúc xử lý; không cập nhật chỉ mục.")
        try:
            for start in range(0, len(ids), batch_size):
                collection.upsert(ids=ids[start:start + batch_size], documents=texts[start:start + batch_size],
                    metadatas=metas[start:start + batch_size], embeddings=embeddings[start:start + batch_size])
        except Exception:
            fresh_ids = list(set(ids) - old_ids)
            if fresh_ids:
                collection.delete(ids=fresh_ids)
            raise
        obsolete = list(old_ids - set(ids))
        if obsolete:
            collection.delete(ids=obsolete)
        if not model:
            collection.modify(metadata={**(collection.metadata or {}), "embedding_model": self.embedding_fn.model_name})
        return len(ids)

    def delete_document(self, course_id, document_id):
        try:
            collection = self.chroma_client.get_collection(name=self._collection_name(course_id))
            collection.delete(where={"document_id": str(document_id)})
            return True
        except Exception:
            return False

    def delete_course(self, course_id):
        try:
            self.chroma_client.delete_collection(name=self._collection_name(course_id))
            return True
        except NotFoundError:
            return True
        except Exception:
            return False

    def list_documents(self, course_id):
        try:
            collection = self.chroma_client.get_collection(name=self._collection_name(course_id))
            data = collection.get(include=["metadatas"])
            summaries = {}
            for meta in data.get("metadatas") or []:
                if not meta:
                    continue
                doc_id = meta.get("document_id", meta.get("document_title", "Unknown"))
                summary = summaries.setdefault(doc_id, {"document_id": doc_id,
                    "document_title": meta.get("document_title", doc_id), "chunks_count": 0,
                    "allowed_roles": meta.get("allowed_roles", ""), "chunking_version": meta.get("chunking_version", "legacy"),
                    "checksum_sha256": meta.get("checksum_sha256", ""),
                    "review_status": meta.get("review_status", "LEGACY")})
                summary["chunks_count"] += 1
            return list(summaries.values())
        except Exception:
            return []
