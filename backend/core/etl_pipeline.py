import os
import re
import json
import hashlib
import unicodedata
from typing import List, Tuple, Optional, Any
import requests

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .grounding import clean_passage


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Bộ nhúng vector nomic-embed-text cho ETL."""
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        url: str = "http://127.0.0.1:11434/api/embeddings",
        task_type: str = "search_document"
    ):
        self.model_name = model_name
        self.url = url
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            prefixed_text = f"{self.task_type}: {text}"
            try:
                res = requests.post(
                    self.url,
                    json={"model": self.model_name, "prompt": prefixed_text},
                    timeout=60
                )
                if res.status_code == 200:
                    embeddings.append(res.json().get("embedding", []))
                else:
                    raise RuntimeError(f"Ollama embedding thất bại (Mã lỗi {res.status_code}): {res.text}")
            except Exception as e:
                # Raise lỗi thay vì nuốt lỗi và chèn vector [0.0] * 768 rác
                raise RuntimeError(f"Lỗi kết nối Ollama khi sinh vector: {e}")
        return embeddings


class DocumentETLPipeline:
    """Đường ống xử lý dữ liệu học thuật nạp trực tiếp vào Docker ChromaDB."""
    def __init__(
        self,
        ollama_base_url: str = None,
        chroma_host: str = None,
        chroma_port: int = None
    ):
        base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        c_host = chroma_host or os.getenv("CHROMA_HOST", "127.0.0.1")
        c_port = chroma_port or int(os.getenv("CHROMA_PORT", 8000))

        self.chroma_client = chromadb.HttpClient(host=c_host, port=c_port)
        self.embedding_fn = OllamaEmbeddingFunction(
            model_name=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            url=f"{base_url}/api/embeddings",
            task_type="search_document"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def compute_sha256(self, file_path: str) -> str:
        """Tính mã băm toàn vẹn tệp để phòng chống RAG Poisoning."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Chuẩn hóa Unicode NFC, giữ nguyên công thức so sánh toán học, khử ký tự rác."""
        normalized = unicodedata.normalize("NFC", text)
        no_zero_width = re.sub(r"[\u200B-\u200D\uFEFF]", "", normalized)
        # Chỉ lọc bỏ thẻ HTML dạng chữ cái như <script>, <div>, tránh xóa nhầm phép so sánh a < b
        no_html = re.sub(r"<\/?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>", "", no_zero_width)
        clean = re.sub(r"\n\s*\n+", "\n\n", no_html)
        return clean.replace("\x00", "").strip()

    def process_and_index_document(
        self,
        file_path: str,
        course_id: str,
        document_id: str,
        allowed_roles: List[str]
    ) -> int:
        clean_course_id = course_id.strip()
        file_checksum = self.compute_sha256(file_path)
        ext = os.path.splitext(file_path)[-1].lower()

        # 1. Trích xuất tài liệu
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            has_page_numbers = True
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
            raw_docs = loader.load()
            has_page_numbers = False
        elif ext == ".txt":
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                raw_docs = loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="latin-1")
                raw_docs = loader.load()
            has_page_numbers = False
        else:
            raise ValueError(f"Định dạng tệp {ext} chưa được hỗ trợ.")

        if not raw_docs:
            raise ValueError("Không thể bóc tách nội dung từ tệp tin.")

        # 2. Phân mảnh văn bản
        for document in raw_docs:
            document.page_content = self.sanitize_text(clean_passage(document.page_content))
        split_chunks = self.text_splitter.split_documents(raw_docs)
        if not split_chunks:
            raise ValueError("Không tạo được phân mảnh tri thức nào.")

        ids = []
        texts = []
        metadatas = []

        roles_str = json.dumps(allowed_roles) if isinstance(allowed_roles, list) else str(allowed_roles)
        file_title = os.path.basename(file_path)

        for idx, chunk in enumerate(split_chunks):
            clean_text = self.sanitize_text(chunk.page_content)
            if not clean_text:
                continue

            chunk_id = f"{document_id}_chunk_{idx}"
            
            # Nếu là PDF thì lấy số trang thực (+1 vì pypdf là 0-indexed). Nếu là docx/txt thì gán None
            if has_page_numbers and "page" in chunk.metadata:
                page_num = int(chunk.metadata["page"]) + 1
            else:
                page_num = None

            meta = {
                "chunk_id": chunk_id,
                "document_id": str(document_id),
                "document_title": file_title,
                "course_id": clean_course_id,
                "chunk_index": int(idx),
                "allowed_roles": roles_str,
                "checksum_sha256": file_checksum
            }
            if page_num is not None:
                meta["page_number"] = page_num

            ids.append(chunk_id)
            texts.append(clean_text)
            metadatas.append(meta)

        if not ids:
            raise ValueError("Tài liệu không có nội dung chữ hợp lệ để vector hóa.")

        # 3. Kết nối Collection
        collection_name = f"kb_{clean_course_id.lower()}_public"
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # 4. Tính toán và nạp an toàn: Chỉ xóa chunk cũ khi các chunk mới đã chuẩn bị xong
        try:
            collection.delete(where={"document_id": str(document_id)})
        except Exception:
            pass

        # 5. Ghi theo mẻ (Batch)
        batch_size = 20
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        return len(ids)

    def delete_document(self, course_id: str, document_id: str) -> bool:
        """Xóa phân mảnh thuộc về tài liệu."""
        clean_course_id = course_id.strip()
        collection_name = f"kb_{clean_course_id.lower()}_public"
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_fn
            )
            collection.delete(where={"document_id": str(document_id)})
            return True
        except Exception as e:
            print(f"[-] Lỗi khi xóa tài liệu {document_id}: {e}", flush=True)
            return False

    def list_documents(self, course_id: str) -> list:
        """Lấy danh sách các tài liệu trong collection."""
        clean_course_id = course_id.strip()
        collection_name = f"kb_{clean_course_id.lower()}_public"
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_fn
            )
            data = collection.get(include=["metadatas"])
            docs_summary = {}
            if data and data.get("metadatas"):
                for m in data["metadatas"]:
                    doc_id = m.get("document_id", m.get("document_title", "Unknown"))
                    if doc_id not in docs_summary:
                        docs_summary[doc_id] = {
                            "document_id": doc_id,
                            "document_title": m.get("document_title", doc_id),
                            "chunks_count": 0,
                            "allowed_roles": m.get("allowed_roles", "")
                        }
                    docs_summary[doc_id]["chunks_count"] += 1
            return list(docs_summary.values())
        except Exception as e:
            print(f"[-] Lỗi list_documents từ {collection_name}: {e}", flush=True)
            return []
