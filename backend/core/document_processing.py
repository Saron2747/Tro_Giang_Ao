"""Bóc tách và chia giáo trình theo trang, chương/mục trước khi vector hóa."""
import re
import unicodedata
from collections import Counter
from pathlib import Path
from pypdf import PdfReader
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKING_VERSION = "sections-v2"
MAX_DOCUMENT_PAGES = 1000
HEADING = re.compile(r"^(?:CHƯƠNG\s+\d+[.:]?|\d+(?:\.\d+)+\.?)\s+\S", re.I)
PAGE_NUMBER = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,8})$", re.I)


def sanitize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF\x00]", "", text)
    text = re.sub(r"<\s*(script|style)\b[^>]*>.*?</\s*\1\s*>", "", text, flags=re.S | re.I)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def is_catalog(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    leaders = sum(bool(re.search(r"\.{5,}|…{3,}", line)) for line in lines)
    return leaders >= 3 and leaders / max(1, len(lines)) >= 0.3


def read_pages(file_path, max_pages=MAX_DOCUMENT_PAGES, max_chars=2_000_000):
    extension = Path(file_path).suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(file_path)
        if len(reader.pages) > max_pages:
            raise ValueError(f"Tài liệu vượt giới hạn {max_pages} trang.")
        pages, total = [], 0
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            total += len(text)
            if total > max_chars:
                raise ValueError(f"Tài liệu vượt giới hạn {max_chars} ký tự.")
            pages.append({"text": text, "page_number": i + 1})
        return pages
    if extension in (".doc", ".docx"):
        docs = Docx2txtLoader(file_path).load()
    elif extension == ".txt":
        try:
            docs = TextLoader(file_path, encoding="utf-8").load()
        except UnicodeDecodeError:
            docs = TextLoader(file_path, encoding="latin-1").load()
    else:
        raise ValueError(f"Định dạng {extension} chưa được hỗ trợ.")
    if sum(len(doc.page_content) for doc in docs) > max_chars:
        raise ValueError(f"Tài liệu vượt giới hạn {max_chars} ký tự.")
    return [{"text": doc.page_content} for doc in docs]


def prepare_chunks(pages, chunk_size=1400, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    boundary = Counter()
    catalog_sections = set()
    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        boundary.update(set(lines[:2] + lines[-2:]))
        if is_catalog(page["text"]):
            for line in lines:
                number = re.match(r"^(\d+(?:\.\d+)+)\.?\s+", line)
                if number:
                    catalog_sections.add(number[1])
    recurring = {line for line, count in boundary.items()
                 if count >= max(3, len(pages) * 0.2) and len(line) < 160
                 and not HEADING.match(line) and not PAGE_NUMBER.fullmatch(line)}
    chunks = []
    chapter, section = "", ""
    for page in pages:
        text = sanitize_text(page["text"])
        if not text or is_catalog(text):
            continue
        lines = text.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        boundary_indices = set(nonempty[:2] + nonempty[-2:])
        printed = ""
        for i in sorted(boundary_indices):
            line = lines[i].strip()
            if page.get("page_number") is not None and PAGE_NUMBER.fullmatch(line) and (i == nonempty[0] or i == nonempty[-1]):
                printed = line
                lines[i] = ""
            elif line in recurring:
                lines[i] = ""
        buffer = []

        def flush():
            body = "\n".join(buffer).strip()
            if not body:
                return
            if "\n" not in body and HEADING.match(body):
                return
            for fragment in splitter.split_text(body):
                metadata = {"chapter_title": chapter, "section_title": section,
                            "printed_page": printed, "chunking_version": CHUNKING_VERSION}
                if page.get("page_number") is not None:
                    metadata["page_number"] = page["page_number"]
                if re.search(r"[∑∫√]", fragment):
                    metadata["extraction_warning"] = "Công thức trích xuất dạng văn bản; kiểm tra số mũ/chỉ số với trang PDF gốc."
                chunks.append((fragment, metadata))

        for line in lines:
            clean = line.strip()
            section_number = re.match(r"^(\d+(?:\.\d+)+)\.?\s+", clean)
            known_section = not catalog_sections or not section_number or section_number[1] in catalog_sections
            if len(clean) < 160 and HEADING.match(clean) and known_section:
                flush()
                buffer = []
                if clean.upper().startswith("CHƯƠNG"):
                    chapter = clean
                section = clean
            buffer.append(line.rstrip())
        flush()
    if not chunks:
        raise ValueError("Không trích xuất được nội dung chữ. PDF scan cần OCR trước khi nạp.")
    return chunks
