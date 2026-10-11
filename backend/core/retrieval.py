"""Tìm ngữ nghĩa + BM25, kiểm tra quyền hiện tại và giới hạn context."""
import json
import logging
import math
import os
import time
from threading import RLock
import requests
from .lexical_search import LexicalIndex, tokens

logger = logging.getLogger(__name__)
_INDEX_LOCK = RLock()


class ContextRetrieval:
    @staticmethod
    def _parse_roles(raw):
        if isinstance(raw, list):
            return [str(role) for role in raw]
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            parsed = json.loads(raw)
            return [str(role) for role in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            # Support legacy comma-separated roles, but reject malformed JSON.
            if raw.lstrip().startswith(("[", "{")):
                return []
            return [role.strip() for role in raw.split(",") if role.strip()]

    def _allowed(self, metadata, role):
        return isinstance(metadata, dict) and (role == "ROLE_ADMIN" or role in self._parse_roles(metadata.get("allowed_roles")))

    def _get_embedding(self, text):
        response = requests.post(self.ollama_embed_url, json={
            "model": self.embed_model_name, "input": f"search_query: {text}", "truncate": False,
        }, timeout=30)
        response.raise_for_status()
        vectors = response.json().get("embeddings") or []
        if not vectors or not vectors[0] or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in vectors[0]):
            raise RuntimeError("Ollama trả về vector không hợp lệ.")
        if not any(vectors[0]):
            raise RuntimeError("Ollama trả về vector toàn số 0.")
        return vectors[0]

    def invalidate_retrieval_cache(self, course_id):
        with _INDEX_LOCK:
            self._lexical_cache.pop(f"kb_{course_id.strip().lower()}_public", None)

    def _lexical_index(self, collection, name):
        # Only text/IDs are cached. Access rights and content are re-read before use.
        with _INDEX_LOCK:
            cached = self._lexical_cache.get(name)
            if cached and time.monotonic() - cached[0] < 60:
                return cached[1]
            records, offset = [], 0
            while True:
                page = collection.get(limit=500, offset=offset, include=["documents"])
                ids = page.get("ids") or []
                records.extend((cid, text or "") for cid, text in zip(ids, page.get("documents") or []))
                if len(ids) < 500:
                    break
                offset += len(ids)
            index = LexicalIndex(records)
            self._lexical_cache[name] = (time.monotonic(), index)
            return index

    def retrieve_safe_context(self, query, course_id, user_role, top_k=4, context_budget=None):
        if not self.chroma_client or top_k < 1:
            return "", []
        name = f"kb_{course_id.strip().lower()}_public"
        try:
            collection = self.chroma_client.get_collection(name=name)
            metadata = collection.metadata or {}
            indexed_model = metadata.get("embedding_model")
            if indexed_model and indexed_model != self.embed_model_name:
                logger.error("Embedding model mismatch; reindex required for %s", name)
                return "", []
            total = collection.count()
            if not total:
                return "", []
        except Exception:
            logger.exception("Không mở được kho tài liệu %s", name)
            return "", []

        candidate_limit = min(total, max(32, top_k * 8))
        vector_ids = []
        candidates = {}
        try:
            results = collection.query(query_embeddings=[self._get_embedding(query)], n_results=candidate_limit,
                                       include=["documents", "metadatas", "distances"])
            for cid, doc, meta, distance in zip(results["ids"][0], results["documents"][0],
                                               results["metadatas"][0], results["distances"][0]):
                if doc and self._allowed(meta, user_role) and math.isfinite(distance) and distance <= self.COSINE_DISTANCE_THRESHOLD:
                    vector_ids.append(cid)
                    candidates[cid] = (doc, meta, distance)
        except Exception:
            logger.exception("Tìm ngữ nghĩa thất bại; thử tìm từ khóa")

        lexical_ids = []
        try:
            hits = self._lexical_index(collection, name).search(query, candidate_limit)
            if hits:
                current = collection.get(ids=[cid for cid, _ in hits], include=["documents", "metadatas"])
                fresh = {cid: (doc, meta) for cid, doc, meta in zip(current.get("ids") or [], current.get("documents") or [], current.get("metadatas") or [])}
                wanted = set(tokens(query))
                for cid, _ in hits:
                    doc, meta = fresh.get(cid, (None, None))
                    # Cached IDs may now refer to changed content or different roles.
                    if doc and self._allowed(meta, user_role) and len(wanted.intersection(tokens(doc))) >= max(1, math.ceil(len(wanted) * 0.5)):
                        lexical_ids.append(cid)
                        candidates.setdefault(cid, (doc, meta, None))
        except Exception:
            logger.exception("Tìm từ khóa thất bại")

        if not candidates:
            return "", []
        # Reciprocal rank fusion avoids combining incomparable BM25/cosine scores.
        scores = {}
        for ranking in (vector_ids, lexical_ids):
            for rank, cid in enumerate(ranking, 1):
                scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)
        # Deduplicate overlapping chunks while keeping distinct facts for comparisons.
        selected, seen = [], set()
        for cid in ranked:
            doc, meta, _ = candidates[cid]
            words = set(tokens(doc))
            duplicate = any(len(words & old) / max(1, len(words | old)) > 0.85 for old in seen)
            if duplicate:
                continue
            selected.append((doc, meta))
            seen.add(frozenset(words))
            if len(selected) >= top_k:
                break

        if not selected:
            return "", []
        seen_ids = {m.get("chunk_id") for _, m in selected}
        before, after = self._neighbor_chunks(collection, selected[0][1], user_role, seen_ids)
        selected = selected[:1] + before + after + selected[1:]
        guard = getattr(self, "guardrail", None)
        validator = getattr(self, "document_validator", None)
        safe = []
        for doc, meta in selected:
            if validator:
                try:
                    if not validator(meta):
                        continue
                except Exception:
                    logger.exception("Không xác minh được nguồn; bỏ qua đoạn %s", meta.get("chunk_id"))
                    continue
            if guard and guard.scan_actionable_text(doc):
                guard.log_security_event(event_category="INDIRECT_PROMPT_INJECTION", guardrail_action="BLOCKED",
                                         patterns=f"chunk_id={meta.get('chunk_id', '')}")
                continue
            clean = guard.redact_pii(doc) if guard else doc
            if guard and clean != doc:
                guard.log_security_event(event_category="PII_CONTEXT_REDACTED", guardrail_action="SANITIZED", patterns=f"chunk_id={meta.get('chunk_id', '')}")
            clean_meta = dict(meta)
            if guard:
                for key in ("document_title", "section_title", "extraction_warning"):
                    if isinstance(clean_meta.get(key), str):
                        clean_meta[key] = guard.redact_pii(clean_meta[key])
            safe.append((clean, clean_meta))
        selected = safe
        return self._format_context(selected, context_budget)

    def _neighbor_chunks(self, collection, meta, user_role, seen_ids):
        try:
            idx = int(meta["chunk_index"])
            doc_id = meta["document_id"]
        except (KeyError, ValueError, TypeError):
            return [], []
        prefix = meta.get("chunk_prefix", str(doc_id))
        prev_id, next_id = f"{prefix}_chunk_{idx - 1}", f"{prefix}_chunk_{idx + 1}"
        try:
            got = collection.get(ids=[prev_id, next_id], include=["documents", "metadatas"])
        except Exception:
            return [], []
        found = {}
        for cid, doc, neighbor in zip(got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []):
            if not doc or cid in seen_ids or not self._allowed(neighbor, user_role):
                continue
            if neighbor.get("document_id") != doc_id:
                continue
            if meta.get("section_title") and neighbor.get("section_title") != meta["section_title"]:
                continue
            found[cid] = (doc, neighbor)
        return ([found[prev_id]] if prev_id in found else []), ([found[next_id]] if next_id in found else [])

    @staticmethod
    def _format_context(selected, context_budget=None):
        budget = int(os.getenv("RAG_CONTEXT_CHARS", "11000"))
        if context_budget is not None:
            budget = min(budget, max(0, context_budget))
        blocks, citations = [], []
        for doc, meta in selected:
            title = meta.get("document_title", "Tài liệu học phần")
            page = meta.get("page_number")
            printed = meta.get("printed_page")
            location = f"{title}, trang PDF {page}" if page is not None else title
            if printed:
                location += f", trang in {printed}"
            heading = meta.get("section_title", "")
            label = f"[{len(blocks) + 1}] {location}" + (f" — {heading}" if heading else "")
            if meta.get("extraction_warning"):
                label += f"\nLưu ý chất lượng nguồn: {meta['extraction_warning']}"
            block = f"{label}\n{doc}"
            if len(block) + 2 > budget:
                # Keep whole chunks: clipping can destroy code/table rows or formulas.
                continue
            blocks.append(block)
            budget -= len(block) + 2
            citations.append({"document_title": title, "page_number": page,
                              "printed_page": printed, "section_title": heading,
                              "chunk_id": meta.get("chunk_id", ""),
                              "snippet": doc[:250] + ("..." if len(doc) > 250 else "")})
        return "\n\n".join(blocks), citations
