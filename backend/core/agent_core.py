"""Điểm vào của agent; điều phối truy xuất, prompt, mô hình và trí nhớ."""
import os
import re
import json
import logging
from typing import Any, Dict, Optional
import requests
import chromadb
from chromadb.config import Settings
import redis
from .conversation import ConversationMemory
from .retrieval import ContextRetrieval
from .query_intent import classify_query, topic_of, needs_history, is_source_question
from .prompts import build_messages
from .generation import generate_answer
from .security import resource_slot

logger = logging.getLogger(__name__)

class SecureTeachingAgent(ConversationMemory, ContextRetrieval):
    def __init__(
        self,
        ollama_base_url: str = None,
        chroma_host: str = None,
        chroma_port: int = None,
        redis_host: str = None,
        redis_port: int = None,
        redis_password: str = None,
        model_name: str = None,
        db_config: Optional[Dict[str, Any]] = None,
    ):
        # 1. Cấu hình dịch vụ: tham số truyền vào ưu tiên hơn biến môi trường
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_chat_url = f"{self.ollama_base_url}/api/chat"
        self.ollama_embed_url = f"{self.ollama_base_url}/api/embed"
        self.model_name = model_name or os.getenv("LLM_MODEL", "qwen2.5:3b")
        self._lexical_cache = {}
        self.embed_model_name = os.getenv("EMBED_MODEL", "nomic-embed-text")
        
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST", "127.0.0.1")
        self.chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", 8000))
        
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", 6379))
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD")  # Không để mật khẩu mặc định trong mã nguồn
        
        self.FALLBACK_RESPONSE = "Tài liệu học phần hiện tại chưa cung cấp thông tin này."
        self.COSINE_DISTANCE_THRESHOLD = float(os.getenv("COSINE_DISTANCE_THRESHOLD", "0.65"))

        # 2. Khởi tạo ChromaDB Client
        try:
            self.chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(allow_reset=False, anonymized_telemetry=False)
            )
            logger.info(f"[+] Kết nối ChromaDB tại {self.chroma_host}:{self.chroma_port} thành công.")
        except Exception as e:
            logger.error(f"[-] Không thể kết nối ChromaDB: {e}")
            self.chroma_client = None

        # 3. Khởi tạo Redis Client
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"[+] Kết nối Redis tại {self.redis_host}:{self.redis_port} thành công.")
        except Exception as e:
            logger.error(f"[-] Kết nối Redis thất bại ({e}). Kiểm tra biến môi trường REDIS_PASSWORD. "
                         "Trí nhớ hội thoại và cache trích dẫn đang TẮT.")
            self.redis_client = None

        # 4. Lịch sử hội thoại lưu bền trong PostgreSQL (bảng chat_sessions/chat_messages trong init_db.sql)
        self.db_config = db_config


    def rewrite_query_with_history(self, user_query: str, history: str) -> str:
        """Sử dụng LLM độc lập để chuẩn hóa câu hỏi nối tiếp thành câu hoàn chỉnh."""
        prompt = (
            f"Lịch sử hội thoại:\n{history}\n\n"
            f"Câu hỏi mới: \"{user_query}\"\n\n"
            "Viết lại câu hỏi mới thành MỘT câu hỏi độc lập, đầy đủ, thay các đại từ "
            "(nó, đó, này...) bằng thuật ngữ cụ thể trong lịch sử. Không thêm thông tin "
            "không có trong lịch sử. Nếu câu hỏi mới đã tự đầy đủ nghĩa, không phụ thuộc lịch sử "
            "thì giữ nguyên. Chỉ xuất ra câu hỏi."
        )
        try:
            res = requests.post(self.ollama_chat_url, json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 256},
            }, timeout=30)
            res.raise_for_status()
            out = res.json().get("message", {}).get("content", "")
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip().strip("\"'")
            if 3 < len(out) < 300:
                logger.info(f"[REWRITE] '{user_query}' -> '{out}'")
                return out
        except Exception as e:
            logger.warning(f"[-] Lỗi viết lại query: {e}")
        return user_query

    def _source_answer(self, session_id):
        citations = []
        if self.redis_client:
            try:
                citations = json.loads(self.redis_client.get(f"last_citations:{session_id}") or "[]")
            except (redis.RedisError, ValueError, TypeError):
                logger.warning("Không đọc được nguồn của phiên")
        if not citations:
            return {"answer": "Mình chưa có nguồn của câu trả lời trước. Bạn hãy hỏi nội dung cụ thể trong giáo trình trước nhé.", "citations": []}
        locations = []
        for c in citations:
            location = c["document_title"]
            if c.get("page_number") is not None:
                location += f", trang PDF {c['page_number']}"
            if c.get("printed_page"):
                location += f", trang in {c['printed_page']}"
            if location not in locations:
                locations.append(location)
        return {"answer": "Các đoạn tài liệu dùng để tham khảo cho câu trả lời trước:\n" + "\n".join(f"- {s}" for s in locations), "citations": citations}

    def _cache_citations(self, session_id, citations):
        if not self.redis_client:
            return
        try:
            key = f"last_citations:{session_id}"
            if citations:
                self.redis_client.set(key, json.dumps(citations), ex=7200)
            else:
                self.redis_client.delete(key)
        except redis.RedisError:
            logger.warning("Không cập nhật được cache nguồn")

    def execute_react_cycle(self, user_query: str, course_id: str, user_role: str, session_id: str) -> Dict[str, Any]:
        guard = getattr(self, "guardrail", None)
        if guard:
            valid, score, pattern = guard.validate_input(user_query)
            if not valid:
                guard.log_security_event(session_id=session_id, event_category="INPUT_LENGTH_EXCEEDED" if len(user_query) > guard.max_input_length else "DIRECT_PROMPT_INJECTION", threat_score=score, guardrail_action="BLOCKED", patterns=pattern)
                return {"answer": "Yêu cầu không hợp lệ hoặc có dấu hiệu điều khiển chỉ thị hệ thống.", "citations": []}
        with resource_slot() as acquired:
            if not acquired:
                guard = getattr(self, "guardrail", None)
                if guard:
                    guard.log_security_event(session_id=session_id, event_category="RESOURCE_LIMIT", guardrail_action="BLOCKED", patterns="Hết lượt xử lý đồng thời")
                return {"answer": "Hệ thống đang bận. Vui lòng thử lại sau.", "citations": []}
            result = self._execute_cycle(user_query, course_id, user_role, session_id)
            guard = getattr(self, "guardrail", None)
            if guard:
                result["answer"] = guard.protect_output(result["answer"], session_id=session_id)
                cleaned = guard.sanitize_citations(result.get("citations", []))
                if cleaned != result.get("citations", []):
                    guard.log_security_event(session_id=session_id, event_category="PII_CITATION_REDACTED", guardrail_action="SANITIZED", patterns="Đã lọc nội dung nguồn tham khảo")
                result["citations"] = cleaned
            return result

    def _execute_cycle(self, user_query: str, course_id: str, user_role: str, session_id: str) -> Dict[str, Any]:
        """Điểm gọi tương thích với frontend; các bước xử lý nằm trong module riêng."""
        if is_source_question(user_query):
            return self._source_answer(session_id)
        history = self.get_session_history(session_id, turns=3)
        # Keep the most recent context within the input budget.
        history = history[-3500:]
        guard = getattr(self, "guardrail", None)
        if guard:
            user_query = guard.redact_pii(user_query)
            history = guard.redact_pii(history)
            if guard.scan_actionable_text(history):
                guard.log_security_event(session_id=session_id, event_category="HISTORY_PROMPT_INJECTION", guardrail_action="BLOCKED", patterns="Bỏ lịch sử chứa mẫu chỉ dẫn đáng ngờ")
                history = ""
        task = classify_query(user_query)
        followup = bool(history) and needs_history(user_query)
        if history and re.fullmatch(r"\s*[abcd][.!]?\s*", user_query, re.I):
            task = "evaluate"
        search_query = topic_of(user_query) if task == "generate" else user_query
        if followup:
            rewritten = self.rewrite_query_with_history(user_query, history)
            rewritten_task = classify_query(rewritten)
            if task == "explain" and rewritten_task != "explain":
                task = rewritten_task
            search_query = topic_of(rewritten) if task == "generate" else rewritten
            # If the model could not rewrite a short reference, retain the last topic.
            if rewritten == user_query:
                previous = re.findall(r"^Sinh viên: (.*)$", history, re.M)
                if previous:
                    if task == "explain" and classify_query(previous[-1]) == "generate" and re.search(r"khó hơn|dễ hơn|thêm|tiếp", user_query, re.I):
                        task = "generate"
                    search_query = f"{topic_of(previous[-1])} {search_query}".strip()
        if not search_query.strip():
            return {"answer": "Bạn muốn hỏi hoặc luyện tập chủ đề nào trong giáo trình?", "citations": []}
        effective_history = history if followup else ""
        window = int(os.getenv("LLM_NUM_CTX", "8192"))
        output = int(os.getenv("LLM_NUM_PREDICT", "2048"))
        # Conservative Vietnamese character estimate; reserve system prompt and output.
        context_budget = max(0, (window - output) * 2 - len(user_query) - len(effective_history) - 2200)
        context, citations = self.retrieve_safe_context(
            search_query, course_id, user_role, top_k=8 if task in ("generate", "compare", "summarize") else 5,
            context_budget=context_budget,
        )
        if not context:
            self._cache_citations(session_id, [])
            return {"answer": self.FALLBACK_RESPONSE, "citations": []}
        messages = build_messages(user_query, course_id, context, effective_history, task, self.FALLBACK_RESPONSE)
        answer, ok = generate_answer(self.ollama_chat_url, self.model_name, messages, task)
        if guard:
            protected = guard.protect_output(answer, system_prompt=messages[0]["content"], session_id=session_id)
            if protected != answer and "chỉ thị hệ thống" in protected:
                ok = False
                citations = []
            answer = protected
            citations = guard.sanitize_citations(citations)
        if self.FALLBACK_RESPONSE in answer:
            citations = []
        if not ok:
            citations = []
        elif self._is_memory_worthy(answer):
            # Store the same sanitized answer that the frontend displays.
            sanitizer = getattr(self, "output_sanitizer", None)
            if sanitizer:
                answer = sanitizer(answer)
            self.append_session_history(session_id, user_query, answer)
        self._cache_citations(session_id, citations)
        return {"answer": answer, "citations": citations}


# Giữ nguyên API import của frontend và tên cũ.
AgentCore = SecureTeachingAgent

