import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg2
import redis

logger = logging.getLogger("SecurityGuardrail")

# ---------------------------------------------------------
# Biểu thức chính quy dùng chung (biên dịch một lần)
# ---------------------------------------------------------
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")

# Output: chỉ gỡ thẻ nguy hiểm thật sự, không xóa nhầm "a < b" hay "vector<int>"
_SCRIPT_BLOCK_RE = re.compile(r"<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
_DANGEROUS_TAG_RE = re.compile(
    r"<\s*/?\s*(?:script|style|iframe|object|embed|link|meta|form|svg|math)\b[^>\n]*>",
    re.IGNORECASE,
)
_DANGEROUS_ATTR_TAG_RE = re.compile(
    r"<[^>\n]*(?:\bon\w+\s*=|(?:href|src)\s*=\s*[\"']?\s*javascript\s*:)[^>\n]*>",
    re.IGNORECASE,
)
_MD_JS_LINK_RE = re.compile(r"\]\(\s*javascript\s*:[^)]*\)", re.IGNORECASE)

# PII
_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+84|0)[\s.-]?[35789](?:[\s.-]?\d){8}(?!\d)")
_STUDENT_ID_RE = re.compile(r"\bB\d{2}[A-Za-z]{4}\d{3}\b", re.IGNORECASE)


class AISecurityGuardrail:
    # Mẫu phát hiện Direct Prompt Injection & Jailbreak (OWASP LLM01).
    # Văn bản được chuẩn hóa (NFKC, chữ thường, gộp khoảng trắng) trước khi quét.
    # Các mẫu nhắm vào HÀNH VI ra lệnh, không chặn việc chỉ nhắc đến thuật ngữ,
    # vì sinh viên môn ATTT có quyền hỏi "prompt injection là gì?".
    INJECTION_PATTERNS = [
        # --- Ghi đè chỉ thị (tiếng Anh) ---
        r"(ignore|disregard)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
        r"forget\s+(all\s+|everything\s+|your\s+)(previous\s+|prior\s+)?(instructions?|rules?)",
        # --- Ghi đè chỉ thị (tiếng Việt) ---
        r"(bỏ\s+qua|phớt\s+lờ|không\s+cần\s+tuân\s+theo)\s+(toàn\s+bộ|tất\s+cả|hết|mọi)\s+(các\s+)?(hướng\s+dẫn|chỉ\s+thị|chỉ\s+dẫn|quy\s+tắc)",
        r"(bỏ\s+qua|phớt\s+lờ)\s+(các\s+)?(hướng\s+dẫn|chỉ\s+thị|chỉ\s+dẫn|quy\s+tắc)\s+(trước\s+đó|phía\s+trên|ở\s+trên|ban\s+đầu|của\s+hệ\s+thống)",
        # --- Đổi vai / chế độ jailbreak ---
        r"you\s+are\s+now\s+(in\s+)?(developer|debug|god|dan)\s+mode",
        r"đóng\s+vai\s+(một\s+)?(hacker|dan|quản\s+trị(\s+viên)?|admin)",
        # --- Đòi lộ system prompt ---
        r"reveal\s+(the\s+|your\s+)?(secret|prompt|instructions?)",
        r"(reveal|show|print|display|repeat|output|leak|tiết\s+lộ|hiển\s+thị|in\s+ra|xuất\s+ra|đọc\s+lại|lặp\s+lại"
        r"|cho\s+(tôi\s+|mình\s+|em\s+)?xem).{0,40}"
        r"(system\s*prompt|system\s*instructions?|prompt\s+hệ\s+thống|chỉ\s+thị\s+hệ\s+thống"
        r"|câu\s+lệnh\s+hệ\s+thống|hướng\s+dẫn\s+hệ\s+thống)",
    ]

    LEETSPEAK_MAP = {
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s",
    }

    # Dấu vết của system prompt trong agent_core; phân biệt hoa/thường.
    # Nên đặt thêm một mã canary ngẫu nhiên vào system prompt rồi thêm vào extra_leak_markers.
    LEAKAGE_MARKERS = (
        "[CHỈ THỊ HỆ THỐNG]",
        "QUY TẮC CỐT LÕI:",
        "QUY TẮC HOẠT ĐỘNG BẮT BUỘC",
        "NGUYÊN TẮC HOẠT ĐỘNG",
        "XỬ LÝ NHIỆM VỤ DỰA TRÊN KHỐI",
        "AN TOÀN HỆ THỐNG",
        "Bạn là Trợ giảng Ảo",
        "<context>",
        "</context>",
    )

    def __init__(
        self,
        db_config: Dict[str, Any],
        redis_client: redis.Redis,
        max_input_length: int = 4000,
        rate_limit_fail_open: bool = True,
        extra_leak_markers: Iterable[str] = (),
    ):
        self.db_config = db_config
        self.redis = redis_client
        self.max_input_length = max_input_length
        self.rate_limit_fail_open = rate_limit_fail_open
        self._leak_markers = tuple(self.LEAKAGE_MARKERS) + tuple(extra_leak_markers)
        self._leet_table = str.maketrans(self.LEETSPEAK_MAP)
        self._injection_res = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    # ---------------------------------------------------------
    # Chuẩn hóa văn bản trước khi quét
    # ---------------------------------------------------------
    @staticmethod
    def _canonicalize(text: str) -> str:
        """NFKC (gộp dấu tiếng Việt, ký tự full-width), bỏ ký tự vô hình, chữ thường, gộp khoảng trắng."""
        t = unicodedata.normalize("NFKC", text or "")
        t = _ZERO_WIDTH_RE.sub("", t)
        t = _CONTROL_RE.sub("", t)
        return _SPACE_RE.sub(" ", t.lower()).strip()

    def normalize_leetspeak(self, text: str) -> str:
        """Giải mã ký tự thay thế (leetspeak) để đưa về văn bản gốc trước khi quét."""
        return self._canonicalize(text).translate(self._leet_table)

    def _candidates(self, text: str) -> List[str]:
        canon = self._canonicalize(text)
        leet = canon.translate(self._leet_table)
        return [canon] if leet == canon else [canon, leet]

    # ---------------------------------------------------------
    # Rate limit
    # ---------------------------------------------------------
    def rate_limit_check(self, user_id: str, limit: int = 15, period: int = 60) -> bool:
        """Giới hạn cửa sổ cố định trên Redis: tối đa `limit` yêu cầu mỗi `period` giây.

        Nếu khóa tồn tại mà không có TTL (ví dụ tiến trình chết giữa INCR và EXPIRE)
        thì gán lại TTL, tránh khóa người dùng vĩnh viễn.
        """
        key = f"ratelimit:{user_id}"
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            count, ttl = pipe.execute()
            if ttl == -1:
                self.redis.expire(key, period)
            return int(count) <= limit
        except redis.RedisError as e:
            logger.error(f"[-] Lỗi Redis khi kiểm tra rate limit: {e}")
            return self.rate_limit_fail_open

    # ---------------------------------------------------------
    # Input guardrail
    # ---------------------------------------------------------
    def validate_input(self, text: str) -> Tuple[bool, float, Optional[str]]:
        """Trả về (hợp lệ, điểm đe dọa, mẫu khớp)."""
        if text and len(text) > self.max_input_length:
            return False, 0.5, f"Độ dài vượt giới hạn {self.max_input_length} ký tự"

        for candidate in self._candidates(text):
            for rx in self._injection_res:
                if rx.search(candidate):
                    return False, 0.95, f"Khớp mẫu độc hại: {rx.pattern}"
        return True, 0.0, None

    def scan_untrusted_text(self, text: str) -> List[str]:
        """Quét văn bản không tin cậy (nội dung tài liệu, đoạn truy xuất) tìm mẫu injection gián tiếp.

        Chỉ trả về danh sách mẫu khớp để nơi gọi quyết định cảnh báo hay chặn. Không nên chặn tự động,
        vì giáo trình môn ATTT có thể chứa các câu ví dụ tấn công hợp lệ.
        """
        hits = []
        for candidate in self._candidates(text):
            for rx in self._injection_res:
                if rx.search(candidate) and rx.pattern not in hits:
                    hits.append(rx.pattern)
        return hits

    # ---------------------------------------------------------
    # Output guardrail
    # ---------------------------------------------------------
    def sanitize_output(self, text: str) -> str:
        """Che chỉ thị hệ thống, gỡ mã web nguy hiểm, che thông tin cá nhân (PII)."""
        if not text:
            return text or ""
        clean = text

        # 1) Rò rỉ system prompt: làm trước khi gỡ thẻ để bắt được <context>
        for marker in self._leak_markers:
            clean = clean.replace(marker, "[REDACTED_SYSTEM_DATA]")

        # 2) Mã web nguy hiểm (giữ nguyên các phép so sánh như "a < b")
        clean = _SCRIPT_BLOCK_RE.sub("", clean)
        clean = _DANGEROUS_TAG_RE.sub("", clean)
        clean = _DANGEROUS_ATTR_TAG_RE.sub("", clean)
        clean = _MD_JS_LINK_RE.sub("](#)", clean)

        # 3) PII: email trước để không bị regex mã sinh viên cắt dở
        clean = _EMAIL_RE.sub("[REDACTED_EMAIL]", clean)
        clean = _PHONE_RE.sub("[REDACTED_PHONE]", clean)
        clean = _STUDENT_ID_RE.sub("[REDACTED_STUDENT_ID]", clean)
        return clean

    # ---------------------------------------------------------
    # Nhật ký kiểm toán
    # ---------------------------------------------------------
    def log_security_event(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: str = "127.0.0.1",
        event_category: str = "UNKNOWN",
        threat_score: float = 0.0,
        guardrail_action: str = "BLOCKED",
        patterns: Optional[str] = None,
        **legacy: Any,
    ) -> bool:
        """Ghi sự cố vào PostgreSQL. Trả về True nếu ghi thành công.

        Tên tham số khớp với app.py. Vẫn chấp nhận tên cũ (event_cat, score, action).
        """
        event_category = legacy.pop("event_cat", event_category)
        threat_score = legacy.pop("score", threat_score)
        guardrail_action = legacy.pop("action", guardrail_action)
        if legacy:
            raise TypeError(f"Tham số không hợp lệ: {sorted(legacy)}")

        cfg = {"connect_timeout": 3, **self.db_config}
        conn = None
        try:
            conn = psycopg2.connect(**cfg)
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO security_audit_logs
                            (user_id, session_id, ip_address, event_category,
                             threat_score, guardrail_action, detected_patterns)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            user_id,
                            session_id,
                            ip_address,
                            event_category,
                            threat_score,
                            guardrail_action,
                            (patterns or "")[:500] or None,
                        ),
                    )
            return True
        except Exception as e:
            logger.error(f"[-] Lỗi ghi Security Audit Log: {e}")
            return False
        finally:
            if conn is not None:
                conn.close()