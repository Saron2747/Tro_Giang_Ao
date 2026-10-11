"""Trí nhớ ngắn hạn Redis và lịch sử PostgreSQL của trợ giảng."""
import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

class ConversationMemory:
    def get_session_history(self, session_id: str, turns: int = 2) -> str:
        """Lấy đúng số lượt hội thoại (turns * 2 phần tử)."""
        if not self.redis_client:
            return ""
        try:
            lines = self.redis_client.lrange(f"chat_history:{session_id}", -turns * 2, -1)
            return "\n".join(lines) if lines else ""
        except Exception as e:
            logger.warning(f"[-] Lỗi đọc Redis history: {e}")
            return ""


    def append_session_history(self, session_id: str, user_msg: str, assistant_msg: str):
        """Lưu lịch sử sạch: cắt ngắn câu trả lời quá dài để tiết kiệm context."""
        if not self.redis_client:
            return
        try:
            key = f"chat_history:{session_id}"
            self.redis_client.rpush(key, f"Sinh viên: {user_msg}")
            # Cắt bớt câu trả lời quá dài (như đề thi 5 câu) tránh tràn context ở lượt sau
            truncated_reply = assistant_msg[:1800] + ("..." if len(assistant_msg) > 1800 else "")
            self.redis_client.rpush(key, f"Trợ giảng: {truncated_reply}")
            self.redis_client.ltrim(key, -10, -1)
            self.redis_client.expire(key, 86400)
        except Exception as e:
            logger.warning(f"[-] Lỗi ghi Redis history: {e}")


    DEFAULT_TITLE = "Đoạn chat mới"


    _NON_MEMORY_PREFIXES = (
        "🚫", "⚠️", "Lỗi", "Quá thời gian",
        "Nội dung trên được trích dẫn", "Mình chưa có nguồn", "Bạn muốn tôi ra câu hỏi",
    )


    @contextmanager
    def _cursor(self):
        """Mở kết nối ngắn hạn, tự commit khi thành công, rollback khi lỗi, luôn đóng."""
        if not self.db_config:
            raise RuntimeError("Chưa truyền db_config cho agent nên không lưu được lịch sử hội thoại.")
        conn = psycopg2.connect(**{"connect_timeout": 3, **self.db_config})
        try:
            with conn:
                with conn.cursor() as cur:
                    yield cur
        finally:
            conn.close()


    @staticmethod
    def _norm_course(course_id: str) -> str:
        return (course_id or "").strip().upper()


    def save_message_turn(
        self,
        session_id: str,
        user_msg: str,
        bot_msg: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        course_id: str = "",
        owner_id: Optional[str] = None,
    ) -> None:
        """Lưu một lượt hỏi - đáp vào chat_sessions/chat_messages. Tự tạo phiên ở tin nhắn đầu tiên
        (tiêu đề = câu hỏi đầu, tối đa 50 ký tự).

        owner_id là user_id (UUID) trong bảng users. Từ chối ghi nếu phiên đã thuộc về người dùng khác.
        """
        if not owner_id:
            raise ValueError("Thiếu owner_id (user_id) nên không lưu được lịch sử hội thoại.")
        owner_id = str(owner_id)
        title = " ".join((user_msg or "").split())[:50] or self.DEFAULT_TITLE
        with self._cursor() as cur:
            cur.execute("SELECT user_id::text FROM chat_sessions WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO chat_sessions (session_id, user_id, course_id, session_title) "
                    "VALUES (%s, %s, %s, %s)",
                    (session_id, owner_id, self._norm_course(course_id), title),
                )
            elif row[0] != owner_id:
                raise PermissionError("Phiên hội thoại này thuộc về người dùng khác.")
            else:
                cur.execute("UPDATE chat_sessions SET updated_at = now() WHERE session_id = %s", (session_id,))

            cur.execute(
                "INSERT INTO chat_messages (session_id, sender_type, raw_content) VALUES (%s, 'USER', %s)",
                (session_id, (user_msg or "")[:20000]),
            )
            bot = (bot_msg or "")[:20000]
            cur.execute(
                "INSERT INTO chat_messages (session_id, sender_type, raw_content, filtered_content, retrieved_sources) "
                "VALUES (%s, 'ASSISTANT', %s, %s, %s)",
                (session_id, bot, bot, Json(citations or [])),
            )


    def list_sessions(self, course_id: str, owner_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Danh sách phiên của owner trong môn học, mới nhất trước. Bỏ qua phiên chưa có tin nhắn."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT s.session_id::text, s.session_title, s.updated_at FROM chat_sessions s "
                "WHERE s.user_id = %s AND s.course_id = %s "
                "AND EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.session_id) "
                "ORDER BY s.updated_at DESC LIMIT %s",
                (str(owner_id), self._norm_course(course_id), limit),
            )
            return [{"session_id": r[0], "title": r[1], "updated_at": r[2]} for r in cur.fetchall()]


    def load_messages(self, session_id: str, owner_id: str, limit: int = 400) -> List[Dict[str, Any]]:
        """Đọc tin nhắn của một phiên theo thứ tự thời gian. Phiên của người khác trả về danh sách rỗng."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT LOWER(m.sender_type), COALESCE(m.filtered_content, m.raw_content), m.retrieved_sources "
                "FROM chat_messages m JOIN chat_sessions s ON s.session_id = m.session_id "
                "WHERE m.session_id = %s AND s.user_id = %s ORDER BY m.seq ASC LIMIT %s",
                (session_id, str(owner_id), limit),
            )
            out = []
            for role, content, cits in cur.fetchall():
                if isinstance(cits, str):
                    try:
                        cits = json.loads(cits)
                    except json.JSONDecodeError:
                        cits = []
                out.append({"role": role, "content": content, "citations": cits or []})
            return out


    def _is_memory_worthy(self, answer: str) -> bool:
        return not answer.startswith(self._NON_MEMORY_PREFIXES) and self.FALLBACK_RESPONSE not in answer


    def seed_session_cache(self, session_id: str, messages: List[Dict[str, Any]], max_turns: int = 5) -> None:
        """Nạp lại trí nhớ Redis từ lịch sử đã lưu khi người dùng mở lại hội thoại cũ.

        Redis chỉ giữ 24 giờ, nên nếu không nạp lại thì giao diện hiện cả cuộc trò chuyện
        nhưng mô hình đã quên. Chỉ nạp các lượt trả lời bình thường.
        """
        if not self.redis_client:
            return
        pairs, i = [], 0
        while i + 1 < len(messages):
            u, a = messages[i], messages[i + 1]
            if u.get("role") == "user" and a.get("role") == "assistant":
                if self._is_memory_worthy(a.get("content", "")):
                    pairs.append((u.get("content", ""), a.get("content", "")))
                i += 2
            else:
                i += 1
        key = f"chat_history:{session_id}"
        try:
            self.redis_client.delete(key)
            for u, a in pairs[-max_turns:]:
                self.redis_client.rpush(key, f"Sinh viên: {u}")
                self.redis_client.rpush(key, f"Trợ giảng: {a[:1800]}" + ("..." if len(a) > 1800 else ""))
            if pairs:
                self.redis_client.expire(key, 86400)
        except Exception as e:
            logger.warning(f"[-] Lỗi nạp lại trí nhớ Redis: {e}")

