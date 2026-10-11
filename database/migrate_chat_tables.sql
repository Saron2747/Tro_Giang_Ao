-- Chạy MỘT LẦN trên database đang chạy (init_db.sql chỉ tự chạy khi volume còn trống)
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS seq BIGSERIAL;
ALTER TABLE chat_messages ALTER COLUMN retrieved_sources SET DEFAULT '[]'::jsonb;
CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner_course ON chat_sessions (user_id, course_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_seq ON chat_messages (session_id, seq);

-- Xóa 2 bảng cũ do agent_core tự tạo (MẤT lịch sử trong đó, backup trước nếu cần)
DROP TABLE IF EXISTS chat_history_messages;
DROP TABLE IF EXISTS chat_history_sessions;
