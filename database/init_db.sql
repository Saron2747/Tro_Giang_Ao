CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. Bảng Roles (RBAC)
CREATE TABLE IF NOT EXISTS roles (
    role_id VARCHAR(32) PRIMARY KEY,
    role_name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT
);

INSERT INTO roles (role_id, role_name, description) VALUES 
('ROLE_STUDENT', 'Sinh viên', 'Quyền truy cập tài liệu công khai và đặt câu hỏi trợ giảng'),
('ROLE_LECTURER', 'Giảng viên', 'Quyền nạp tài liệu và tra cứu kho tài liệu nội bộ'),
('ROLE_ADMIN', 'Quản trị viên', 'Toàn quyền cấu hình hệ thống và kiểm tra nhật ký an ninh')
ON CONFLICT (role_id) DO NOTHING;

-- 2. Bảng Users
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role_id VARCHAR(32) NOT NULL REFERENCES roles(role_id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Bảng Knowledge Documents
CREATE TABLE IF NOT EXISTS knowledge_documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id VARCHAR(32) NOT NULL,
    uploaded_by UUID REFERENCES users(user_id),
    file_name VARCHAR(255) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    sha256_hash VARCHAR(64) NOT NULL UNIQUE,
    access_level VARCHAR(32) DEFAULT 'ROLE_STUDENT',
    parsing_status VARCHAR(32) DEFAULT 'PENDING',
    total_chunks INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Bảng Chat Sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    course_id VARCHAR(32) NOT NULL,
    session_title VARCHAR(255) DEFAULT 'Phiên hỏi đáp mới',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()   
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner_course
    ON chat_sessions (user_id, course_id, updated_at DESC);      

-- 5. Bảng Chat Messages
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seq BIGSERIAL,                                               
    session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    sender_type VARCHAR(16) NOT NULL
        CHECK (sender_type IN ('USER', 'ASSISTANT', 'SYSTEM')),  
    raw_content TEXT NOT NULL,
    filtered_content TEXT,
    retrieved_sources JSONB NOT NULL DEFAULT '[]'::jsonb,        
    latency_ms INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_seq
    ON chat_messages (session_id, seq);                         

-- 6. Bảng Security Audit Logs (OWASP LLM Compliance)
CREATE TABLE IF NOT EXISTS security_audit_logs (
    log_id BIGSERIAL PRIMARY KEY,
    user_id UUID,
    session_id UUID,
    ip_address VARCHAR(45) NOT NULL,
    event_category VARCHAR(64) NOT NULL, -- DIRECT_PROMPT_INJECTION, PII_LEAK_PREVENTED, RBAC_VIOLATION, RATE_LIMIT_EXCEEDED
    threat_score FLOAT NOT NULL,
    guardrail_action VARCHAR(32) NOT NULL, -- BLOCKED, SANITIZED, PASSED
    detected_patterns TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);