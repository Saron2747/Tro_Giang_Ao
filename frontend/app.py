import html
import os
import sys
import uuid
import re

import bcrypt
import pandas as pd
import psycopg2
import redis
import streamlit as st

# Đồng bộ đường dẫn module backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.core.agent_core import SecureTeachingAgent
from backend.core.etl_pipeline import DocumentETLPipeline
from backend.core.security import AISecurityGuardrail
from backend.core.config import database_config, load_environment, redis_config

load_environment()

st.set_page_config(
    page_title="Trợ giảng Ảo An Toàn",
    page_icon="🛡️️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cấu hình CSDL PostgreSQL
DB_CONFIG = database_config()

PAGE_CHAT = "Hỏi đáp"
PAGE_UPLOAD = "Nạp tài liệu"
PAGE_AUDIT = "Giám sát"

# CSS chung
BASE_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.16);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1 {
        font-size: 1.15rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    .brand-kicker {
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.62;
        margin: 0 0 0.2rem 0;
    }
    .session-pill {
        display: inline-block;
        font-size: 0.78rem;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: color-mix(in srgb, var(--primary-color) 14%, transparent);
        color: var(--text-color);
        margin-right: 0.35rem;
    }

    /* Bong bóng chat */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.15rem 0;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: color-mix(in srgb, var(--primary-color) 16%, var(--secondary-background-color));
        border-radius: 18px 18px 6px 18px;
        padding: 0.72rem 0.95rem;
        max-width: min(78%, 640px);
        margin-left: auto;
    }
    [data-testid="stChatMessageAvatarAssistant"] + [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 18px 18px 18px 6px;
        padding: 0.8rem 1rem;
        max-width: min(92%, 760px);
    }

    .empty-hero {
        text-align: center;
        padding: 12vh 1rem 2rem 1rem;
    }
    .empty-hero h2 {
        font-weight: 600;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }
    .empty-hero p {
        opacity: 0.7;
        margin: 0;
    }

    /* Ô chat ghim đáy */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background: color-mix(in srgb, var(--background-color) 92%, transparent);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(128, 128, 128, 0.14);
    }
    [data-testid="stBottom"] [data-testid="stChatInput"],
    [data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] {
        max-width: 860px;
        width: 100%;
        margin: 0 auto;
        padding-bottom: 0.35rem;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 22px !important;
    }
</style>
"""

CHAT_LAYOUT_CSS = """
<style>
    section.main > div.block-container {
        max-width: 860px;
        padding-top: 1.25rem;
        padding-bottom: 7.5rem;
    }
</style>
"""

WIDE_LAYOUT_CSS = """
<style>
    section.main > div.block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
</style>
"""


# --- HÀM XÁC THỰC NGƯỜI DÙNG TỪ POSTGRESQL ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def authenticate_user(username: str, password: str):
    """Kiểm tra tài khoản và mật khẩu băm trong CSDL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, password_hash, role_id, is_active FROM users WHERE username = %s;",
            (username.strip(),),
        )
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user:
            user_id, hashed_pw, role_id, is_active = user
            if not is_active:
                return False, "Tài khoản của bạn đã bị khóa.", None, None
            if verify_password(password, hashed_pw):
                return True, "Đăng nhập thành công!", str(user_id), role_id
        return False, "Tên đăng nhập hoặc mật khẩu không chính xác.", None, None
    except Exception as exc:
        return False, f"Lỗi kết nối cơ sở dữ liệu: {exc}", None, None


@st.cache_resource
def init_system():
    """Khởi tạo tài nguyên hệ thống và đưa vào bộ đệm RAM của Streamlit."""
    redis_settings = redis_config()
    r = redis.Redis(**redis_settings, decode_responses=True)
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    agent = SecureTeachingAgent(
        ollama_base_url=ollama_url,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        redis_host=redis_settings["host"],
        redis_port=redis_settings["port"],
        redis_password=redis_settings["password"],
        model_name=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        db_config=DB_CONFIG,
    )
    guardrail = AISecurityGuardrail(db_config=DB_CONFIG, redis_client=r)
    pipeline = DocumentETLPipeline(
        ollama_base_url=ollama_url,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
    )
    return agent, guardrail, pipeline, r


agent, guardrail, pipeline, redis_client = init_system()

# Quản lý trạng thái phiên
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.user_role = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


SESSION_SELECT_KEY = "session_selector_widget"


def _persist_turn(user_input: str, answer: str, citations: list, course_id: str) -> None:
    """Lưu lượt hỏi-đáp vào PostgreSQL gắn với tài khoản người dùng."""
    try:
        agent.save_message_turn(
            session_id=st.session_state.session_id,
            user_msg=user_input,
            bot_msg=answer,
            citations=citations,
            course_id=course_id,
            owner_id=st.session_state.user_id,
        )
    except Exception as exc:
        st.session_state.history_error = f"Không lưu được lịch sử hội thoại: {exc}"


def handle_user_turn(user_input: str, course_id: str, role_selection: str) -> None:
    """Chuỗi chốt chặn an ninh: Rate limit -> Input Guardrail -> Agent -> Output Guardrail."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    citations: list = []

    # Kiểm tra Rate Limit theo user_id thực tế
    if not guardrail.rate_limit_check(user_id=st.session_state.user_id):
        guardrail.log_security_event(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            ip_address="127.0.0.1",
            event_category="RATE_LIMIT_EXCEEDED",
            threat_score=0.85,
            guardrail_action="BLOCKED",
            patterns="Vượt ngưỡng 15 requests/phút",
        )
        answer = "⚠️ Cảnh báo: Tần suất gửi câu hỏi quá nhanh. Vui lòng chờ 1 phút trước khi thử lại."
    else:
        is_valid, threat_score, pattern = guardrail.validate_input(user_input)
        if not is_valid:
            guardrail.log_security_event(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                ip_address="127.0.0.1",
                event_category="DIRECT_PROMPT_INJECTION",
                threat_score=threat_score,
                guardrail_action="BLOCKED",
                patterns=pattern,
            )
            answer = "🚫 Yêu cầu bị từ chối: Vi phạm chính sách an toàn thông tin (Prompt Injection)."
        else:
            with st.spinner("Trợ giảng đang tra cứu giáo trình và tổng hợp câu trả lời..."):
                result = agent.execute_react_cycle(
                    user_query=user_input,
                    course_id=course_id,
                    user_role=role_selection,
                    session_id=st.session_state.session_id,
                )
            answer = guardrail.sanitize_output(result["answer"])
            citations = result.get("citations", [])

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
    _persist_turn(user_input, answer, citations, course_id)


def _switch_session(sid: str) -> None:
    st.session_state.session_id = sid
    try:
        msgs = agent.load_messages(sid, owner_id=st.session_state.user_id)
    except Exception as exc:
        msgs = []
        st.session_state.history_error = f"Không tải được hội thoại: {exc}"
    st.session_state.messages = msgs
    try:
        agent.seed_session_cache(sid, msgs)
    except Exception:
        pass


def _on_pick_session() -> None:
    _switch_session(st.session_state[SESSION_SELECT_KEY])


def _on_new_chat() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def _render_citations(citations) -> None:
    if not citations:
        return
    with st.expander(f"📎 Nguồn tham khảo ({len(citations)})"):
        for c in citations:
            page = c.get("page_number")
            where = f"trang {page}" if page is not None else "tài liệu không phân trang"
            st.markdown(f"**{c.get('document_title', 'Tài liệu')}** · {where}")
            if c.get("snippet"):
                st.caption(c["snippet"])


def render_sidebar() -> tuple[str, str, str]:
    st.sidebar.markdown('<p class="brand-kicker">Secure Teaching Agent</p>', unsafe_allow_html=True)
    st.sidebar.title("Trợ giảng ảo")

    # Hiển thị thông tin định danh và vai trò cố định
    role_labels = {
        "ROLE_ADMIN": "Quản trị viên (Admin)",
        "ROLE_LECTURER": "Giảng viên (Lecturer)",
        "ROLE_STUDENT": "Sinh viên (Student)",
    }
    user_role = st.session_state.user_role
    role_display = role_labels.get(user_role, user_role)

    st.sidebar.markdown(f"👤 **Tài khoản:** `{st.session_state.username}`")
    st.sidebar.markdown(f"🏷️ **Vai trò:** `{role_display}`")

    if st.sidebar.button("🚪 Đăng xuất", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.session_state.user_role = None
        st.session_state.messages = []
        st.rerun()

    st.sidebar.divider()

    # Điều hướng phân hệ
    pages = [PAGE_CHAT]
    if user_role in ["ROLE_LECTURER", "ROLE_ADMIN"]:
        pages.append(PAGE_UPLOAD)
    if user_role == "ROLE_ADMIN":
        pages.append(PAGE_AUDIT)

    page = st.sidebar.radio(
        "Phân hệ",
        pages,
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    course_id = st.sidebar.text_input("Mã môn học", "ATTT_101").strip() or "ATTT_101"

    st.sidebar.divider()

    # --- LỊCH SỬ TRÒ CHUYỆN (POSTGRESQL) ---
    st.sidebar.markdown("### 💬 Lịch sử trò chuyện")

    history_error = st.session_state.pop("history_error", None)
    if history_error:
        st.sidebar.warning(history_error)

    st.sidebar.button("➕ Đoạn chat mới", use_container_width=True, on_click=_on_new_chat)

    try:
        sessions = agent.list_sessions(course_id=course_id, owner_id=st.session_state.user_id)
    except Exception as exc:
        sessions = []
        st.sidebar.warning(f"Không tải được danh sách hội thoại: {exc}")

    current_sid = st.session_state.session_id
    session_dict = {s["session_id"]: s["title"] for s in sessions}
    if current_sid not in session_dict:
        session_dict = {current_sid: "Đoạn chat mới", **session_dict}

    st.session_state[SESSION_SELECT_KEY] = current_sid
    st.sidebar.selectbox(
        "Chọn cuộc hội thoại",
        options=list(session_dict),
        format_func=lambda sid: session_dict.get(sid, "Cuộc trò chuyện"),
        key=SESSION_SELECT_KEY,
        on_change=_on_pick_session,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Phiên `{st.session_state.session_id[:8]}`")
    return page, user_role, course_id


def render_chat(course_id: str, role_selection: str) -> None:
    st.markdown(
        f'<span class="session-pill">{_esc(course_id)}</span>'
        f'<span class="session-pill">{_esc(role_selection)}</span>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-hero">
                <h2>Hỏi bất cứ điều gì trong giáo trình</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🛡️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            _render_citations(msg.get("citations"))

    if hasattr(st, "bottom"):
        user_input = st.bottom.chat_input("Nhập thắc mắc môn học của bạn...")
    else:
        user_input = st.chat_input("Nhập thắc mắc môn học của bạn...")

    if user_input:
        handle_user_turn(user_input, course_id, role_selection)
        st.rerun()


def render_upload(course_id: str, role_selection: str) -> None:
    st.subheader("Quản lý & Nạp giáo trình học phần")
    st.caption(f"Học phần `{course_id}` · Quyền thao tác: `{role_selection}`")

    # Kiểm tra phân quyền cứng (RBAC)
    if role_selection not in ["ROLE_LECTURER", "ROLE_ADMIN"]:
        st.error("⛔ Truy cập bị từ chối: Tài khoản Sinh viên không có thẩm quyền tải hoặc xóa tài liệu.")
        return

    # --- KHU VỰC 1: BẢNG DANH MỤC TÀI LIỆU HIỆN HÀNH ---
    st.markdown("#### 📚 Danh mục tài liệu trong môn học")
    current_docs = pipeline.list_documents(course_id)

    if not current_docs:
        st.info("Hiện tại chưa có tài liệu nào được lập chỉ mục cho môn học này.")
    else:
        for doc in current_docs:
            col1, col2, col3 = st.columns([5, 2, 2])
            with col1:
                st.write(f"📄 **{doc['document_title']}**")
                st.caption(f"ID: `{doc['document_id']}` | Phân quyền: `{doc['allowed_roles']}`")
            with col2:
                st.write(f"🧩 `{doc['chunks_count']} phân mảnh`")
            with col3:
                if st.button("🗑️️ Xóa file", key=f"del_{doc['document_id']}", type="secondary"):
                    success = pipeline.delete_document(course_id, doc["document_id"])
                    physical_file = os.path.join("knowledge_storage", course_id, doc["document_title"])
                    if os.path.exists(physical_file):
                        try:
                            os.remove(physical_file)
                        except Exception:
                            pass

                    if success:
                        st.success(f"Đã dọn sạch phân mảnh của `{doc['document_title']}`!")
                        st.rerun()
                    else:
                        st.error("Không thể xóa tài liệu này.")
            st.divider()

    # --- KHU VỰC 2: TẢI TÀI LIỆU MỚI HOẶC CẬP NHẬT ---
    st.markdown("#### 📤 Nạp thêm hoặc Cập nhật tài liệu")
    uploaded_files = st.file_uploader(
        "Chọn tệp giáo trình học phần (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    target_roles = st.multiselect(
        "Cấp quyền truy cập tài liệu này cho các nhóm vai trò",
        options=["ROLE_STUDENT", "ROLE_LECTURER", "ROLE_ADMIN"],
        default=["ROLE_STUDENT", "ROLE_LECTURER", "ROLE_ADMIN"],
        help="Cần chọn ROLE_STUDENT để sinh viên có thể tra cứu khi đặt câu hỏi.",
    )

    if st.button("Bắt đầu xử lý và vector hóa", type="primary"):
        if not uploaded_files:
            st.error("Vui lòng đính kèm ít nhất một tệp tài liệu trước khi bấm nạp.")
            return

        storage_dir = os.path.join("knowledge_storage", course_id)
        os.makedirs(storage_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(storage_dir, uploaded_file.name)
            with open(file_path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())

            with st.spinner(f"Đang bóc tách, băm và vector hóa: {uploaded_file.name}..."):
                try:
                    doc_id = uploaded_file.name
                    total_chunks = pipeline.process_and_index_document(
                        file_path=file_path,
                        course_id=course_id,
                        document_id=doc_id,
                        allowed_roles=target_roles,
                    )
                    st.success(
                        f"Hoàn tất `{uploaded_file.name}`: đã tạo/cập nhật {total_chunks} phân mảnh trong ChromaDB."
                    )
                except Exception as exc:
                    st.error(f"Xử lý tệp {uploaded_file.name} thất bại: {exc}")
        st.rerun()


def render_dashboard(role_selection: str) -> None:
    st.subheader("Nhật ký sự cố an toàn thông tin")
    st.caption("Audit trail đọc trực tiếp từ security_audit_logs, 100 bản ghi mới nhất.")

    # Kiểm tra phân quyền cứng (RBAC)
    if role_selection != "ROLE_ADMIN":
        st.error("⛔ Truy cập bị từ chối: Chỉ Quản trị viên (ROLE_ADMIN) mới có thẩm quyền xem nhật ký an ninh.")
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql_query(
            "SELECT log_id, user_id, ip_address, event_category, threat_score, "
            "guardrail_action, detected_patterns, timestamp "
            "FROM security_audit_logs ORDER BY timestamp DESC LIMIT 100;",
            conn,
        )
        conn.close()
    except Exception as exc:
        st.error(f"Lỗi kết nối cơ sở dữ liệu kiểm toán: {exc}")
        return

    if df.empty:
        st.info("Hệ thống an toàn. Chưa phát hiện sự cố an ninh nào trong CSDL.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Sự kiện an ninh", len(df))
    c2.metric(
        "Prompt injection bị chặn",
        len(df[df["event_category"] == "DIRECT_PROMPT_INJECTION"]),
    )
    c3.metric(
        "Vi phạm rate limit",
        len(df[df["event_category"] == "RATE_LIMIT_EXCEEDED"]),
    )
    st.dataframe(df, use_container_width=True)


def render_login():
    """Giao diện màn hình đăng nhập."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 4rem; margin-bottom: 2rem;">
            <h2>🛡️ Hệ thống Trợ giảng Ảo An Toàn</h2>
            <p style="opacity: 0.7;">Đăng nhập để xác thực danh tính và phân quyền truy cập học phần</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập / Mã định danh:")
            password = st.text_input("Mật khẩu:", type="password")
            submitted = st.form_submit_button("Đăng nhập", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.warning("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")
                else:
                    success, msg, u_id, role = authenticate_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username.strip()
                        st.session_state.user_id = u_id
                        st.session_state.user_role = role
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


def main() -> None:
    st.markdown(BASE_CSS, unsafe_allow_html=True)

    # Nếu chưa đăng nhập: chặn lại ở màn hình xác thực
    if not st.session_state.logged_in:
        render_login()
        return

    # Khi đã đăng nhập: nạp trang theo quyền hạn
    page, role_selection, course_id = render_sidebar()

    if page == PAGE_CHAT:
        st.markdown(CHAT_LAYOUT_CSS, unsafe_allow_html=True)
        render_chat(course_id, role_selection)
    elif page == PAGE_UPLOAD:
        st.markdown(WIDE_LAYOUT_CSS, unsafe_allow_html=True)
        render_upload(course_id, role_selection)
    else:
        st.markdown(WIDE_LAYOUT_CSS, unsafe_allow_html=True)
        render_dashboard(role_selection)


main()
