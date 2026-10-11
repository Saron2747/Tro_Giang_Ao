# Trợ giảng ảo an toàn

Ứng dụng Streamlit hỗ trợ hỏi đáp RAG, duyệt tài liệu, phân quyền theo vai trò
và nhật ký bảo mật. Dùng Ollama, ChromaDB, PostgreSQL và Redis chạy cục bộ.

## Chạy trên Windows

1. Cài Python và Docker Desktop có Docker Compose.
2. Sao chép `.env.example` thành `.env`, đặt tên người dùng, tên cơ sở dữ liệu
   và hai mật khẩu riêng. Nếu đã có `.env`, giữ cấu hình hiện tại.
   Frontend tự đọc `.env` ở thư mục gốc; biến môi trường đã đặt có ưu tiên cao hơn.
3. Tạo môi trường và cài thư viện:

   ```powershell
   python -m venv venv
   .\venv\Scripts\python.exe -m pip install -r backend\requirements.txt -r frontend\requirements.txt
   ```

4. Khởi động dịch vụ và tải mô hình:

   ```powershell
   docker compose up -d
   docker compose exec ollama ollama pull qwen2.5:3b
   docker compose exec ollama ollama pull nomic-embed-text
   ```

   Compose hiện yêu cầu GPU NVIDIA cho Ollama. Máy không có GPU phù hợp cần
   bỏ phần `deploy.resources.reservations.devices` của Ollama để chạy CPU.

5. Chạy giao diện:

   ```powershell
   .\venv\Scripts\python.exe -m streamlit run frontend\app.py
   ```

Schema mới chỉ tạo vai trò, chưa có tài khoản đăng nhập. Script `seed_users.py`
trên máy phát triển chứa thông tin riêng nên không đưa vào Git.
Người triển khai cần tạo tài khoản qua quy trình quản trị của mình, dùng mật khẩu
bcrypt và vai trò trong `database/init_db.sql`.

Cấu hình PostgreSQL phải khớp cơ sở dữ liệu đang chạy. Thay `.env` không tự đổi
mật khẩu PostgreSQL trong volume đã tồn tại. Đọc kỹ và sao lưu dữ liệu trước khi
chạy migration có thao tác xóa bảng.

## Nạp lên Git

Xem [GIT_UPLOAD.md](GIT_UPLOAD.md). Kho chứa mã, schema SQL và cấu hình mẫu.
Các thư mục `tests`, `docs`, dữ liệu người dùng và cấu hình bí mật được giữ trên máy.
