# Cấu hình môi trường

Ứng dụng và `seed_users.py` dùng `backend/core/config.py` để đọc `.env` ở gốc dự án.
Các biến đã được đặt trong môi trường chạy có ưu tiên hơn `.env`.

## Máy hiện tại

Mật khẩu dịch vụ trong `.env` đã được giữ nguyên. Mật khẩu tài khoản mẫu trước đây
ở `seed_users.py` đã được chuyển sang các biến `SEED_*_PASSWORD` trong `.env` local.
Việc chuyển mã không đổi mật khẩu của PostgreSQL, Redis hay tài khoản đã lưu trong DB.

Cài thư viện đọc `.env` vào môi trường Python dùng để chạy ứng dụng:

```powershell
cd D:\Tro_Giang_Ao
.\venv\Scripts\python.exe -m pip install python-dotenv
```

Sau đó khởi động lại ứng dụng bằng môi trường đó:

```powershell
.\venv\Scripts\python.exe -m streamlit run frontend\app.py
```

## Khi sao chép dự án sang máy mới

1. Cài các thư viện trong `backend/requirements.txt` vào môi trường Python của dự án.
2. Sao chép `.env.example` thành `.env` nếu chưa có `.env`.
3. Điền `POSTGRES_PASSWORD` và `REDIS_PASSWORD`, cùng tên DB và tài khoản phù hợp.
4. Chỉ điền `SEED_*_PASSWORD` nếu cần chạy `seed_users.py`.

Mật khẩu trống sẽ khiến ứng dụng báo rõ tên biến còn thiếu. Không có mật khẩu dự phòng
viết trong mã. `POSTGRES_PORT=5433` là cổng trên máy host trong Docker Compose hiện tại.

Docker Compose cũng đọc các biến PostgreSQL/Redis từ `.env`. Với database đã có dữ liệu,
chỉ sửa `.env` không tự đổi mật khẩu tài khoản PostgreSQL bên trong database.

## Đưa lên Git

Commit `.env.example`, `.gitignore` và mã nguồn. `.env` là cấu hình riêng của máy, được
`.gitignore` loại khỏi Git cùng với `venv`, file tạm, giáo trình và báo cáo thử chứa trích
đoạn giáo trình. Nếu `.env` đã được theo dõi trong một repository khác, `.gitignore`
không tự xóa file đó khỏi lịch sử Git.
