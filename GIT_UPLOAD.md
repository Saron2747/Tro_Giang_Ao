# Hướng dẫn nạp dự án lên GitHub

## 1. Tạo kho

Đăng nhập GitHub, chọn **New repository**, đặt tên ví dụ `Tro_Giang_A`.
Nên chọn **Private**. Không tích tạo README, .gitignore hay license vì đã có
tệp trên máy. Bấm **Create repository**, sao chép URL HTTPS của kho.

## 2. Kiểm tra tệp

Mở PowerShell:

```powershell
cd D:\Tro_Giang_A
git status --short
git status --short --ignored
```

Nếu Git báo chưa phải kho Git, chạy `git init -b main` rồi kiểm tra lại.
Các mục có `!!` đang bị bỏ qua. `.env`, `venv`, `knowledge_storage`,
`temp_uploads`, `output`, `tmp`, `seed_users.py`, `tests`, `docs` và các tệp sơ đồ phải bị bỏ qua.

`.gitignore` chỉ bỏ qua tệp chưa được theo dõi. Nếu sau này thêm nhầm tệp,
dùng `git rm --cached -- duong-dan-tep` để bỏ theo dõi và giữ tệp trên máy.
Không dùng `git add -f` để ép thêm mục bị bỏ qua.

## 3. Chọn tệp và tạo bản lưu

Dùng danh sách cụ thể để tránh đưa nhầm tệp mới:

```powershell
git add -- .gitignore .env.example README.md GIT_UPLOAD.md docker-compose.yml backend frontend database
git diff --cached --stat
git diff --cached
git ls-files
```

Đọc danh sách và nội dung trước khi commit: không có mật khẩu thật, khóa API,
tài liệu riêng, dữ liệu người dùng hoặc tệp database. Hai thư mục `tests` và
`docs` được giữ trên máy và không đưa vào kho Git.

Đặt thông tin tác giả riêng cho kho, thay phần ví dụ bằng thông tin của bạn:

```powershell
git config user.name "TEN_CUA_BAN"
git config user.email "EMAIL_CUA_BAN"
git commit -m "Initial project import"
```

Có thể dùng email noreply do GitHub cung cấp để hạn chế lộ email cá nhân.

## 4. Kết nối và tải lên

Thay URL mẫu bằng URL kho vừa tạo:

```powershell
git remote add origin https://github.com/TAI_KHOAN/TEN_KHO.git
git remote -v
git push -u origin main
```

Nếu `origin` đã tồn tại, kiểm tra `git remote -v` trước; chỉ dùng
`git remote set-url origin URL_KHO_CUA_BAN` khi muốn đổi đúng kho đó.
Git có thể mở trình duyệt đăng nhập. Không dán token vào URL remote hoặc mã nguồn.
Nếu kho từ xa có nội dung và Git từ chối push, không dùng `--force`;
cần đối chiếu lịch sử trước khi đồng bộ.

## 5. Kiểm tra sau khi tải

Mở kho GitHub, kiểm tra đúng các thư mục đã chọn và không có các mục bị bỏ qua.
Chạy `git status` để xem thay đổi còn lại trên máy.

`.gitignore` không phát hiện mọi bí mật. Với lần cập nhật tiếp theo, luôn đọc
`git diff --cached` trước khi commit. Nếu bí mật từng bị tải lên, phải thay/thu hồi
bí mật và xử lý lịch sử Git; xóa tệp ở commit mới là chưa đủ.
