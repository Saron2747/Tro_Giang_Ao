import psycopg2
import bcrypt
from backend.core.config import database_config, load_environment, required_env

load_environment()
DB_CONFIG = database_config()

users_to_seed = [
    ('admin', required_env('SEED_ADMIN_PASSWORD'), 'Quản Trị Viên Hệ Thống', 'ROLE_ADMIN'),
    ('gv_an', required_env('SEED_GV_AN_PASSWORD'), 'ThS. Nguyễn Văn An', 'ROLE_LECTURER'),
    ('sv_hung', required_env('SEED_SV_HUNG_PASSWORD'), 'Lê Mạnh Hùng', 'ROLE_STUDENT'),
    ('AT210444', required_env('SEED_AT210444_PASSWORD'), 'Trần Tuyết Mai', 'ROLE_STUDENT'),
    ('GV2104', required_env('SEED_GV2104_PASSWORD'), 'TS. Hoàng Quang Minh', 'ROLE_LECTURER'),
]

def hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    for username, pwd, full_name, role in users_to_seed:
        hashed = hash_pw(pwd)
        cur.execute("""
            INSERT INTO users (username, password_hash, full_name, role_id, is_active)
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (username) DO UPDATE 
            SET password_hash = EXCLUDED.password_hash,
                full_name = EXCLUDED.full_name,
                role_id = EXCLUDED.role_id;
        """, (username, hashed, full_name, role))

    conn.commit()
    cur.close()
    conn.close()
    print("--> Đã tạo xong tài khoản mẫu thành công!")
except Exception as e:
    print(f"[-] Lỗi kết nối CSDL: {e}")