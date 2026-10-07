"""Cấu hình dùng chung; không lưu mật khẩu hoặc khóa bí mật trong mã nguồn."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment():
    # Nạp đúng .env ở gốc dự án, dù khởi động từ thư mục khác.
    # Biến đã được đặt trong hệ điều hành/container vẫn có ưu tiên.
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Thiếu python-dotenv. Chạy: pip install python-dotenv") from exc
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def required_env(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Thiếu biến {name}. Hãy điền biến này trong .env hoặc môi trường chạy.")
    return value


def database_config():
    return {
        "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "dbname": required_env("POSTGRES_DB"),
        "user": required_env("POSTGRES_USER"),
        "password": required_env("POSTGRES_PASSWORD"),
    }


def redis_config():
    return {
        "host": os.getenv("REDIS_HOST", "127.0.0.1"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "password": required_env("REDIS_PASSWORD"),
    }
