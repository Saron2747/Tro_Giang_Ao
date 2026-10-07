"""Chỉ đọc trạng thái dịch vụ local, không thay đổi mô hình hoặc dữ liệu."""
import json
from urllib.request import urlopen

for url in ("http://127.0.0.1:11434/api/tags", "http://127.0.0.1:8000/api/v2/heartbeat",
            "http://127.0.0.1:8000/api/v2/tenants/default_tenant/databases/default_database/collections"):
    try:
        with urlopen(url, timeout=3) as response:
            data = json.load(response)
        if isinstance(data, list):
            print(url, [{"id": item.get("id"), "name": item.get("name"), "metadata": item.get("metadata")} for item in data])
        else:
            print(url, [model.get("name") for model in data.get("models", [])] if "models" in data else data)
    except Exception as exc:
        print(url, type(exc).__name__)
