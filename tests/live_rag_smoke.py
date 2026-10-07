"""Thử đường sinh thực với Ollama/Chroma local, không ghi lịch sử hoặc sửa DB.

Dùng thư viện chuẩn để chạy cả khi môi trường kiểm thử chưa cài SDK dịch vụ.
"""
import json
import logging
import os
from pathlib import Path
import sys
import time
import types
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_agent_grounding import module


class Response:
    def __init__(self, url, payload=None, timeout=120):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                self.status_code = response.status
                self.data = json.load(response)
                if url.endswith("/api/chat"):
                    print(json.dumps({"draft": self.data.get("message", {}).get("content", ""), "done_reason": self.data.get("done_reason")}, ensure_ascii=True), flush=True)
        except HTTPError as exc:
            self.status_code = exc.code
            self.data = {}

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def post(url, json, timeout):
    return Response(url, json, timeout)


class Collection:
    def __init__(self, identifier):
        self.url = "http://127.0.0.1:8000/api/v2/tenants/default_tenant/databases/default_database/collections/" + identifier

    def count(self):
        return Response(self.url + "/count").json()

    def query(self, **payload):
        response = Response(self.url + "/query", payload)
        response.raise_for_status()
        return response.json()

    def get(self, **payload):
        response = Response(self.url + "/get", payload)
        response.raise_for_status()
        return response.json()


class Client:
    def get_collection(self, name):
        collections = Response("http://127.0.0.1:8000/api/v2/tenants/default_tenant/databases/default_database/collections").json()
        return Collection(next(c["id"] for c in collections if c["name"] == name))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    module.requests = types.SimpleNamespace(post=post, exceptions=types.SimpleNamespace(ConnectionError=URLError, Timeout=TimeoutError))
    agent = module.SecureTeachingAgent.__new__(module.SecureTeachingAgent)
    agent.chroma_client = Client()
    agent.redis_client = None
    agent.ollama_chat_url = "http://127.0.0.1:11434/api/chat"
    agent.ollama_embed_url = "http://127.0.0.1:11434/api/embed"
    agent.model_name = os.getenv("LLM_MODEL", "qwen2.5:3b")
    agent.embed_model_name = os.getenv("EMBED_MODEL", "nomic-embed-text")
    agent.COSINE_DISTANCE_THRESHOLD = 0.65
    agent.DISTANCE_MARGIN = 0.10
    agent.FALLBACK_RESPONSE = "Tài liệu học phần hiện tại chưa cung cấp thông tin này."
    queries = sys.argv[1:] or ["Hàm băm là gì?", "Khóa công khai và khóa riêng được dùng để làm gì?",
                               "Sinh 2 câu hỏi trắc nghiệm về hàm băm, kèm đáp án và giải thích."]
    results = []
    for query in queries:
        started = time.monotonic()
        result = agent.execute_react_cycle(query, "ATTT_101", "ROLE_STUDENT", "smoke-test")
        row = {"query": query, "seconds": round(time.monotonic() - started, 1), **result}
        results.append(row)
        Path("tests/live_rag_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=True), flush=True)
    Path("tests/live_rag_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
