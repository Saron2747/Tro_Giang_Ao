"""Gọi Ollama; không đưa phản hồi lỗi hoặc rỗng vào trí nhớ."""
import os
import re
import requests


def generate_answer(url, model, messages, task):
    payload = {
        "model": model, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": 0.25 if task == "generate" else 0.1,
                    "top_p": 0.9, "repeat_penalty": 1.05,
                    "num_ctx": int(os.getenv("LLM_NUM_CTX", "8192")),
                    "num_predict": int(os.getenv("LLM_NUM_PREDICT", "2048"))},
    }
    try:
        response = requests.post(url, json=payload, timeout=240 if task == "generate" else 120)
        response.raise_for_status()
        data = response.json()
        answer = re.sub(r"<think>.*?</think>", "", data.get("message", {}).get("content", ""), flags=re.S | re.I).strip()
        if not answer:
            return "Lỗi: Mô hình chưa tạo được câu trả lời. Vui lòng thử lại.", False
        if data.get("done_reason") == "length":
            answer += "\n\n⚠️ Câu trả lời đã chạm giới hạn độ dài; hãy yêu cầu giải thích tiếp phần còn thiếu."
            return answer, False
        return answer, True
    except requests.exceptions.Timeout:
        return "Quá thời gian phản hồi từ mô hình LLM. Vui lòng thử lại với yêu cầu ngắn hơn.", False
    except requests.exceptions.ConnectionError:
        return "Lỗi kết nối máy chủ LLM. Vui lòng kiểm tra dịch vụ Ollama.", False
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return "Lỗi: Máy chủ LLM trả về dữ liệu không hợp lệ hoặc yêu cầu thất bại.", False
