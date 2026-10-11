"""Phân loại yêu cầu đang được dùng trực tiếp trong luồng agent."""
import re

GEN_RE = re.compile(r"\b(tạo|soạn|biên soạn|ra|sinh|viết|luyện)\b.{0,60}\b(câu hỏi|câu(?!\s+lệnh)|đề|trắc nghiệm|tự luận|bài tập)\b|\bcho\b.{0,30}\b(?:(?:\d+|vài|một số)\s+câu|câu hỏi|bài tập|trắc nghiệm)\b", re.I)
SOURCE_RE = re.compile(r"trang\s+(mấy|bao nhiêu|nào)|tài liệu nào|giáo trình nào|lấy\s+(từ|ở)\s+đâu|nguồn\s*(tài liệu|trích dẫn|tham khảo)?|ở\s+(chương|mục|phần)\s+nào", re.I)
FOLLOW_RE = re.compile(r"\b(nó|đó|này|ấy|ở trên|vừa rồi|tiếp theo|thì sao|giải thích thêm|rõ hơn|khó hơn|dễ hơn|gợi ý|em chọn|tôi chọn)\b", re.I)


def is_source_question(query):
    # Không nhận nhầm "nguồn điện là gì" hoặc "giải thích bộ nhớ ở trang 10".
    return bool(SOURCE_RE.search(query)) and bool(re.search(
        r"trang (mấy|bao nhiêu|nào)|tài liệu nào|giáo trình nào|lấy.*đâu|^(?:cho tôi (?:xem|biết) )?nguồn(?:\s+(?:tài liệu|trích dẫn|tham khảo))?[?.! ]*$|nguồn\s+(?:tài liệu|trích dẫn|tham khảo|của câu trả lời).*(?:ở đâu|là gì)|ở (?:chương|mục|phần) nào", query, re.I
    )) and len(query.split()) <= 14


def needs_history(query):
    return bool(FOLLOW_RE.search(query)) or bool(re.fullmatch(r"\s*[abcd][.!]?\s*", query, re.I)) or len(query.split()) <= 4


def classify_query(query):
    if re.search(r"chấm\s+(bài|điểm)|(?:đánh giá|nhận xét|kiểm tra|sửa|chữa)\s+(?:giúp\s+)?(?:bài làm|bài giải|câu trả lời|đáp án)|em chọn|tôi chọn", query, re.I):
        return "evaluate"
    if "gợi ý" in query.lower():
        return "hint"
    if GEN_RE.search(query):
        return "generate"
    if re.search(r"so sánh|phân biệt", query, re.I):
        return "compare"
    if re.search(r"tóm tắt|ôn tập", query, re.I):
        return "summarize"
    return "explain"


def topic_of(query):
    explicit = re.search(r"\b(?:về|chủ đề)\s+(.+)", query, re.I)
    text = explicit[1] if explicit else query
    text = re.sub(r"[,;].*(?:đáp án|lời giải).*$", "", text, flags=re.I)
    text = re.sub(r"\b\d+\s*(?:câu|bài tập)\b", " ", text, flags=re.I)
    text = re.sub(r"\b(hãy|giúp|cho|tôi|mình|em|tạo|soạn|biên soạn|ra|sinh|viết|luyện|câu hỏi|trắc nghiệm|tự luận|đề thi|đề|bài tập|câu|khó|dễ|nâng cao|cơ bản|kèm đáp án)\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" ,.?!:;")
