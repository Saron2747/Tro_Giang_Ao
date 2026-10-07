"""Nhận diện yêu cầu học tập bằng quy tắc, không gọi thêm mô hình."""
import re

TASKS = ("explain", "compare", "summarize", "generate", "evaluate", "hint", "answer")

def learning_plan(query, history="", previous_plan=None):
    """Tách yêu cầu phổ biến; yêu cầu gốc vẫn được gửi đầy đủ cho LLM."""
    q = query.lower()
    follow = bool(re.search(r"khó hơn|dễ hơn|thêm|tiếp|câu\s*\d+|đáp án|gợi ý|rõ hơn|em chọn|tôi chọn", q)) or bool(re.fullmatch(r"\s*[abcd][.!]?\s*", q))
    previous = re.findall(r"^Sinh viên: (.*)$", history, re.M)
    inherited = dict(previous_plan) if follow and previous_plan else None
    if follow and previous:
        inherited = learning_plan(previous[0])
        for previous_query in previous[1:]:
            inherited = learning_plan(previous_query, previous_plan=inherited)
    plan = dict(inherited) if inherited else {
        "task": "answer", "topic": query, "count": 5,
        "question_type": "mixed", "option_count": 4, "difficulty": "mixed", "answer_mode": "show",
        "interactive": False, "style": "", "clarification": "", "needs_history": False,
    }
    plan["needs_history"] = bool((history or previous_plan) and follow)
    if re.search(r"(tạo|soạn|ra|sinh|cho|hỏi|luyện).*(câu|đề|bài tập|trắc nghiệm|tự luận)|khó hơn|dễ hơn|thêm\s+\d+\s+câu", q):
        plan["task"] = "generate"
    elif re.search(r"chấm|chữa|bài làm|bài giải|em chọn|tôi chọn|đáp án của em", q) or (history and re.fullmatch(r"\s*[abcd][.!]?\s*", q)):
        plan["task"] = "evaluate"
    elif "gợi ý" in q:
        plan["task"] = "hint"
    elif re.search(r"so sánh|phân biệt", q):
        plan["task"] = "compare"
    elif re.search(r"tóm tắt|ôn tập", q):
        plan["task"] = "summarize"
    elif re.search(r"giải thích|chưa hiểu|không hiểu", q):
        plan["task"] = "explain"
    elif re.search(r"đáp án|lời giải", q):
        plan["task"] = "answer"
    if re.search(r"\b(về|chủ đề)\s+", q):
        plan["topic"] = re.split(r"\b(?:về|chủ đề)\s+", query, maxsplit=1, flags=re.I)[1]
    elif not plan["needs_history"]:
        plan["topic"] = query
    if plan["task"] == "generate":
        plan["topic"] = re.sub(r"[,;].*(?:đáp án|lời giải).*$", "", plan["topic"], flags=re.I).strip()
        if not plan["needs_history"] and not re.search(r"\b(về|chủ đề)\s+", q):
            topic = re.sub(r"\b(hãy|giúp|cho|tôi|em|mình|tạo|soạn|ra|sinh|hỏi|từng|câu hỏi|câu|đề thi|đề|bài tập|trắc nghiệm|tự luận|khó|dễ|cơ bản|nâng cao)\b|\d+", " ", q)
            plan["topic"] = " ".join(topic.split()).strip(" ,.?!")
            if not plan["topic"]:
                plan["clarification"] = "Bạn muốn luyện tập chủ đề hoặc chương nào?"
    return normalize_plan(plan, query, history or ("previous" if previous_plan else ""))


def normalize_plan(data, query, history=""):
    if not isinstance(data, dict) or data.get("task") not in TASKS:
        raise ValueError("Kế hoạch học tập không hợp lệ")
    plan = dict(data)
    for field in ("topic", "style", "clarification"):
        if not isinstance(plan.get(field), str):
            raise ValueError(f"Thiếu {field}")
    for field, choices in (("question_type", ("multiple_choice", "essay", "mixed", "true_false", "short_answer")),
                           ("difficulty", ("easy", "medium", "hard", "mixed")),
                           ("answer_mode", ("show", "hide"))):
        if plan.get(field) not in choices:
            raise ValueError(f"Sai {field}")
    if type(plan.get("count")) is not int or plan["count"] < 1:
        raise ValueError("Số câu không hợp lệ")
    plan.setdefault("option_count", 4)
    if type(plan["option_count"]) is not int or not 2 <= plan["option_count"] <= 26:
        raise ValueError("Số lựa chọn không hợp lệ")
    if any(type(plan.get(f)) is not bool for f in ("interactive", "needs_history")):
        raise ValueError("Chế độ hội thoại không hợp lệ")
    q = query.lower()
    # Các ràng buộc tường minh có ưu tiên hơn kết quả phân tích của LLM.
    count = re.search(r"\b(\d+)\s*(?:câu|bài tập)\b", q)
    if count:
        plan["count"] = max(1, int(count[1]))
    if "trắc nghiệm" in q and "tự luận" not in q:
        plan["question_type"] = "multiple_choice"
    elif "tự luận" in q and "trắc nghiệm" not in q:
        plan["question_type"] = "essay"
    if re.search(r"đúng\s*[/–-]?\s*sai", q):
        plan["question_type"] = "true_false"
        plan["option_count"] = 2
    elif re.search(r"trả lời ngắn|điền.{0,10}(?:trống|khuyết)", q):
        plan["question_type"] = "short_answer"
    option_count = re.search(r"\b(\d+)\s*(?:lựa chọn|phương án)\b", q)
    if option_count and 2 <= int(option_count[1]) <= 26:
        plan["option_count"] = int(option_count[1])
    if re.search(r"khó hơn|nâng cao|\bkhó\b", q):
        plan["difficulty"] = "hard"
    elif re.search(r"dễ hơn|\bdễ\b|cơ bản", q):
        plan["difficulty"] = "easy"
    hide = re.search(r"(?:không|đừng|chưa).{0,25}(?:đáp án|lời giải)|(?:giấu|ẩn)\s+(?:đáp án|lời giải)", q)
    if hide:
        plan["answer_mode"] = "hide"
    elif re.search(r"(?:kèm|hiện|xem|đưa|cho|có).{0,15}(?:đáp án|lời giải)", q):
        plan["answer_mode"] = "show"
    if re.search(r"(?:hỏi|đưa|ra|cho).{0,15}từng câu|mỗi lần\s*(?:một|1)\s*câu", q):
        plan["interactive"] = True
    if re.search(r"tất cả|toàn bộ|cùng lúc|không.{0,10}từng câu", q):
        plan["interactive"] = False
    if not history:
        plan["needs_history"] = False
    if plan["interactive"] and plan["task"] == "generate":
        plan["answer_mode"] = "hide"
    return plan


def learning_history(messages, budget=2500):
    """Giữ lượt gần nhất và yêu cầu luyện tập gốc khi hội thoại kéo dài."""
    valid = [m for m in messages if m.get("role") in ("user", "assistant")]
    selected = valid[-6:]
    # Giữ yêu cầu sinh đề gần nhất để 'tiếp', 'khó hơn' không mất phạm vi.
    for message in reversed(valid[:-6]):
        if message["role"] == "user" and re.search(
                r"(tạo|soạn|ra|cho|hỏi).*(câu|đề|bài tập)", str(message.get("content", "")), re.I):
            selected.insert(0, message)
            break
    lines, remaining = [], budget
    for message in reversed(selected):
        label = "Sinh viên" if message["role"] == "user" else "Trợ giảng"
        content = str(message.get("content", ""))
        if message["role"] == "assistant":
            content = content[:1800]
        line = f"{label}: {content}"
        if len(line) > remaining:
            if not lines:
                lines.append(line[:remaining] + "\n[Lượt này vượt dung lượng lịch sử]")
            break
        lines.append(line)
        remaining -= len(line) + 1
    return "\n".join(reversed(lines))


def task_instructions(plan):
    instructions = {
        "explain": "Giải thích theo trình độ người học: khái niệm, cách hoạt động, ví dụ minh họa khi hữu ích. Nếu chưa hiểu, đổi cách giải thích, không lặp nguyên văn.",
        "compare": "So sánh các đối tượng theo cùng tiêu chí; nêu giống, khác và điều kiện áp dụng khi nguồn đủ căn cứ.",
        "summarize": "Tổng hợp ý trọng tâm đúng phạm vi, liên kết các ý và chỉ ra điểm dễ nhầm. Không tự sinh đề nếu chưa được yêu cầu.",
        "generate": "Biên soạn câu hỏi mới từ kiến thức, không đòi tài liệu chứa sẵn đề. Phủ các ý khác nhau, tránh trùng câu trước. Đúng số lượng, dạng câu và mức độ yêu cầu.",
        "evaluate": "Đối chiếu đúng bài làm sinh viên với nguồn: phần đúng, lỗi cụ thể, cách sửa. Không coi câu trả lời của trợ giảng là bài làm. Thiếu đề hoặc bài làm thì hỏi đúng phần thiếu. Chỉ chấm điểm khi được yêu cầu; nếu chấm, nêu tiêu chí.",
        "hint": "Chỉ đưa gợi ý cho bước tiếp theo, tăng dần theo yêu cầu. Không tiết lộ đáp án hay lời giải hoàn chỉnh.",
        "answer": "Trả lời trực tiếp đúng câu hỏi và phạm vi người học yêu cầu, giải thích đủ để hiểu.",
    }
    text = instructions[plan["task"]]
    if plan["task"] == "generate":
        count = 1 if plan["interactive"] else plan["count"]
        kinds = {"multiple_choice": "trắc nghiệm", "essay": "tự luận", "true_false": "đúng/sai", "short_answer": "trả lời ngắn", "mixed": "theo yêu cầu gốc; mặc định trắc nghiệm"}
        levels = {"easy": "dễ", "medium": "trung bình", "hard": "khó", "mixed": "theo yêu cầu gốc; mặc định trung bình"}
        text += f"\nSinh {count} câu, dạng {kinds[plan['question_type']]}, mức độ {levels[plan['difficulty']]}."
        if plan["question_type"] in ("multiple_choice", "mixed", "true_false"):
            text += f" Trắc nghiệm có {plan['option_count']} lựa chọn, chỉ một đáp án đúng."
        else:
            text += " Không có lựa chọn A/B/C/D; mỗi câu là một yêu cầu trả lời bằng lời."
        text += " Dễ = nhận biết; trung bình = áp dụng trực tiếp; khó = phân tích."
        if plan["answer_mode"] == "show" and not plan["interactive"]:
            text += " Kèm đáp án và giải thích ngắn, trừ khi yêu cầu gốc chỉ cần đáp án."
    text += "\nTuân thủ định dạng, độ dài, ví dụ, trình độ, ngôn ngữ và các yêu cầu bổ sung trong câu người học. Chỉ hỏi lại khi thiếu thông tin quyết định kết quả."
    if (plan["answer_mode"] == "hide" and plan["task"] != "evaluate") or plan["task"] == "hint":
        text += "\nKhông hiển thị đáp án, lời giải hoặc câu chữ làm lộ đáp án."
    if plan["interactive"] and plan["task"] in ("generate", "evaluate"):
        text += "\nLuyện tập từng câu: chỉ đưa MỘT câu mới mỗi lượt và chờ sinh viên trả lời. Khi sinh viên trả lời, nhận xét câu đang làm trước; không tự trả lời thay sinh viên."
        if plan["task"] == "evaluate":
            text += "\nSinh viên đã trả lời nên được nhận đáp án đúng và giải thích của câu vừa làm. Chỉ tiếp tục câu mới khi được yêu cầu hoặc chế độ luyện tập trước đó yêu cầu tiếp tục sau mỗi câu."
    return text


def generation_schema(plan):
    """LLM tự viết từng câu; cấu trúc buộc đúng số lượng và dạng câu."""
    if plan["question_type"] == "mixed":
        schema = generation_schema({**plan, "question_type": "multiple_choice"})
        schema["properties"]["questions"]["items"] = {"oneOf": [
            generation_schema({**plan, "question_type": kind})["properties"]["questions"]["items"]
            for kind in ("multiple_choice", "essay")
        ]}
        return schema
    kind = "multiple_choice" if plan["question_type"] == "mixed" else plan["question_type"]
    count = 1 if plan["interactive"] else plan["count"]
    choice = kind in ("multiple_choice", "true_false")
    options = plan["option_count"] if choice else 0
    properties = {
        "question_type": {"type": "string", "enum": [kind]},
        "question": {"type": "string"},
        "options": {"type": "array", "minItems": options, "maxItems": options, "items": {"type": "string"}},
        "explanation": {"type": "string"},
        "sources": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "integer", "minimum": 1}},
    }
    properties["answer_index" if choice else "answer"] = (
        {"type": "integer", "minimum": 0, "maximum": options - 1} if choice else {"type": "string"})
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "sufficient": {"type": "boolean"},
            "questions": {"type": "array", "minItems": count, "maxItems": count,
                          "items": {"type": "object", "properties": properties,
                                    "required": list(properties), "additionalProperties": False}},
        },
        "required": ["sufficient", "questions"],
    }

