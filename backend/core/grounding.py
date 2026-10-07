"""Kiểm tra cấu trúc và trích dẫn; không coi trích dẫn là chứng minh ngữ nghĩa."""
import json
import re
import unicodedata


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "evidence": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {"source": {"type": "integer"}},
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "sufficient", "evidence"],
    "additionalProperties": False,
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
    "additionalProperties": False,
}


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def context_sources(context):
    """Tách nhãn do ứng dụng tạo ra, không cho phép trích từ nhãn/tên tệp."""
    return {
        int(match[1]): match[2].strip()
        for match in re.finditer(
            r"(?:^|\n\n---\n\n)\[Đoạn (\d+) - [^\n]*\]\n(.*?)(?=\n\n---\n\n\[Đoạn |\Z)",
            context, re.S,
        )
    }


def parse_grounded_answer(raw, context, citations, fallback):
    """Chỉ chấp nhận nguồn tồn tại và câu trích thực sự có trong nguồn đó."""
    data = json.loads(raw)
    if not isinstance(data, dict) or type(data.get("sufficient")) is not bool:
        raise ValueError("Thiếu sufficient")
    if not data["sufficient"]:
        return fallback, []
    answer = data.get("answer")
    evidence = data.get("evidence")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(evidence, list) or not evidence:
        raise ValueError("Câu trả lời thiếu nội dung hoặc căn cứ")
    if fallback in answer:
        return fallback, []
    sources = context_sources(context)
    used = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Căn cứ không hợp lệ")
        source = item.get("source")
        if type(source) is not int or source not in sources or not 1 <= source <= len(citations):
            raise ValueError("Nguồn không tồn tại")
        # Trích nguyên văn tại ứng dụng: mô hình nhỏ thường tự sửa câu khi chép.
        # Vẫn kiểm tra quote nếu nhận đầu ra từ một phiên bản cũ.
        quote = item.get("quote", sources[source])
        if not isinstance(quote, str) or len(normalized(quote)) < 12:
            raise ValueError("Trích dẫn quá ngắn")
        if normalized(quote) not in normalized(sources[source]):
            raise ValueError("Trích dẫn không có trong đoạn nguồn")
        if source not in used:
            citation = dict(citations[source - 1])
            citation["snippet"] = quote
            used[source] = citation
    return answer.strip(), list(used.values())


def search_text(text):
    """So khớp cả biến thể chính tả 'khóa/khoá', 'hóa/hoá' và chữ OCR thiếu dấu."""
    decomposed = unicodedata.normalize("NFD", normalized(text).lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char)).replace("đ", "d")


def parse_generated_questions(raw, plan, context, citations, fallback):
    """Kiểm tra bộ câu do LLM sinh, chỉ định dạng hiển thị; không tạo nội dung câu hỏi."""
    data = json.loads(raw)
    if not isinstance(data, dict) or type(data.get("sufficient")) is not bool:
        raise ValueError("Thiếu sufficient")
    if not data["sufficient"]:
        return fallback, [], fallback
    questions = data.get("questions")
    count = 1 if plan["interactive"] else plan["count"]
    if not isinstance(questions, list) or len(questions) != count:
        raise ValueError(f"Phải sinh đúng {count} câu hỏi")
    allowed = ("multiple_choice", "essay") if plan["question_type"] == "mixed" else (plan["question_type"],)
    visible, review, evidence, seen = [], [], [], set()
    show = plan["answer_mode"] == "show" and not plan["interactive"]
    for n, item in enumerate(questions, 1):
        if not isinstance(item, dict) or item.get("question_type") not in allowed:
            raise ValueError(f"Câu {n} phải là dạng {'/'.join(allowed)}")
        kind = item["question_type"]
        question = item.get("question")
        explanation = item.get("explanation")
        if not isinstance(question, str) or not question.strip() or not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Câu {n} thiếu nội dung hoặc giải thích")
        key = normalized(question).casefold()
        if key in seen:
            raise ValueError("Câu hỏi bị trùng")
        seen.add(key)
        options = item.get("options")
        if not isinstance(options, list):
            raise ValueError(f"Câu {n} thiếu options")
        block = f"**Câu {n}.** {question.strip()}"
        if kind in ("multiple_choice", "true_false"):
            if len(options) != plan["option_count"] or any(not isinstance(o, str) or not o.strip() for o in options):
                raise ValueError(f"Câu {n} phải có {plan['option_count']} lựa chọn")
            if len({normalized(o).casefold() for o in options}) != len(options):
                raise ValueError(f"Câu {n} có lựa chọn trùng nhau")
            index = item.get("answer_index")
            if type(index) is not int or not 0 <= index < len(options):
                raise ValueError(f"Câu {n} có chỉ số đáp án sai")
            block += "\n\n" + "\n".join(f"{chr(65 + i)}. {o.strip()}" for i, o in enumerate(options))
            answer = f"{chr(65 + index)}. {options[index].strip()}"
        else:
            answer = item.get("answer")
            if options or not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Câu {n} tự luận/trả lời ngắn phải có đáp án và không có lựa chọn")
            if len(re.findall(r"(?:^|\s)[A-D]\s*[.)]\s+", question)) >= 2:
                raise ValueError(f"Câu {n} tự luận không được chứa lựa chọn A/B/C/D")
        sources = item.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Câu {n} thiếu nguồn cho đáp án")
        evidence.extend({"source": source} for source in sources)
        full = block + f"\n\n**Đáp án:** {answer}\n\n**Giải thích:** {explanation.strip()}"
        review.append(full)
        visible.append(full if show else block)
    if plan["question_type"] == "mixed" and count > 1 and len({item["question_type"] for item in questions}) < 2:
        raise ValueError("Bộ câu hỏi hỗn hợp phải có cả trắc nghiệm và tự luận")
    review_text, used = parse_grounded_answer(json.dumps({
        "sufficient": True, "answer": "\n\n".join(review), "evidence": evidence,
    }, ensure_ascii=False), context, citations, fallback)
    if not show:
        used = [{**citation, "snippet": ""} for citation in used]
    return "\n\n".join(visible), used, review_text


_STOP_WORDS = set(search_text("là gì các của và có trong như thế nào hãy cho biết trình bày nêu giải thích về một những được để với theo giáo trình tài liệu câu hỏi tôi bạn làm").split())


def clean_passage(text):
    """Bỏ dòng đầu/chân trang của slide, giữ các dòng nội dung và công thức."""
    return re.sub(r"^.*\|\s*(?:Page|Trang)\s+\d+\s*$", "", text, flags=re.M | re.I).strip()


def lexical_score(query, document):
    """Ưu tiên thuật ngữ và cụm liên tiếp trong các ứng viên tìm bằng vector."""
    query_words = re.findall(r"\w+", search_text(query))
    doc_words = re.findall(r"\w+", search_text(document))
    terms = set(query_words) - _STOP_WORDS
    if not terms:
        return 0.0
    coverage = len(terms.intersection(doc_words)) / len(terms)
    phrases = {tuple(query_words[i:i + size]) for size in (2, 3, 4)
               for i in range(len(query_words) - size + 1)
               if all(word in terms for word in query_words[i:i + size])}
    doc_phrases = {tuple(doc_words[i:i + size]) for size in (2, 3, 4)
                   for i in range(len(doc_words) - size + 1)}
    phrase_coverage = len(phrases & doc_phrases) / len(phrases) if phrases else coverage
    return 0.6 * coverage + 0.4 * phrase_coverage


def definition_score(query, document):
    """Khi hỏi định nghĩa, ưu tiên đoạn định nghĩa hơn sơ đồ/ứng dụng cùng tên."""
    if not re.search(r"\blà gì\b|\bđịnh nghĩa\b|\bkhái niệm\b", query, re.I):
        return 0.0
    topic = re.sub(r"\blà gì\b.*|\b(?:định nghĩa|khái niệm|hãy|nêu|về|của)\b", " ", query.lower())
    topic = search_text(topic).strip(" ?.!")
    lines = [re.sub(r"^[\W_]+", "", search_text(line)) for line in document.splitlines()]
    title_match = bool(topic) and topic in lines
    definition = bool(re.search(r"\b(?:là|định nghĩa|khái niệm|được gọi)\b", document, re.I))
    return (0.12 if title_match else 0.0) + (0.08 if definition and lexical_score(query, document) >= 0.65 else 0.0)


def purpose_score(query, document):
    if not re.search(r"dùng để|vai trò|chức năng|mục đích", query, re.I):
        return 0.0
    if lexical_score(query, document) >= 0.65 and re.search(r"(?:được )?dùng để|được sử dụng để", document, re.I):
        return 0.15
    return 0.0


def extractive_answer(query, context, citations):
    """Dùng nguyên văn đoạn định nghĩa/vai trò khớp rõ, tránh diễn giải sai bằng LLM."""
    sources = context_sources(context)
    passage = sources.get(1, "")
    if not citations or not passage:
        return None
    # Chỉ áp dụng cho câu hỏi trực tiếp, không thay thế yêu cầu giải thích/so sánh/sinh đề.
    if re.search(r"so sánh|phân biệt|ví dụ|giải thích|chấm|tóm tắt|câu hỏi|bài tập", query, re.I):
        return None
    if definition_score(query, passage) < 0.19 and purpose_score(query, passage) == 0.0:
        return None
    citation = dict(citations[0])
    citation["snippet"] = passage
    return "Theo giáo trình:\n\n> " + passage.replace("\n", "\n> "), [citation]

