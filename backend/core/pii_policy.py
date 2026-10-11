"""Hide private profile fields; keep names, email and telephone contacts visible."""
import re

_FLAGS = re.IGNORECASE
_DATE = r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:ngày\s+)?\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})"
_DOB_LABEL = r"(?:ngày\s+sinh|sinh\s+ngày|sinh\s+vào\s+ngày|date\s+of\s+birth|DOB)"
_DOB = re.compile(r"(" + _DOB_LABEL + r"(?:\s+(?:mô\s+phỏng|giả|theo\s+định\s+dạng\s+khác))?\s*[*_]*\s*[:=]?\s*[*_]*\s*(?:là\s+)?)" + _DATE, _FLAGS)
_ADDRESS_LABEL = r"(?:địa\s+chỉ(?:\s+(?:nhà|ký\s+túc\s+xá|cư\s+trú|thường\s+trú|tạm\s+trú|liên\s+hệ))?|nơi\s+(?:cư\s+trú|ở)|home\s+address|residential\s+address)"
_ACADEMIC_LABEL = r"(?:thông\s+tin\s+học\s+tập|thành\s+tích\s+học\s+tập|kết\s+quả\s+học\s+tập|điểm\s+trung\s+bình(?:\s+(?:học\s+kỳ|tích\s+lũy))?|GPA|CGPA|học\s+lực|hạnh\s+kiểm|xếp\s+loại(?:\s+học\s+tập)?|điểm\s+(?:thi|môn\s+[^:\n|]{1,40})|học\s+bổng)"
_CONTACT_BOUNDARY = r"(?=[,;]\s*(?:email|e-mail|họ\s*(?:và\s*)?tên|số\s+điện\s+thoại|SĐT|điện\s+thoại)\s*[:=]|[;\n|]|$)"
_ADDRESS = re.compile(r"(" + _ADDRESS_LABEL + r"(?:\s+(?:mô\s+phỏng|giả))?\s*[*_]*\s*(?::|=|là\b|tại\b)\s*[*_]*\s*)([^\n|;]+?)" + _CONTACT_BOUNDARY, _FLAGS)
_ACADEMIC = re.compile(r"(" + _ACADEMIC_LABEL + r"(?:\s+(?:mô\s+phỏng|giả))?\s*[*_]*\s*(?::|=|là\b|đạt\b)\s*[*_]*\s*)([^\n|;]+?)" + _CONTACT_BOUNDARY, _FLAGS)
_ID_LABEL = r"(?:CCCD|CMND|căn\s+cước(?:\s+công\s+dân)?|chứng\s+minh\s+(?:nhân\s+dân|thư))"
_ID = re.compile(r"(" + _ID_LABEL + r"(?:\s+(?:có\s+(?:khoảng\s+trắng|dấu\s+gạch\s+ngang)|mô\s+phỏng|giả))?\s*[*_]*\s*[:=]?\s*[*_]*\s*(?:là\s+)?)((?:\d[ .-]?){8,11}\d)(?!\d)", _FLAGS)
_BIRTH_YEAR = re.compile(r"(\bsinh\s+năm\s+)\d{4}\b", _FLAGS)
_GPA = re.compile(r"(\b(?:GPA|CGPA)\s*(?:là|:|=)?\s*)(\d(?:[.,]\d+)?(?:\s*/\s*\d(?:[.,]\d+)?)?)", _FLAGS)
_BARE_CCCD = re.compile(r"(?<![\w@.+-])\d{12}(?![\w@])")
_TECHNICAL_ADDRESS = re.compile(r"(?:0x[0-9a-f]+|\b(?:\d{1,3}\.){3}\d{1,3}\b|https?://|@|\b(?:IP|IPv4|IPv6|MAC|bộ\s+nhớ|thanh\s+ghi|ô\s+nhớ)\b)", _FLAGS)
_SENSITIVE_HEADERS = (
    (re.compile(_DOB_LABEL, _FLAGS), "[REDACTED_DOB]"),
    (re.compile(_ADDRESS_LABEL, _FLAGS), "[REDACTED_ADDRESS]"),
    (re.compile(_ACADEMIC_LABEL, _FLAGS), "[REDACTED_ACADEMIC]"),
    (re.compile(_ID_LABEL, _FLAGS), "[REDACTED_ID]"),
)


def _tables(text):
    """Recognize Markdown/plain pipe tables and redact only protected columns."""
    result, columns = [], {}
    for line in text.splitlines():
        if "|" not in line:
            columns = {}
            result.append(line)
            continue
        cells = line.split("|")
        detected = {i: marker for i, cell in enumerate(cells) for pattern, marker in _SENSITIVE_HEADERS
                    if pattern.search(cell) and not re.search(r"\d|\[REDACTED_", cell)
                    and not re.search(r"\b(?:email|e-mail|điện\s+thoại|SĐT|IP|IPv4|IPv6|MAC)\b", cell, _FLAGS)}
        if detected:
            columns = detected
        elif columns and not all(not c.strip() or re.fullmatch(r"\s*:?-+:?\s*", c) for c in cells):
            for i, marker in columns.items():
                if i < len(cells) and cells[i].strip():
                    cells[i] = " " + marker + " "
            line = "|".join(cells)
        result.append(line)
    return "\n".join(result)


def redact_private_data(text):
    text = _tables(text or "")
    text = _ID.sub(lambda m: m[1] + "[REDACTED_ID]", text)
    lines = []
    previous = ""
    for line in text.splitlines():
        # Unlabelled 9-digit CMND on its own line; do not match ten-digit phones.
        if re.fullmatch(r"\s*\d{9}\s*", line) and re.search(_ID_LABEL, previous, _FLAGS):
            line = "[REDACTED_ID]"
        if re.search(_DOB_LABEL, previous, _FLAGS) and re.fullmatch(r"\s*" + _DATE + r"\s*", line, _FLAGS):
            line = "[REDACTED_DOB]"
        if re.search(_ADDRESS_LABEL + r"\s*[:=]\s*$", previous, _FLAGS) and line.strip() and not re.search(r"[:=]", line) and not _TECHNICAL_ADDRESS.search(line):
            line = "[REDACTED_ADDRESS]"
        if not re.search(r"hằng\s+số|số\s+liệu\s+kỹ\s+thuật|địa\s+chỉ\s+bộ\s+nhớ", line, _FLAGS):
            line = _BARE_CCCD.sub("[REDACTED_ID]", line)
        line = _DOB.sub(lambda m: m[1] + "[REDACTED_DOB]", line)
        line = _BIRTH_YEAR.sub(lambda m: m[1] + "[REDACTED_DOB]", line)
        # A DOB field can contain both dd/mm/yyyy and yyyy-mm-dd alternatives.
        if re.search(_DOB_LABEL, line, _FLAGS) and "[REDACTED_DOB]" in line:
            line = re.sub(_DATE, "[REDACTED_DOB]", line, flags=_FLAGS)
        line = _ADDRESS.sub(lambda m: m[0] if _TECHNICAL_ADDRESS.search(m[2]) or "[REDACTED_ADDRESS]" in m[2]
                            else m[1] + "[REDACTED_ADDRESS]", line)
        line = _ACADEMIC.sub(lambda m: m[1] + "[REDACTED_ACADEMIC]", line)
        line = _GPA.sub(lambda m: m[1] + "[REDACTED_ACADEMIC]", line)
        lines.append(line)
        previous = line
    return "\n".join(lines)
