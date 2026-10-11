"""BM25 cho từ khóa và tên lệnh; không cần thêm dịch vụ hoặc mô hình."""
import math
import re
import unicodedata
from collections import Counter

STOP_WORDS = set("là gì của và các những một có cho tôi hãy giúp về được trong với như nào tại sao giải thích tạo soạn câu hỏi bài tập trắc nghiệm so sánh phân biệt tóm tắt chức năng cách hoạt động lệnh số".split())


def tokens(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\bbù\s+(hai|2)\b", "bù2", text)
    text = re.sub(r"\bbộ nhớ (?:đệm|cache)\b", "cache", text)
    return [t for t in re.findall(r"[^\W_]+(?:[._-][^\W_]+)*", text, re.UNICODE)
            if t not in STOP_WORDS and (len(t) > 1 or t.isdigit())]


class LexicalIndex:
    def __init__(self, records):
        self.ids = [cid for cid, _ in records]
        self.texts = [text for _, text in records]
        self.terms = [Counter(tokens(text)) for _, text in records]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = sum(self.lengths) / max(1, len(self.lengths))
        self.frequency = Counter(t for terms in self.terms for t in terms)

    def search(self, query, limit=32):
        wanted = set(tokens(query))
        if not wanted:
            return []
        scored = []
        definition_query = bool(re.search(r"là gì|định nghĩa|khái niệm|giải thích", query, re.I))
        for cid, terms, length, text in zip(self.ids, self.terms, self.lengths, self.texts):
            overlap = wanted.intersection(terms)
            # Nearest lexical match must contain a substantial part of the query.
            if len(overlap) < max(1, math.ceil(len(wanted) * 0.5)):
                continue
            score = 0.0
            for term in overlap:
                df = self.frequency[term]
                idf = math.log(1 + (len(self.ids) - df + 0.5) / (df + 0.5))
                tf = terms[term]
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / max(1, self.average)))
            coverage = len(overlap) / len(wanted)
            score *= coverage
            title = text.split("\n", 1)[0]
            heading_overlap = len(wanted.intersection(tokens(title))) / len(wanted)
            # Prefer a section about the topic over a fragment that merely mentions it.
            if re.match(r"\s*(?:CHƯƠNG|\d+\.)", title, re.I):
                score *= 1 + 0.4 * heading_overlap
            if definition_query:
                if re.search(r"định nghĩa\s*\d*\s*[:.]|\b\w{2,20}\s*\([^\n()]{0,80}\)\s*:", text, re.I):
                    score *= 1.4
                code_lines = sum(bool(re.search(r"\b(?:x\d+|a\d|t\d|s\d|r\d+)\s*,|#|←", line)) for line in text.splitlines())
                score *= 1 / (1 + min(code_lines, 12) * 0.035)
                if re.search(r"\bBài tập\s*\d|\d+\.\d+\.\s*Bài tập", text, re.I):
                    score *= 0.65
            scored.append((cid, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
