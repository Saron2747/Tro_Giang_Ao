import json
import unittest

from backend.core.grounding import clean_passage, definition_score, extractive_answer, lexical_score, parse_grounded_answer


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.context = (
            "[Đoạn 1 - Giáo trình, trang 4]\nHàm băm tạo giá trị có độ dài cố định.\n\n---\n\n"
            "[Đoạn 2 - Giáo trình, trang 5]\nMã hóa đối xứng sử dụng cùng một khóa."
        )
        self.citations = [{"chunk_id": "a", "snippet": "cũ"}, {"chunk_id": "b", "snippet": "cũ"}]
        self.data = {"answer": "Hàm băm tạo giá trị có độ dài cố định.", "sufficient": True,
                     "evidence": [{"source": 1, "quote": "Hàm băm tạo giá trị có độ dài cố định."}]}

    def parse(self):
        return parse_grounded_answer(json.dumps(self.data), self.context, self.citations, "Thiếu căn cứ.")

    def test_only_used_source_is_returned(self):
        answer, citations = self.parse()
        self.assertEqual(answer, self.data["answer"])
        self.assertEqual([c["chunk_id"] for c in citations], ["a"])
        self.assertEqual(citations[0]["snippet"], self.data["evidence"][0]["quote"])
        self.assertEqual(self.citations[0]["snippet"], "cũ")

    def test_application_extracts_quote_for_source_reference(self):
        self.data["evidence"] = [{"source": 2}]
        self.data["answer"] = "Mã hóa đối xứng sử dụng cùng một khóa."
        self.assertEqual(self.parse()[1][0]["snippet"], "Mã hóa đối xứng sử dụng cùng một khóa.")

    def test_wrong_source_and_invented_quote_rejected(self):
        for source, quote in [(2, self.data["answer"]), (3, self.data["answer"]),
                              (True, self.data["answer"]), (1, "Hàm băm có thể giải mã bằng khóa.")]:
            with self.subTest(source=source, quote=quote):
                self.data["evidence"] = [{"source": source, "quote": quote}]
                with self.assertRaises(ValueError):
                    self.parse()

    def test_title_cannot_be_evidence(self):
        self.data["evidence"][0]["quote"] = "Giáo trình, trang 4"
        with self.assertRaises(ValueError):
            self.parse()

    def test_whitespace_and_unicode_normalization(self):
        import unicodedata
        self.data["evidence"][0]["quote"] = unicodedata.normalize("NFD", "Hàm băm tạo giá trị\n có độ dài cố định.")
        self.assertEqual(len(self.parse()[1]), 1)

    def test_insufficient_information_has_no_citations(self):
        self.data["sufficient"] = False
        self.assertEqual(self.parse(), ("Thiếu căn cứ.", []))

    def test_empty_answer_or_evidence_rejected(self):
        for field, value in [("answer", ""), ("evidence", []), ("sufficient", "true")]:
            old = self.data[field]
            self.data[field] = value
            with self.assertRaises(ValueError):
                self.parse()
            self.data[field] = old

    def test_exact_term_outranks_related_topic(self):
        query = "Mã hóa đối xứng là gì?"
        self.assertGreater(lexical_score(query, "Mã hóa đối xứng sử dụng cùng một khóa."),
                           lexical_score(query, "Mã hóa bất đối xứng sử dụng hai khóa."))

    def test_slide_header_removed_without_changing_formula(self):
        text = "Bộ môn Khoa Học An Toàn Thông Tin 30 September 2026 | Page 5\nh = H(M)\nĐộ dài cố định"
        self.assertEqual(clean_passage(text), "h = H(M)\nĐộ dài cố định")

    def test_definition_preferred_over_usage_for_definition_question(self):
        self.assertGreater(definition_score("Hàm băm là gì?", "Hàm băm\nCó thể gọi là bản tóm lược của thông báo."),
                           definition_score("Hàm băm là gì?", "Sử dụng hàm băm\nXác thực thông điệp."))
        self.assertEqual(definition_score("Các ứng dụng hàm băm?", "Hàm băm\nCó thể gọi là bản tóm lược."), 0.0)

    def test_definition_answer_uses_only_original_passage(self):
        passage = "Hàm băm\nCó thể gọi là bản tóm lược của thông báo.\nĐộ dài cố định."
        result = extractive_answer("Hàm băm là gì?", "[Đoạn 1 - Giáo trình]\n" + passage, self.citations)
        self.assertEqual(result[0], "Theo giáo trình:\n\n> " + passage.replace("\n", "\n> "))
        self.assertEqual(len(result[1]), 1)
        self.assertIsNone(extractive_answer("Giải thích hàm băm là gì và cho ví dụ?", "[Đoạn 1 - Giáo trình]\n" + passage, self.citations))

    def test_vietnamese_spelling_variants_match_without_changing_source(self):
        self.assertEqual(lexical_score("Mã hóa đối xứng", "Mã hoá đối xứng"), 1.0)
        passage = "Khoá công khai được dùng để mã hoá, khoá riêng để giải mã."
        result = extractive_answer("Khóa công khai và khóa riêng được dùng để làm gì?", "[Đoạn 1 - Giáo trình]\n" + passage, self.citations)
        self.assertIsNotNone(result)
        self.assertEqual(result[1][0]["snippet"], passage)


if __name__ == '__main__':
    unittest.main()
