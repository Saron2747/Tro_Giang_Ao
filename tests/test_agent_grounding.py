"""Kiểm tra điều phối bằng dịch vụ giả, không cần DB hay mô hình thật."""
import importlib
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def load_agent_module():
    # Chỉ giả các phụ thuộc ngoài khi môi trường kiểm thử chưa cài chúng.
    missing = {}
    for name in ("requests", "psycopg2", "chromadb", "redis"):
        try:
            importlib.import_module(name)
        except ImportError:
            missing[name] = types.ModuleType(name)
    if "requests" in missing:
        missing["requests"].post = MagicMock()
        missing["requests"].exceptions = types.SimpleNamespace(ConnectionError=ConnectionError, Timeout=TimeoutError)
    if "psycopg2" in missing:
        missing["psycopg2.extras"] = types.ModuleType("psycopg2.extras")
        missing["psycopg2.extras"].Json = lambda value: value
    if "chromadb" in missing:
        missing["chromadb.config"] = types.ModuleType("chromadb.config")
        missing["chromadb.config"].Settings = MagicMock()
    with patch.dict(sys.modules, missing):
        return importlib.import_module("backend.core.agent_core")


module = load_agent_module()


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = module.SecureTeachingAgent.__new__(module.SecureTeachingAgent)
        self.agent.COSINE_DISTANCE_THRESHOLD = 0.65
        self.agent.DISTANCE_MARGIN = 0.10
        self.agent.FALLBACK_RESPONSE = "Tài liệu học phần hiện tại chưa cung cấp thông tin này."
        self.agent.redis_client = MagicMock()
        self.agent.ollama_chat_url = "http://test/api/chat"
        self.agent.model_name = "test"
        self.agent._get_embedding = MagicMock(return_value=[1.0])
        self.collection = MagicMock()
        self.collection.count.return_value = 30
        self.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        self.agent.chroma_client = MagicMock()
        self.agent.chroma_client.get_collection.return_value = self.collection

    def meta(self, index, roles='["ROLE_STUDENT"]'):
        return {"document_id": "doc", "chunk_id": f"doc_chunk_{index}", "chunk_index": index,
                "document_title": "Giáo trình", "page_number": index + 1, "allowed_roles": roles}

    def test_retrieval_keeps_exact_concept_outside_relative_margin(self):
        self.collection.query.return_value = {
            "documents": [["Mã hóa bất đối xứng sử dụng hai khóa.", "Mã hóa đối xứng sử dụng cùng một khóa."]],
            "metadatas": [[self.meta(0), self.meta(4)]], "distances": [[0.30, 0.43]],
        }
        context, citations = self.agent.retrieve_safe_context("Mã hóa đối xứng là gì?", "ATTT", "ROLE_STUDENT")
        self.assertEqual(citations[0]["chunk_id"], "doc_chunk_4")
        self.assertIn("cùng một khóa", context)
        self.assertEqual(self.collection.get.call_count, 2)

    def test_rbac_filters_matching_document_and_neighbors(self):
        self.collection.query.return_value = {
            "documents": [["Mã hóa đối xứng dùng một khóa.", "Mã hóa đối xứng dùng khóa bí mật."]],
            "metadatas": [[self.meta(0, '["ROLE_ADMIN"]'), self.meta(4)]], "distances": [[0.1, 0.2]],
        }
        self.collection.get.return_value = {
            "ids": ["doc_chunk_3"], "documents": ["Nội dung hạn chế"],
            "metadatas": [self.meta(3, '["ROLE_ADMIN"]')],
        }
        context, citations = self.agent.retrieve_safe_context("Mã hóa đối xứng", "ATTT", "ROLE_STUDENT")
        self.assertNotIn("Nội dung hạn chế", context)
        self.assertEqual([c["chunk_id"] for c in citations], ["doc_chunk_4"])

    def test_missing_distances_are_not_treated_as_perfect_matches(self):
        self.collection.query.return_value = {"documents": [["Nội dung"]], "metadatas": [[self.meta(0)]]}
        self.assertEqual(self.agent.retrieve_safe_context("Nội dung", "ATTT", "ROLE_STUDENT"), ("", []))

    def test_unrelated_neighbor_is_excluded_from_definition(self):
        self.collection.query.return_value = {
            "documents": [["Hàm băm\nTóm lược thông báo thành bản tóm lược có độ dài cố định."]],
            "metadatas": [[self.meta(4)]], "distances": [[0.4]],
        }
        self.collection.get.return_value = {
            "ids": ["doc_chunk_3", "doc_chunk_5"],
            "documents": ["Xác thực mẫu tin gồm kiểm chứng danh tính và nguồn gốc.", "Hàm băm như hàm nghiền hay tóm lược."],
            "metadatas": [self.meta(3), self.meta(5)],
        }
        context, citations = self.agent.retrieve_safe_context("Hàm băm là gì?", "ATTT", "ROLE_STUDENT", top_k=1)
        self.assertNotIn("kiểm chứng danh tính", context)
        self.assertEqual([c["chunk_id"] for c in citations], ["doc_chunk_4", "doc_chunk_5"])

    def test_generation_topic_drops_answer_format_and_keeps_acronym(self):
        self.assertEqual(self.agent._topic_of("Sinh 2 câu hỏi trắc nghiệm về hàm băm, kèm đáp án và giải thích."), "hàm băm")
        self.assertEqual(self.agent._topic_of("Tạo 2 câu hỏi về RSA"), "rsa")

    def prepare_answer(self):
        self.agent.get_session_history = MagicMock(return_value="")
        self.agent.append_session_history = MagicMock()
        self.agent.retrieve_safe_context = MagicMock(return_value=(
            "[Đoạn 1 - Giáo trình, trang 1]\nMã hóa đối xứng sử dụng cùng một khóa.", [self.meta(0)]))
        self.answer = {"answer": "Mã hóa đối xứng sử dụng cùng một khóa.", "sufficient": True,
                       "evidence": [{"source": 1, "quote": "Mã hóa đối xứng sử dụng cùng một khóa."}]}

    def response(self, data):
        response = MagicMock(status_code=200)
        response.json.return_value = {"message": {"content": json.dumps(data, ensure_ascii=False)}}
        return response

    def run_answer(self):
        return self.agent.execute_react_cycle("Mã hóa đối xứng là gì?", "ATTT", "ROLE_STUDENT", "session")

    def test_verified_answer_is_cached(self):
        self.prepare_answer()
        with patch.object(module.requests, "post", side_effect=[self.response(self.answer), self.response({"supported": True, "reason": "Khớp nguồn"})]) as post:
            result = self.run_answer()
        self.assertEqual(result["answer"], self.answer["answer"])
        self.assertEqual(post.call_count, 2)
        self.agent.append_session_history.assert_called_once()
        self.agent.redis_client.set.assert_called_once()

    def test_valid_quote_with_wrong_claim_is_rejected_by_verifier(self):
        self.prepare_answer()
        self.answer["answer"] = "Mã hóa đối xứng sử dụng hai khóa."
        with patch.object(module.requests, "post", side_effect=[self.response(self.answer), self.response({"supported": False, "reason": "Sai số lượng khóa"})]):
            result = self.run_answer()
        self.assertEqual(result, {"answer": self.agent.FALLBACK_RESPONSE, "citations": []})
        self.agent.append_session_history.assert_not_called()
        self.agent.redis_client.delete.assert_called_once_with("last_citations:session")

    def test_fabricated_quote_rejected_after_one_repair(self):
        self.prepare_answer()
        self.answer["evidence"][0]["quote"] = "Mã hóa đối xứng sử dụng hai khóa."
        with patch.object(module.requests, "post", return_value=self.response(self.answer)) as post:
            result = self.run_answer()
        self.assertEqual(result["citations"], [])
        self.assertEqual(post.call_count, 2)
        self.agent.append_session_history.assert_not_called()

    def test_corrected_quote_is_verified_before_returning(self):
        self.prepare_answer()
        wrong = {**self.answer, "evidence": [{"source": 1, "quote": "Mã hóa đối xứng sử dụng hai khóa."}]}
        with patch.object(module.requests, "post", side_effect=[self.response(wrong), self.response(self.answer), self.response({"supported": True, "reason": "Khớp nguồn"})]) as post:
            result = self.run_answer()
        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["answer"], self.answer["answer"])
        self.agent.append_session_history.assert_called_once()

    def test_verifier_failure_does_not_cache_draft(self):
        self.prepare_answer()
        with patch.object(module.requests, "post", side_effect=[self.response(self.answer), TimeoutError("timeout")]):
            result = self.run_answer()
        self.assertEqual(result["citations"], [])
        self.agent.append_session_history.assert_not_called()

    def test_simple_quiz_always_uses_llm_even_when_source_supports_cloze(self):
        self.prepare_answer()
        self.agent.retrieve_safe_context.return_value = (
            "[Đoạn 1 - Giáo trình]\n"
            "❖ Hàm băm tạo ra bản tóm lược có độ dài cố định.\n"
            "❖ Thông báo đầu vào của hàm băm có độ dài tùy ý.\n"
            "❖ Giá trị băm còn được gọi là bản tóm lược.\n"
            "❖ Bản tóm lược thông báo được xem như dấu vân tay số.", [self.meta(0)])
        generated = {"sufficient": True, "questions": [{
            "question_type": "multiple_choice", "question": "Hàm băm tạo đầu ra có đặc điểm nào?",
            "options": ["Độ dài cố định", "Dài bằng đầu vào", "Luôn rỗng", "Luôn là văn bản gốc"],
            "answer_index": 0, "explanation": "Giáo trình nêu độ dài cố định.", "sources": [1]}]}
        with patch.object(module.requests, "post", side_effect=[self.response(generated), self.response({"supported": True, "reason": "Khớp nguồn"})]) as post:
            result = self.agent.execute_react_cycle("Tạo 1 câu hỏi trắc nghiệm đơn giản về hàm băm", "ATTT", "ROLE_STUDENT", "session")
        self.assertIn(generated["questions"][0]["question"], result["answer"])
        self.assertIn("A. Độ dài cố định", result["answer"])
        self.assertEqual(post.call_count, 2)
        prompt = post.call_args_list[0].kwargs["json"]["messages"][0]["content"]
        self.assertIn("Sinh 1 câu", prompt)
        self.assertIn("chỉ một đáp án đúng", prompt)
        self.assertIn("tự biên soạn", prompt)
        self.assertIn("mức độ dễ", prompt)

    def test_quiz_instructions_preserve_count_and_hidden_answers(self):
        self.prepare_answer()
        generated = {"sufficient": True, "questions": [{
            "question_type": "multiple_choice", "question": f"Câu hỏi khác nhau số {i}?",
            "options": ["Dùng cùng khóa", "Hai khóa", "Không khóa", "Ba khóa"], "answer_index": 0,
            "explanation": "Theo nguồn dùng cùng khóa.", "sources": [1]} for i in range(3)]}
        with patch.object(module.requests, "post", side_effect=[self.response(generated), self.response({"supported": True, "reason": "Khớp nguồn"})]) as post:
            result = self.agent.execute_react_cycle("Tạo 3 câu hỏi trắc nghiệm về mã hóa đối xứng, không kèm đáp án", "ATTT", "ROLE_STUDENT", "session")
        prompt = post.call_args_list[0].kwargs["json"]["messages"][0]["content"]
        self.assertIn("Sinh 3 câu", prompt)
        self.assertIn("Không hiển thị đáp án", prompt)
        self.assertNotIn("**Đáp án:**", result["answer"])
        self.assertEqual(result["citations"][0]["snippet"], "")
        self.assertIn("**Đáp án:**", post.call_args_list[1].kwargs["json"]["messages"][1]["content"])

    def essay(self, n):
        return {"question_type": "essay", "question": f"Trình bày đặc điểm được hỏi số {n}.",
                "options": [], "answer": "Dùng cùng một khóa theo nguồn.", "explanation": "Nguồn nêu dùng cùng khóa.", "sources": [1]}

    def test_three_essays_request_uses_essay_schema_without_mcq_instructions(self):
        self.prepare_answer()
        generated = {"sufficient": True, "questions": [self.essay(i) for i in range(3)]}
        with patch.object(module.requests, "post", side_effect=[self.response(generated), self.response({"supported": True, "reason": "Khớp nguồn"})]) as post:
            result = self.agent.execute_react_cycle("tạo cho tôi 3 câu hỏi tự luận về chế độ hoạt động của mã khối đi", "ATTT", "ROLE_STUDENT", "session")
        self.assertEqual(result["answer"].count("**Câu "), 3)
        self.assertNotIn("A.", result["answer"])
        payload = post.call_args_list[0].kwargs["json"]
        schema = payload["format"]["properties"]["questions"]
        self.assertEqual((schema["minItems"], schema["maxItems"]), (3, 3))
        self.assertEqual(schema["items"]["properties"]["question_type"]["enum"], ["essay"])
        self.assertNotIn("Trắc nghiệm có 4 lựa chọn", payload["messages"][0]["content"])

    def test_incorrect_essay_count_is_repaired_before_publish(self):
        self.prepare_answer()
        wrong = {"sufficient": True, "questions": [self.essay(0)]}
        fixed = {"sufficient": True, "questions": [self.essay(i) for i in range(3)]}
        with patch.object(module.requests, "post", side_effect=[self.response(wrong), self.response(fixed), self.response({"supported": True, "reason": "Khớp nguồn"})]) as post:
            result = self.agent.execute_react_cycle("Tạo 3 câu hỏi tự luận về mã hóa đối xứng", "ATTT", "ROLE_STUDENT", "session")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["answer"].count("**Câu "), 3)

    def test_mcq_cannot_be_published_for_essay_request(self):
        self.prepare_answer()
        wrong = {"sufficient": True, "questions": [{**self.essay(i), "question_type": "multiple_choice", "options": ["A", "B", "C", "D"]} for i in range(3)]}
        with patch.object(module.requests, "post", return_value=self.response(wrong)) as post:
            result = self.agent.execute_react_cycle("Tạo 3 câu hỏi tự luận về mã hóa đối xứng", "ATTT", "ROLE_STUDENT", "session")
        self.assertEqual(post.call_count, 2)
        self.assertIn("Chưa thể sinh", result["answer"])
        self.assertEqual(result["citations"], [])
        self.agent.append_session_history.assert_not_called()

    def test_no_source_does_not_fabricate_quiz_with_or_without_llm(self):
        self.prepare_answer()
        self.agent.retrieve_safe_context.return_value = ("", [])
        with patch.object(module.requests, "post") as post:
            result = self.agent.execute_react_cycle("Tạo 2 câu hỏi trắc nghiệm về RSA", "ATTT", "ROLE_STUDENT", "session")
        post.assert_not_called()
        self.assertEqual(result, {"answer": self.agent.FALLBACK_RESPONSE, "citations": []})


if __name__ == "__main__":
    unittest.main()
