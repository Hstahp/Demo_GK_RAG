"""
==============================================================================
DEDOCU AI SCHOLAR - MULTI-DOCUMENT RAG & RECOMMENDER DEMO (SCRUM-7)
Đề tài: Mạng xã hội chia sẻ tài liệu học thuật phi tập trung (StuDocu Clone)
Chức năng: Khung Chatbot AI tương tác thời gian thực, đọc hiểu kho tài liệu đa môn,
           vừa đề xuất tài liệu liên quan trên DeDocu, vừa trả lời có trích dẫn trang.
Công nghệ: Python http.server (ThreadingHTTPServer), PyMuPDF, Dot-Product Vector Search,
           Gemini 1.5 Flash, Deterministic Hash (zlib.crc32).
Quy tắc: Tiếng Việt có dấu 100% chuẩn mực; tuyệt đối không dùng emoji.

[TỐI ƯU v2.0 - SCRUM-7]:
  1. Dot Product thay Cosine Similarity (vectors chuẩn hóa L2) -> tăng tốc ~5x search
  2. Pre-compute _content_lower & _token_set khi load index -> giảm tải mỗi query
  3. Deterministic hash: zlib.crc32 thay hash() Python (tất định, không ngẫu nhiên)
  4. CONVERSATIONAL_STOPWORDS: hằng số module-level, không khai báo trùng lặp
  5. ThreadingHTTPServer: xử lý đa kết nối đồng thời, không block khi gọi Gemini API
  6. Port fallback tự động: 8088 -> 8089 -> 8090...
  7. CORS Headers: sẵn sàng tích hợp Frontend React/Next.js
  8. Latency Badge: hiển thị thời gian phản hồi (ms) + chế độ hoạt động
  9. Render Markdown: bold, italic, blockquote, list, code, heading
  10. Citation Preview Popover: hover [Trang X] hiện đoạn trích nhanh
  11. localStorage API Key: tự nhớ key sau khi F5 trình duyệt
  12. Copy Button: sao chép câu trả lời AI vào clipboard
==============================================================================
"""

import os
import sys
import json
import math
import re
import time
import zlib
import socket
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import List, Dict, Any, Optional

# Đảm bảo UTF-8 cho console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ==============================================================================
# [TỐI ƯU 4] STOPWORDS CHUNG - khai báo 1 lần, dùng mọi nơi trong hệ thống
# ==============================================================================

CONVERSATIONAL_STOPWORDS: set = {
    "vậy", "thì", "tìm", "hiểu", "về", "tí", "đi", "nhỉ", "tôi", "đang", "có", "hứng", "thú",
    "môn", "gì", "cũng", "được", "bạn", "giới", "thiệu", "cho", "không", "học", "nhé", "nào",
    "xem", "thử", "với", "ơi", "ạ", "cần", "muốn", "hãy", "chút", "một", "giáo", "trình",
    "bài", "giảng", "toàn", "phần", "tập", "viên", "đại", "và", "các", "của", "là", "trong",
    "này", "những", "để", "khi", "nhiều", "thế", "như", "ra", "sao", "lại", "đã", "sẽ",
    "nhớ", "thống", "tin", "hệ", "tại", "nhà", "biết"
}


# ==============================================================================
# 1. QUẢN LÝ KHO TRI THỨC ĐA MÔN HỌC (MULTI-DOCUMENT KNOWLEDGE BASE)
# ==============================================================================

class MultiDocumentRAGEngine:
    def __init__(self, index_file_path: str):
        self.index_file_path = index_file_path
        self.vector_dim = 768
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.loaded = False

    def load_or_build_index(self, dataset_base_dir: str, api_key: Optional[str] = None):
        if os.path.exists(self.index_file_path):
            try:
                with open(self.index_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    raw_chunks = data.get("chunks", [])
                    # [TỐI ƯU 2] Pre-compute lowercase & token set khi nạp index
                    # Thay vì gọi .lower() + re.findall() cho 1564 chunks mỗi câu hỏi,
                    # tính 1 lần duy nhất khi khởi động -> O(1) tra cứu trong query.
                    for c in raw_chunks:
                        c["_content_lower"] = c.get("content", "").lower()
                        c["_token_set"] = set(
                            w for w in re.findall(r'\w+', c["_content_lower"])
                            if len(w) > 2
                        )
                    self.chunks = raw_chunks
                    self.loaded = True
                    print(f">> Đã tải chỉ mục sẵn có: {len(self.documents)} tài liệu, {len(self.chunks)} chunks.")
                    return
            except Exception as e:
                print(f"[Cảnh báo] Lỗi đọc file chỉ mục ({e}), tiến hành tạo lại.")

        from ingest_documents import IngestionPipeline
        pipeline = IngestionPipeline(vector_dim=self.vector_dim)
        res = pipeline.ingest_full_dataset(
            base_dataset_dir=dataset_base_dir,
            metadata_file_path=os.path.join(dataset_base_dir, "metadata.json"),
            output_index_path=self.index_file_path,
            api_key=api_key
        )
        if res.get("success"):
            with open(self.index_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
                raw_chunks = data.get("chunks", [])
                for c in raw_chunks:
                    c["_content_lower"] = c.get("content", "").lower()
                    c["_token_set"] = set(
                        w for w in re.findall(r'\w+', c["_content_lower"])
                        if len(w) > 2
                    )
                self.chunks = raw_chunks
                self.loaded = True

    def _deterministic_hash(self, word: str) -> int:
        """
        [TỐI ƯU 3] Thay thế hash() của Python (ngẫu nhiên do PYTHONHASHSEED)
        bằng zlib.crc32 (tất định, nhất quán 100% giữa mọi lần chạy Python).
        Đảm bảo vector offline khi Ingestion và khi Query khớp nhau hoàn toàn.
        """
        return zlib.crc32(word.encode("utf-8")) & 0x7fffffff

    def get_embedding(self, text: str, api_key: Optional[str] = None) -> List[float]:
        if api_key and REQUESTS_AVAILABLE:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
                res = requests.post(url, json={"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}, timeout=8)
                if res.status_code == 200:
                    val = res.json().get("embedding", {}).get("values", [])
                    if val:
                        return val
            except Exception:
                pass

        # [TỐI ƯU 3] Fallback Vectorizer dùng zlib.crc32 thay hash() để tất định
        words = re.findall(r'\w+', text.lower())
        vec = [0.0] * self.vector_dim
        for i, w in enumerate(words):
            h = self._deterministic_hash(w) % self.vector_dim
            vec[h] += 1.0 / (math.log(i + 2))

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def dot_product(v1: List[float], v2: List[float]) -> float:
        """
        [TỐI ƯU 1] Dùng Dot Product thay Cosine Similarity đầy đủ.
        Vì tất cả vector embedding đều đã được chuẩn hóa (L2-norm = 1.0),
        dot_product(u,v) = cos(angle(u,v)) nhưng bỏ được 2 phép sqrt + 1 phép chia.
        Tăng tốc ~5x khi quét toàn bộ 1564 chunks mỗi câu hỏi.
        """
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        return sum(a * b for a, b in zip(v1, v2))

    def match_recommended_document(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Nhận diện chính xác chuyên ngành và giáo trình phù hợp,
        lọc triệt để các từ đệm giao tiếp tiếng Việt và bắt đúng chuyên ngành.
        """
        # [TỐI ƯU 4] Dùng CONVERSATIONAL_STOPWORDS module-level, không khai báo lại
        q_lower = " " + question.lower() + " "
        q_tokens = set(re.findall(r'\w+', question.lower())) - CONVERSATIONAL_STOPWORDS

        # Bảng ánh xạ chuyên ngành học thuật cốt lõi
        academic_domains = {
            "toán": ["Toán & Khoa học Tự nhiên", "Toán học"],
            "toán học": ["Toán & Khoa học Tự nhiên", "Toán học"],
            "giải tích": ["Toán & Khoa học Tự nhiên", "Toán học"],
            "đại số": ["Toán & Khoa học Tự nhiên", "Toán học"],
            "xác suất": ["Toán & Khoa học Tự nhiên"],
            "thống kê": ["Toán & Khoa học Tự nhiên"],
            "vật lý": ["Toán & Khoa học Tự nhiên"],
            "hóa học": ["Hóa học & Khoa học Tự nhiên", "Toán & Khoa học Tự nhiên"],
            "kinh tế": ["Kinh tế & Quản lý"],
            "vi mô": ["Kinh tế & Quản lý"],
            "vĩ mô": ["Kinh tế & Quản lý"],
            "kế toán": ["Kinh tế & Quản lý"],
            "tài chính": ["Kinh tế & Quản lý"],
            "lập trình": ["Công nghệ Thông tin"],
            "oop": ["Công nghệ Thông tin"],
            "phần mềm": ["Công nghệ Thông tin"],
            "an toàn thông tin": ["Công nghệ Thông tin"],
            "bảo mật": ["Công nghệ Thông tin"],
            "mạng máy tính": ["Công nghệ Thông tin"],
            "rag": ["Công nghệ Thông tin"],
            "tiếng anh": ["Ngoại ngữ"],
            "toeic": ["Ngoại ngữ"],
            "y học": ["Y tế Công cộng & Y Dược", "Y sinh & Dịch tễ học"],
            "dược": ["Dược học", "Y tế Công cộng & Y Dược"],
            "triết học": ["Lý luận Chính trị & Xã hội"],
            "pháp luật": ["Pháp luật & Xã hội"]
        }

        best_doc = None
        max_score = 0

        for doc in self.documents:
            score = 0
            fac = doc.get("faculty", "")
            cname = doc.get("course_name", "").lower()
            title = doc.get("title", "").lower()

            # 1. Khớp theo Chuyên ngành đào tạo (Faculty Boosting)
            for dkw, target_facs in academic_domains.items():
                if (" " + dkw + " ") in q_lower or dkw in q_tokens:
                    if any(tf.lower() in fac.lower() for tf in target_facs):
                        score += 8
                        # Ưu tiên các môn cơ sở ngành chuẩn
                        if any(core in title or core in cname for core in ["giải tích 1", "kinh tế vi mô", "lập trình hướng đối tượng", "cơ sở an toàn thông tin"]):
                            score += 5

            # 2. Khớp trực tiếp tên môn học
            if cname and len(cname) > 3 and (" " + cname + " ") in q_lower:
                score += 7

            # 3. Khớp từ khóa chuyên sâu (Tags)
            for kw in doc.get("keywords", []):
                kw_l = kw.lower().strip()
                if kw_l in CONVERSATIONAL_STOPWORDS or len(kw_l) < 3:
                    continue
                if " " in kw_l and (" " + kw_l + " ") in q_lower:
                    score += 5
                elif kw_l in q_tokens:
                    score += 2

            if score > max_score and score >= 4:
                max_score = score
                best_doc = doc

        return best_doc

    def search_chunks(self, question: str, query_vec: List[float], filter_doc_id: Optional[str] = None, recommended_doc_id: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        # [TỐI ƯU 4] Dùng CONVERSATIONAL_STOPWORDS module-level
        q_words = set(w for w in re.findall(r'\w+', question.lower()) if len(w) > 2 and w not in CONVERSATIONAL_STOPWORDS)

        # RÀNG BUỘC CHẶT CHẼ: Nếu đã xác định tài liệu đề xuất, bắt buộc chỉ trích xuất từ chính tài liệu đó
        target_doc_id = filter_doc_id or recommended_doc_id
        candidate_chunks = [c for c in self.chunks if c.get("document_id") == target_doc_id] if target_doc_id else self.chunks

        # Nếu tìm trong tài liệu mục tiêu mà không có, mới mở rộng sang toàn kho
        if not candidate_chunks:
            candidate_chunks = self.chunks

        scored = []
        for c in candidate_chunks:
            c_vec = c.get("embedding", [])
            # [TỐI ƯU 1] Dot Product thay Cosine Similarity (vectors đã chuẩn hóa L2)
            dot_score = self.dot_product(query_vec, c_vec)

            # [TỐI ƯU 2] Dùng pre-computed _token_set thay .lower() + re.findall() mỗi lần
            precomputed_tokens = c.get("_token_set", set())
            keyword_overlap = sum(1 for w in q_words if w in precomputed_tokens)
            hybrid_score = dot_score + (keyword_overlap * 0.12)

            scored.append({
                "chunk_id": c.get("chunk_id"),
                "document_id": c.get("document_id"),
                "document_title": c.get("document_title"),
                "faculty": c.get("faculty"),
                "course_name": c.get("course_name"),
                "university": c.get("university"),
                "page_number": c.get("page_number"),
                "content": c.get("content"),
                "score": round(hybrid_score, 4),
                "pure_cosine": round(dot_score, 4)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def answer_question(self, question: str, filter_doc_id: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
        # [TỐI ƯU 8] Đo latency từ đầu hàm
        t_start = time.monotonic()

        if not self.chunks:
            return {
                "answer": "Kho dữ liệu hiện chưa có tài liệu nào được lập chỉ mục.",
                "sources": [],
                "recommended_doc": None,
                "confidence": "LOW",
                "suggested_questions": [],
                "latency_ms": 0,
                "mode": "NO_INDEX"
            }

        # 1. Kiểm tra Lời chào / Giao tiếp xã giao
        greetings = ["xin chào", "chào bạn", "hello", "hi", "bạn là ai", "bạn tên gì", "help"]
        is_greeting = any(g == question.lower().strip() or question.lower().strip().startswith(g) for g in greetings) and len(question.split()) <= 4
        if is_greeting:
            return {
                "answer": (
                    "Xin chào bạn! Tôi là **Trợ lý Học thuật DeDocu AI Scholar**.\n\n"
                    "Tôi được kết nối với toàn bộ kho tri thức gồm 106 tài liệu, giáo trình và bài giảng đại học của DeDocu Commons. "
                    "Bạn có thể hỏi tôi về bất kỳ môn học nào (Giải tích, Kinh tế vi mô, Lập trình OOP, An toàn thông tin, Hóa học, Vật lý...), "
                    "tôi sẽ vừa giải thích vừa trích dẫn chính xác số trang sách để bạn đối chiếu!"
                ),
                "sources": [],
                "recommended_doc": None,
                "confidence": "GREETING",
                "is_out_of_domain": False,
                "suggested_questions": [
                    "Cách học giỏi Giải tích 1?",
                    "Kinh tế Vi mô nghiên cứu những vấn đề gì?",
                    "Tính đa hình và kế thừa trong OOP Java là gì?"
                ],
                "latency_ms": round((time.monotonic() - t_start) * 1000, 1),
                "mode": "GREETING"
            }

        # 2. Xác định tài liệu phù hợp nhất
        recommended_doc = None
        if filter_doc_id:
            for d in self.documents:
                if d["document_id"] == filter_doc_id:
                    recommended_doc = d
                    break
        else:
            recommended_doc = self.match_recommended_document(question)

        rec_doc_id = recommended_doc["document_id"] if recommended_doc else None
        query_vec = self.get_embedding(question, api_key)
        retrieved = self.search_chunks(question, query_vec, filter_doc_id=filter_doc_id, recommended_doc_id=rec_doc_id, top_k=3)

        # 3. Kiểm soát câu hỏi ngoài phạm vi (Out-of-Domain Filter)
        top_score = retrieved[0]["score"] if retrieved else 0.0
        top_cosine = retrieved[0]["pure_cosine"] if retrieved else 0.0
        is_out_of_domain = (not recommended_doc) and (top_score < 0.22 or top_cosine < 0.15)

        if is_out_of_domain:
            if api_key and REQUESTS_AVAILABLE:
                try:
                    prompt = (
                        f"Bạn là Trợ lý Học thuật DeDocu AI Scholar trên nền tảng mạng xã hội tài liệu sinh viên.\n"
                        f"Người dùng hỏi câu hỏi: '{question}'.\n"
                        f"Chủ đề này hiện KHÔNG CÓ trong kho giáo trình/tài liệu đại học đã được lập chỉ mục của hệ thống.\n\n"
                        f"Yêu cầu:\n"
                        f"1. Trả lời bằng tiếng Việt lịch sự, thông báo rõ ràng rằng chủ đề này hiện chưa có giáo trình chính thức trong thư viện DeDocu.\n"
                        f"2. Cung cấp thông tin ngắn gọn, súc tích mang tính định hướng nếu là câu hỏi học thuật.\n"
                        f"3. Khuyến khích sinh viên tải lên (upload) tài liệu liên quan đến môn này lên DeDocu để nhận 20 điểm thưởng (Upload Points).\n"
                        f"4. Tuyệt đối không dùng emoji trong câu trả lời."
                    )
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                    resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}, timeout=10)
                    if resp.status_code == 200:
                        gemini_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                        return {
                            "answer": gemini_text,
                            "sources": [],
                            "recommended_doc": None,
                            "confidence": "OUT_OF_DOMAIN",
                            "is_out_of_domain": True,
                            "mode": "GEMINI_LIVE_OOD",
                            "suggested_questions": [
                                "Thư viện DeDocu hiện hỗ trợ những môn học nào?",
                                "Làm sao để tải tài liệu lên đổi điểm thưởng?",
                                "Tìm giáo trình Giải tích 1 hoặc Kinh tế vi mô?"
                            ],
                            "latency_ms": round((time.monotonic() - t_start) * 1000, 1)
                        }
                except Exception:
                    pass

            ood_answer = (
                f"Chủ đề bạn đang hỏi (*'{question}'*) hiện **chưa có tài liệu giáo trình chính thức** trong thư viện DeDocu.\n\n"
                f"**Gợi ý cho sinh viên:**\n"
                f"- Thư viện DeDocu hiện lưu trữ 106 tài liệu thuộc các khoa: Toán học, Công nghệ Thông tin, Kinh tế & Quản lý, Ngoại ngữ, Y Dược, v.v.\n"
                f"- Nếu bạn sở hữu tài liệu, giáo trình hoặc đề cương của môn học này, hãy **tải lên DeDocu** để nhận ngay **20 điểm thưởng (Upload Points)** mở khóa tài liệu VIP!"
            )
            return {
                "answer": ood_answer,
                "sources": [],
                "recommended_doc": None,
                "confidence": "OUT_OF_DOMAIN",
                "is_out_of_domain": True,
                "mode": "OOD_FILTER_TRIGGERED",
                "suggested_questions": [
                    "Thư viện DeDocu hiện hỗ trợ những môn học nào?",
                    "Làm sao để tải tài liệu lên đổi điểm thưởng?",
                    "Tìm giáo trình Giải tích 1 hoặc Kinh tế vi mô?"
                ],
                "latency_ms": round((time.monotonic() - t_start) * 1000, 1)
            }

        # 4. Định dạng nguồn trích dẫn
        sources = []
        for r in retrieved:
            sources.append({
                "document_id": r["document_id"],
                "document_title": r["document_title"],
                "course_name": r["course_name"],
                "page_number": r["page_number"],
                "score": r["score"],
                "quote_snippet": r["content"][:220] + "..." if len(r["content"]) > 220 else r["content"]
            })

        # 5. Sinh phản hồi thời gian thực qua Gemini API nếu có Key
        if api_key and REQUESTS_AVAILABLE:
            try:
                context_str = "\n\n".join([
                    f"--- TÀI LIỆU: {r['document_title']} (Trang {r['page_number']}) ---\n{r['content']}"
                    for r in retrieved
                ])
                prompt = (
                    f"Bạn là Trợ lý Học thuật DeDocu AI Scholar trên nền tảng mạng xã hội tài liệu sinh viên.\n"
                    f"Dưới đây là các đoạn trích từ kho giáo trình đại học đã được lập chỉ mục:\n\n"
                    f"{context_str}\n\n"
                    f"Câu hỏi của sinh viên: {question}\n\n"
                    f"Yêu cầu xử lý:\n"
                    f"1. Trả lời bằng tiếng Việt chi tiết, sư phạm, chuẩn xác.\n"
                    f"2. Nếu người dùng hỏi tìm hiểu/giới thiệu hoặc cách học môn này, hãy đưa ra định hướng các bước cụ thể, kết hợp dẫn chứng các kiến thức trọng tâm từ các đoạn trích trên.\n"
                    f"3. Dẫn rõ tên tài liệu và số trang để sinh viên dễ dàng tra cứu lại trong thư viện DeDocu.\n"
                    f"4. Không sử dụng emoji trong toàn bộ câu trả lời."
                )

                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}, timeout=12)
                if resp.status_code == 200:
                    gemini_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "answer": gemini_text,
                        "sources": sources,
                        "recommended_doc": recommended_doc,
                        "confidence": "HIGH",
                        "is_out_of_domain": False,
                        "mode": "GEMINI_LIVE",
                        "suggested_questions": self._generate_followups(recommended_doc),
                        "latency_ms": round((time.monotonic() - t_start) * 1000, 1)
                    }
            except Exception as e:
                print(f"[Cảnh báo] Gọi Gemini API thất bại ({e}), chuyển sang chế độ Offline.")

        # 6. Sinh phản hồi học thuật Offline chuẩn mực
        result = self._generate_offline_response(question, recommended_doc, retrieved, sources)
        # [TỐI ƯU 8] Gắn latency vào kết quả offline
        result["latency_ms"] = round((time.monotonic() - t_start) * 1000, 1)
        return result

    def _generate_offline_response(self, question: str, doc: Optional[Dict[str, Any]], retrieved: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        top = retrieved[0] if retrieved else None
        doc_title = doc["title"] if doc else (top["document_title"] if top else "Tài liệu học thuật")
        course_name = doc["course_name"] if doc else (top["course_name"] if top else "Môn học liên quan")
        faculty_name = doc.get("faculty", "") if doc else (top.get("faculty", "") if top else "")
        page_num = top["page_number"] if top else 1
        snippet = top["content"] if top else ""

        q_lower = question.lower()

        # Dạng câu hỏi 1: Giới thiệu / Tìm hiểu môn học mới
        if any(w in q_lower for w in ["giới thiệu", "tìm hiểu", "môn gì", "bắt đầu", "gợi ý", "có gì", "môn nào"]):
            answer_text = (
                f"Chào bạn! Để bắt đầu tìm hiểu về lĩnh vực **{faculty_name or course_name}**, tài liệu nền tảng phù hợp nhất cho bạn trên DeDocu là cuốn **'{doc_title}'** (Môn học: **{course_name}**).\n\n"
                f"**Căn cứ nội dung trích xuất từ giáo trình (Trang {page_num}):**\n"
                f"> \"{snippet}\"\n\n"
                f"**Định hướng học tập:**\n"
                f"- Đây là môn học nền tảng bắt buộc, cung cấp các khối kiến thức cốt lõi và phương pháp tư duy logic cho sinh viên.\n"
                f"- Bạn có thể nhấp vào thẻ **[Trang {page_num}]** hoặc nhấn nút đề xuất tài liệu bên dưới để mở toàn bộ bài giảng và bài tập minh họa."
            )
        # Dạng câu hỏi 2: Hỏi về phương pháp / Cách học giỏi
        elif any(w in q_lower for w in ["cách học", "phương pháp", "làm sao", "ôn thi", "kinh nghiệm"]):
            answer_text = (
                f"Để học tốt môn **{course_name}**, bạn nên kết hợp giữa việc nắm chắc lý thuyết nền tảng và rèn luyện kỹ năng giải bài tập theo các bước sau:\n\n"
                f"**1. Nắm chắc các khái niệm cốt lõi:**\n"
                f"Dựa trên giáo trình **'{doc_title}'** (Trích dẫn từ **Trang {page_num}**):\n"
                f"> \"{snippet}\"\n\n"
                f"**2. Phương pháp học tập trọng tâm:**\n"
                f"- Đọc kỹ định nghĩa, điều kiện áp dụng định lý trước khi làm bài tập.\n"
                f"- Luyện giải các dạng bài tập mẫu và đối chiếu với đề thi các năm trước của trường.\n"
                f"- Nhấp vào thẻ nguồn trích dẫn bên dưới để mở đúng trang sách chứa ví dụ giải chi tiết."
            )
        # Dạng câu hỏi 3: Hỏi về khái niệm / nội dung cụ thể
        else:
            answer_text = (
                f"Dựa trên giáo trình **'{doc_title}'** thuộc môn **{course_name}** (Trích dẫn từ **Trang {page_num}**):\n\n"
                f"> \"{snippet}\"\n\n"
                f"**Phân tích học thuật:**\n"
                f"- Nội dung trên trình bày trực tiếp về khái niệm bạn đang quan tâm trong chương trình đào tạo của trường.\n"
                f"- Bạn có thể nhấp vào nguồn trích dẫn **[Trang {page_num}]** bên dưới để đối chiếu nội dung gốc trong bài giảng."
            )

        return {
            "answer": answer_text,
            "sources": sources,
            "recommended_doc": doc,
            "confidence": "HIGH" if (top and top["score"] > 0.25) else "MEDIUM",
            "is_out_of_domain": False,
            "mode": "SEMANTIC_OFFLINE",
            "suggested_questions": self._generate_followups(doc)
        }

    def _generate_followups(self, doc: Optional[Dict[str, Any]]) -> List[str]:
        if not doc:
            return ["Bạn có muốn tìm tài liệu về môn học khác?"]
        cname = doc.get("course_name", "")
        if "Giải tích" in cname or "Toán" in cname:
            return [
                "Định nghĩa giới hạn hàm số và tính liên tục?",
                "Định lý Lagrange và công thức tính tích phân suy rộng?",
                "Cách giải bài tập chuỗi số và chuỗi hàm?"
            ]
        elif "Kinh tế" in cname:
            return [
                "Độ co giãn của cầu theo giá được tính như thế nào?",
                "Thế nào là thị trường cạnh tranh hoàn hảo?",
                "Chi phí cận biên (MC) và tối đa hóa lợi nhuận?"
            ]
        elif "Lập trình" in cname or "OOP" in cname:
            return [
                "Sự khác biệt giữa Đa hình (Polymorphism) và Kế thừa (Inheritance)?",
                "Ý nghĩa của tính đóng gói (Encapsulation) trong Java?",
                "Interface khác Abstract Class ở những điểm nào?"
            ]
        elif "An toàn" in cname or "Bảo mật" in cname:
            return [
                "Nguyên lý hoạt động của mã hóa khóa công khai RSA?",
                "Chữ ký số điện tử đảm bảo tính toàn vẹn như thế nào?",
                "Các hình thức tấn công Man-in-the-Middle phổ biến?"
            ]
        else:
            return [
                f"Tài liệu {doc['title']} gồm những chương nào?",
                f"Có bài tập thực hành nào cho môn {cname} không?"
            ]


# ==============================================================================
# 2. GIAO DIỆN WEB HTML / CSS / JAVASCRIPT (TIẾNG VIỆT CÓ DẤU 100%)
# ==============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeDocu AI Scholar - Hệ Thống RAG & Gợi Ý Tài Liệu Học Thuật</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --border: #475569;
            --success: #10b981;
            --warning: #f59e0b;
            --gold: #fbbf24;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-primary); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

        /* Top Header */
        header { background: var(--bg-secondary); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
        .logo-area { display: flex; align-items: center; gap: 12px; }
        .badge { background: rgba(56, 189, 248, 0.2); color: var(--accent); padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; text-transform: uppercase; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); display: inline-block; margin-right: 6px; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        /* Layout */
        .workspace { display: flex; flex: 1; height: calc(100vh - 61px); }

        /* Sidebar */
        .sidebar { width: 360px; background: var(--bg-secondary); border-right: 1px solid var(--border); padding: 18px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
        .panel-title { font-size: 12px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
        .select-input { width: 100%; padding: 10px 12px; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px; color: #fff; font-size: 13px; outline: none; }
        .select-input:focus { border-color: var(--accent); }

        .card { background: var(--bg-card); border-radius: 8px; padding: 12px; border: 1px solid var(--border); }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-top: 8px; }
        .stat-box { background: rgba(15, 23, 42, 0.4); padding: 8px 4px; border-radius: 6px; text-align: center; }
        .stat-val { font-size: 16px; font-weight: 700; color: var(--accent); }
        .stat-label { font-size: 10px; color: var(--text-secondary); }

        .api-input { width: 100%; padding: 8px 12px; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px; color: #fff; font-size: 13px; outline: none; }
        .api-input:focus { border-color: var(--accent); }
        .hint-text { font-size: 11px; color: var(--text-secondary); margin-top: 6px; line-height: 1.4; }

        /* Chat Panel */
        .chat-container { flex: 1; display: flex; flex-direction: column; background: var(--bg-primary); position: relative; }
        .messages-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 18px; }

        .msg { display: flex; gap: 12px; max-width: 85%; }
        .msg.user { align-self: flex-end; flex-direction: row-reverse; }
        .msg.ai { align-self: flex-start; }

        .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; flex-shrink: 0; }
        .msg.user .avatar { background: var(--accent); color: #000; }
        .msg.ai .avatar { background: #6366f1; color: #fff; }

        .msg-body { background: var(--bg-secondary); padding: 14px 18px; border-radius: 12px; border: 1px solid var(--border); font-size: 14px; line-height: 1.7; color: #e2e8f0; position: relative; }
        .msg.user .msg-body { background: #0284c7; color: #fff; border: none; }

        /* [TỐI ƯU 9] Markdown Rendering */
        .msg-body strong { font-weight: 700; color: #f1f5f9; }
        .msg.user .msg-body strong { color: #fff; }
        .msg-body em { font-style: italic; color: #cbd5e1; }
        .msg-body blockquote { border-left: 3px solid var(--accent); margin: 8px 0; padding: 6px 12px; background: rgba(56,189,248,0.07); border-radius: 0 6px 6px 0; color: #94a3b8; font-style: italic; }
        .msg-body ul, .msg-body ol { padding-left: 20px; margin: 6px 0; }
        .msg-body li { margin-bottom: 4px; }
        .msg-body code { background: rgba(99,102,241,0.2); color: #a5b4fc; padding: 1px 5px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 12px; }
        .msg-body h1, .msg-body h2, .msg-body h3 { margin: 10px 0 4px; color: #f1f5f9; }
        .msg-body h1 { font-size: 17px; } .msg-body h2 { font-size: 15px; } .msg-body h3 { font-size: 14px; }
        .msg-body p { margin: 4px 0; }
        .msg-body hr { border: none; border-top: 1px solid var(--border); margin: 10px 0; }

        /* [TỐI ƯU 8] Latency Badge */
        .latency-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 10px; color: var(--text-secondary); margin-top: 8px; padding: 3px 8px; background: rgba(15,23,42,0.5); border-radius: 999px; border: 1px solid var(--border); }
        .latency-badge .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
        .dot-gemini { background: #10b981; } .dot-offline { background: #f59e0b; } .dot-greeting { background: #6366f1; }

        /* [TỐI ƯU 12] Copy Button */
        .copy-btn { position: absolute; top: 10px; right: 10px; background: rgba(71,85,105,0.7); border: 1px solid var(--border); color: var(--text-secondary); font-size: 11px; padding: 3px 8px; border-radius: 5px; cursor: pointer; transition: all 0.2s; opacity: 0; }
        .msg-body:hover .copy-btn { opacity: 1; }
        .copy-btn:hover { background: var(--bg-card); color: #fff; }

        /* Recommended Document Card */
        .recommend-card { background: rgba(30, 41, 59, 0.9); border: 1px solid var(--accent); border-radius: 8px; padding: 12px 14px; margin-top: 12px; }
        .recommend-badge { font-size: 11px; font-weight: 700; color: var(--gold); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .recommend-title { font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 4px; }
        .recommend-meta { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; }
        .recommend-btn { background: rgba(56, 189, 248, 0.2); border: 1px solid var(--accent); color: var(--accent); font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 4px; cursor: pointer; transition: all 0.2s; }
        .recommend-btn:hover { background: var(--accent); color: #000; }

        /* [TỐI ƯU 10] Sources Card với Citation Popover */
        .sources-area { margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--border); }
        .sources-title { font-size: 11px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 6px; }
        .source-tag { display: inline-flex; align-items: center; gap: 4px; background: rgba(56, 189, 248, 0.12); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 8px; border-radius: 6px; font-size: 12px; margin-right: 6px; margin-bottom: 6px; cursor: pointer; transition: all 0.2s; position: relative; }
        .source-tag:hover { background: rgba(56, 189, 248, 0.25); }
        .citation-popover { display: none; position: absolute; bottom: calc(100% + 8px); left: 0; min-width: 280px; max-width: 340px; background: #1e293b; border: 1px solid var(--accent); border-radius: 8px; padding: 10px 12px; z-index: 999; box-shadow: 0 8px 24px rgba(0,0,0,0.5); font-size: 12px; line-height: 1.5; color: #cbd5e1; white-space: normal; }
        .citation-popover .pop-title { font-weight: 700; color: var(--accent); margin-bottom: 6px; font-size: 11px; text-transform: uppercase; }
        .citation-popover .pop-snippet { font-style: italic; color: #94a3b8; }
        .source-tag:hover .citation-popover { display: block; }

        /* Suggestions */
        .suggestions { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 20px 10px 20px; }
        .sugg-chip { background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-secondary); font-size: 12px; padding: 6px 12px; border-radius: 999px; cursor: pointer; transition: all 0.2s; }
        .sugg-chip:hover { color: var(--accent); border-color: var(--accent); background: var(--bg-card); }

        /* Out-of-domain Warning Card */
        .ood-card { background: rgba(245, 158, 11, 0.12); border: 1px solid var(--warning); border-radius: 8px; padding: 12px 14px; margin-top: 12px; }
        .ood-badge { font-size: 11px; font-weight: 700; color: var(--warning); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .ood-title { font-size: 13px; color: #fde68a; line-height: 1.4; margin-bottom: 8px; }
        .ood-btn { background: rgba(245, 158, 11, 0.25); border: 1px solid var(--warning); color: #fbbf24; font-size: 11px; font-weight: 600; padding: 6px 12px; border-radius: 4px; cursor: pointer; transition: all 0.2s; }
        .ood-btn:hover { background: var(--warning); color: #000; }

        /* Input Bar */
        .input-bar { padding: 14px 20px; background: var(--bg-secondary); border-top: 1px solid var(--border); display: flex; gap: 10px; }
        .chat-input { flex: 1; padding: 12px 14px; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 8px; color: #fff; font-size: 14px; outline: none; transition: border-color 0.2s; }
        .chat-input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(56,189,248,0.15); }
        .send-btn { background: var(--accent); color: #000; font-weight: 600; padding: 0 20px; border-radius: 8px; border: none; cursor: pointer; font-size: 14px; transition: background 0.2s; }
        .send-btn:hover { background: var(--accent-hover); color: #fff; }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        /* [TỐI ƯU] Loading dots animation */
        .loading-dots span { animation: blink 1.2s infinite; font-size: 18px; }
        .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
        .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink { 0%, 80%, 100% { opacity: 0; } 40% { opacity: 1; } }
    </style>
</head>
<body>

    <header>
        <div class="logo-area">
            <h2 style="font-size: 17px; font-weight: 700; color: #fff;">DeDocu AI Scholar</h2>
            <span class="badge">RAG Đa Môn Học & Gợi Ý Giáo Trình</span>
        </div>
        <div style="font-size: 12px; color: var(--text-secondary);">
            <span class="status-dot"></span>Kho tri thức đại học đã sẵn sàng
        </div>
    </header>

    <div class="workspace">
        <!-- Sidebar -->
        <div class="sidebar">
            <div>
                <div class="panel-title">Phạm vi tìm kiếm & Đọc tài liệu</div>
                <select id="docScopeSelect" class="select-input" onchange="onScopeChanged()">
                    <option value="">[Tất cả kho tài liệu DeDocu] (Tìm đa môn học)</option>
                </select>
            </div>

            <div>
                <div class="panel-title">Thống kê Kho Tri Thức</div>
                <div class="card">
                    <div style="font-size: 13px; font-weight: 600; color: #fff;" id="activeDocTitle">Chế độ tìm kiếm toàn sàn DeDocu</div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-val" id="statDocsCount">--</div>
                            <div class="stat-label">Tài liệu</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" id="statChunksCount">--</div>
                            <div class="stat-label">Vector Chunks</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" id="statPagesCount">--</div>
                            <div class="stat-label">Trang lập chỉ mục</div>
                        </div>
                    </div>
                </div>
            </div>

            <div>
                <div class="panel-title">Cấu hình Gemini API (Tùy chọn)</div>
                <input type="password" id="geminiApiKey" class="api-input" placeholder="Dán Gemini API Key vào đây..." />
                <div class="hint-text">
                    * Để trống để chạy <strong>Offline Semantic Mode</strong> (hoàn toàn miễn phí, không cần internet).<br/>
                    * Nhập key để Gemini 1.5 Flash tổng hợp văn bản trực tiếp.
                </div>
            </div>

            <div style="margin-top: auto; padding-top: 12px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-secondary); line-height: 1.5;">
                Đề tài: StuDocu Clone (DeDocu Commons)<br/>
                Kiến trúc: PyMuPDF + Chunking + Cosine Vector Index + Recommender + RAG
            </div>
        </div>

        <!-- Chat Area -->
        <div class="chat-container">
            <div class="messages-box" id="messagesBox">
                <div class="msg ai">
                    <div class="avatar">AI</div>
                    <div class="msg-body">
                        Xin chào bạn! Tôi là <strong>Trợ lý Học thuật DeDocu AI Scholar</strong>.<br/>
                        Hệ thống đã nạp sẵn toàn bộ 106 tài liệu giáo trình đại học (Toán học, Kinh tế, Công nghệ thông tin, An toàn thông tin, Hóa học, Vật lý, Y Dược...).<br/><br/>
                        Bạn có thể hỏi bất kỳ câu hỏi nào (ví dụ: <em>"cách học giỏi giải tích"</em>, <em>"tính đa hình trong oop"</em>, <em>"thị trường cạnh tranh hoàn hảo"</em>), tôi sẽ vừa đề xuất giáo trình trong kho DeDocu, vừa trả lời có dẫn chứng số trang chính xác!
                    </div>
                </div>
            </div>

            <!-- Suggestion Chips -->
            <div class="suggestions" id="suggestionsArea">
                <div class="sugg-chip" onclick="askQuick('Cách học giỏi Giải tích 1?')">Cách học giỏi Giải tích 1?</div>
                <div class="sugg-chip" onclick="askQuick('Tôi đang muốn tìm hiểu về toán, giới thiệu cho tôi môn học phù hợp')">Tìm hiểu về môn Toán</div>
                <div class="sugg-chip" onclick="askQuick('Đối tượng nghiên cứu của Kinh tế Vi mô là gì?')">Kinh tế Vi mô nghiên cứu gì?</div>
                <div class="sugg-chip" onclick="askQuick('Tính đa hình và kế thừa trong lập trình OOP là gì?')">Đa hình và kế thừa trong OOP</div>
                <div class="sugg-chip" onclick="askQuick('Nguyên lý mã hóa khóa công khai RSA?')">Mã hóa RSA trong An toàn TT</div>
                <div class="sugg-chip" onclick="askQuick('Mô hình RAG kết hợp Retrieval với Generation như thế nào?')">Mô hình RAG Paper 2020</div>
            </div>

            <!-- Input Bar -->
            <div class="input-bar">
                <input type="text" id="userInput" class="chat-input" placeholder="Nhập câu hỏi của bạn về bất kỳ môn học nào..." onkeypress="handleKeyPress(event)" />
                <button id="sendBtn" class="send-btn" onclick="sendQuestion()">Gửi</button>
            </div>
        </div>
    </div>

    <script>
        // =====================================================================
        // [TỐI ƯU 11] Lưu / Phục hồi API Key từ localStorage (tự nhớ sau F5)
        // =====================================================================
        const API_KEY_STORAGE = 'dedocu_gemini_api_key';
        const savedKey = localStorage.getItem(API_KEY_STORAGE);
        if (savedKey) document.getElementById('geminiApiKey').value = savedKey;
        document.getElementById('geminiApiKey').addEventListener('input', function() {
            localStorage.setItem(API_KEY_STORAGE, this.value.trim());
        });

        let catalogDocuments = [];

        window.onload = function() {
            loadCatalogInfo();
        };

        function loadCatalogInfo() {
            fetch('/api/info')
                .then(r => r.json())
                .then(data => {
                    catalogDocuments = data.documents || [];
                    document.getElementById('statDocsCount').innerText = data.total_documents || 0;
                    document.getElementById('statChunksCount').innerText = data.total_chunks || 0;
                    let totalP = 0;
                    catalogDocuments.forEach(d => totalP += (d.indexed_pages || 0));
                    document.getElementById('statPagesCount').innerText = totalP;
                    const select = document.getElementById('docScopeSelect');
                    select.innerHTML = '<option value="">[T\u1ea5t c\u1ea3 kho t\u00e0i li\u1ec7u DeDocu] (T\u00ecm \u0111a m\u00f4n h\u1ecdc)</option>';
                    catalogDocuments.forEach(d => {
                        const opt = document.createElement('option');
                        opt.value = d.document_id;
                        opt.innerText = `[${d.course_name}] ${d.title}`;
                        select.appendChild(opt);
                    });
                })
                .catch(err => console.error("L\u1ed7i t\u1ea3i danh m\u1ee5c:", err));
        }

        function onScopeChanged() {
            const select = document.getElementById('docScopeSelect');
            const val = select.value;
            const titleEl = document.getElementById('activeDocTitle');
            if (!val) {
                titleEl.innerText = "Ch\u1ebf \u0111\u1ed9 t\u00ecm ki\u1ebfm to\u00e0n s\u00e0n DeDocu";
            } else {
                const found = catalogDocuments.find(d => d.document_id === val);
                if (found) titleEl.innerText = `[${found.course_name}] ${found.title}`;
            }
        }

        function filterToDoc(docId) {
            document.getElementById('docScopeSelect').value = docId;
            onScopeChanged();
        }

        function askQuick(text) {
            document.getElementById('userInput').value = text;
            sendQuestion();
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendQuestion();
        }

        function sendQuestion() {
            const input = document.getElementById('userInput');
            const q = input.value.trim();
            if (!q) return;
            appendMessage('user', q);
            input.value = '';
            const loadingId = appendLoading();
            document.getElementById('sendBtn').disabled = true;
            const apiKey = document.getElementById('geminiApiKey').value.trim();
            const filterDocId = document.getElementById('docScopeSelect').value;
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, filter_doc_id: filterDocId, api_key: apiKey })
            })
            .then(r => r.json())
            .then(data => {
                removeLoading(loadingId);
                appendAiMessage(data);
                updateSuggestions(data.suggested_questions);
                document.getElementById('sendBtn').disabled = false;
            })
            .catch(err => {
                removeLoading(loadingId);
                appendMessage('ai', 'L\u1ed7i k\u1ebft n\u1ed1i: ' + err);
                document.getElementById('sendBtn').disabled = false;
            });
        }

        function appendMessage(sender, text) {
            const box = document.getElementById('messagesBox');
            const div = document.createElement('div');
            div.className = 'msg ' + sender;
            div.innerHTML = `<div class="avatar">${sender === 'user' ? 'ME' : 'AI'}</div><div class="msg-body">${escapeHtml(text)}</div>`;
            box.appendChild(div);
            box.scrollTop = box.scrollHeight;
        }

        function appendLoading() {
            const box = document.getElementById('messagesBox');
            const id = 'loading_' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = 'msg ai';
            // [TỐI ƯU] Loading dots animation thay vì text tĩnh
            div.innerHTML = `<div class="avatar">AI</div><div class="msg-body" style="color:var(--text-secondary);">Đang t\u00ecm ki\u1ebfm trong kho gi\u00e1o tr\u00ecnh <span class="loading-dots"><span>.</span><span>.</span><span>.</span></span></div>`;
            box.appendChild(div);
            box.scrollTop = box.scrollHeight;
            return id;
        }

        function removeLoading(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        // =====================================================================
        // [TỐI ƯU 9] Render Markdown thực thụ (Bold, Italic, Quote, List, Code, HR)
        // =====================================================================
        function renderMarkdown(text) {
            if (!text) return '';
            let html = escapeHtml(text);
            html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
            html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
            html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
            html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
            html = html.replace(/^---$/gm, '<hr/>');
            // Unordered list items
            html = html.replace(/^[*\-] (.+)$/gm, '<li>$1</li>');
            // Wrap consecutive <li> in <ul>
            html = html.replace(/(<li>[\s\S]*?<\/li>)(\s*<li>)/g, '$1$2');
            html = html.replace(/(<li>)/g, (m, p, offset, str) => {
                const before = str.slice(0, offset);
                return before.endsWith('</li>') || before.endsWith('<ul>') ? m : '<ul>' + m;
            });
            html = html.replace(/(<\/li>)(?!\s*<li>)/g, '$1</ul>');
            html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
            html = html.replace(/`(.+?)`/g, '<code>$1</code>');
            html = html.replace(/\n/g, '<br/>');
            return html;
        }

        function appendAiMessage(data) {
            const box = document.getElementById('messagesBox');
            const div = document.createElement('div');
            div.className = 'msg ai';

            // 1. Recommended Document Card
            let recommendHtml = '';
            if (data.recommended_doc) {
                const doc = data.recommended_doc;
                recommendHtml = `<div class="recommend-card">
                    <div class="recommend-badge">[T\u00c0I LI\u1ec6U \u0110\u1ec0 XU\u1ea4T T\u1eea TH\u01af VI\u1ec6N DEDOCU]</div>
                    <div class="recommend-title">${escapeHtml(doc.title)}</div>
                    <div class="recommend-meta">M\u00f4n h\u1ecdc: <strong>${escapeHtml(doc.course_name)}</strong> | Khoa: ${escapeHtml(doc.faculty)} | Tr\u01b0\u1eddng: ${escapeHtml(doc.university)}</div>
                    <button class="recommend-btn" onclick="filterToDoc('${doc.document_id}')">L\u1ecdc ch\u1ec9 h\u1ecfi \u0111\u00e1p trong t\u00e0i li\u1ec7u n\u00e0y</button>
                </div>`;
            }

            // 2. [TỐI ƯU 10] Citation Preview Popover trong source tags
            let sourcesHtml = '';
            if (data.sources && data.sources.length > 0) {
                sourcesHtml = '<div class="sources-area"><div class="sources-title">NGU\u1ed2N TR\u00cdCH D\u1eaaN T\u1eea TRANG S\u00c1CH:</div>';
                data.sources.forEach(s => {
                    const safeSnippet = escapeHtml(s.quote_snippet || '');
                    const safeTitle = escapeHtml(s.document_title || '');
                    sourcesHtml += `<div class="source-tag">
                        [Trang ${s.page_number}] ${escapeHtml(s.course_name)} (Kh\u1edbp ${Math.round(s.score * 100)}%)
                        <div class="citation-popover">
                            <div class="pop-title">${safeTitle} &mdash; Trang ${s.page_number}</div>
                            <div class="pop-snippet">&ldquo;${safeSnippet}&rdquo;</div>
                        </div>
                    </div>`;
                });
                sourcesHtml += '</div>';
            }

            // 3. Out-of-Domain Notice
            let oodHtml = '';
            if (data.is_out_of_domain) {
                oodHtml = `<div class="ood-card">
                    <div class="ood-badge">[TH\u00d4NG B\u00c1O: CH\u1ee6 \u0110\u1ec0 NGO\u00c0I PH\u1ea0M VI TH\u01af VI\u1ec6N HI\u1ec6N T\u1ea0I]</div>
                    <div class="ood-title">Ch\u01b0a c\u00f3 gi\u00e1o tr\u00ecnh cho ch\u1ee7 \u0111\u1ec1 n\u00e0y trong kho DeDocu. H\u00e3y t\u1ea3i l\u00ean t\u00e0i li\u1ec7u \u0111\u1ec3 nh\u1eadn +20 \u0111i\u1ec3m th\u01b0\u1edfng m\u1edf kh\u00f3a!</div>
                    <button class="ood-btn" onclick="alert('M\u1edf c\u1eeda s\u1ed5 Upload: T\u1ea3i l\u00ean t\u00e0i li\u1ec7u m\u00f4n n\u00e0y \u0111\u1ec3 nh\u1eadn +20 \u0110i\u1ec3m th\u01b0\u1edfng m\u1edf kh\u00f3a t\u00e0i li\u1ec7u VIP!')">T\u1ea3i l\u00ean t\u00e0i li\u1ec7u m\u00f4n h\u1ecdc n\u00e0y (+20 \u0110i\u1ec3m)</button>
                </div>`;
            }

            // 4. [TỐI ƯU 9] Render Markdown
            const formattedAnswer = renderMarkdown(data.answer || '');

            // 5. [TỐI ƯU 8] Latency Badge
            const mode = data.mode || 'SEMANTIC_OFFLINE';
            const latency = data.latency_ms != null ? data.latency_ms : '?';
            let dotClass = 'dot-offline', modeLabel = 'Offline Semantic';
            if (mode === 'GEMINI_LIVE') { dotClass = 'dot-gemini'; modeLabel = 'Gemini 1.5 Flash'; }
            else if (mode === 'GEMINI_LIVE_OOD') { dotClass = 'dot-gemini'; modeLabel = 'Gemini (OOD)'; }
            else if (mode === 'GREETING') { dotClass = 'dot-greeting'; modeLabel = 'Greeting'; }
            const latencyHtml = `<div class="latency-badge"><span class="dot ${dotClass}"></span>${latency}ms &bull; ${modeLabel}</div>`;

            // 6. [TỐI ƯU 12] Copy Button
            const rawAnswer = data.answer || '';
            const copyBtn = `<button class="copy-btn" onclick="copyAnswer(this, ${JSON.stringify(rawAnswer)})">Sao ch\u00e9p</button>`;

            div.innerHTML = `<div class="avatar">AI</div><div class="msg-body">${copyBtn}<div>${formattedAnswer}</div>${oodHtml}${recommendHtml}${sourcesHtml}${latencyHtml}</div>`;
            box.appendChild(div);
            box.scrollTop = box.scrollHeight;
        }

        function updateSuggestions(questions) {
            if (!questions || questions.length === 0) return;
            const area = document.getElementById('suggestionsArea');
            area.innerHTML = '';
            questions.forEach(q => {
                const chip = document.createElement('div');
                chip.className = 'sugg-chip';
                chip.innerText = q;
                chip.onclick = function() { askQuick(q); };
                area.appendChild(chip);
            });
        }

        // [TỐI ƯU 12] Sao chép câu trả lời AI vào clipboard
        function copyAnswer(btn, text) {
            navigator.clipboard.writeText(text).then(() => {
                btn.innerText = '\u0110\u00e3 sao ch\u00e9p!';
                setTimeout(() => { btn.innerText = 'Sao ch\u00e9p'; }, 2000);
            }).catch(() => { btn.innerText = 'L\u1ed7i!'; });
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


# ==============================================================================
# 3. THREADING HTTP SERVER (ĐA LUỒNG, PHỤC VỤ NHIỀU KẾT NỐI ĐỒNG THỜI)
# ==============================================================================

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """
    [TỐI ƯU 5] ThreadingHTTPServer: Xử lý mỗi request trong một thread riêng.
    Khi một câu hỏi đang gọi Gemini API (mất 2-5 giây), server KHÔNG bị block.
    Người dùng khác vẫn load được trang chủ và gọi /api/info bình thường.
    """
    daemon_threads = True


class MultiDocRAGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_cors_headers(self):
        """[TỐI ƯU 7] CORS Headers để Frontend React/Next.js có thể gọi API trực tiếp."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """Xử lý preflight CORS request từ browser hiện đại."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            resp = {
                "total_documents": len(engine.documents),
                "total_chunks": len(engine.chunks),
                "documents": engine.documents
            }
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/chat":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                payload = json.loads(body)
                question = payload.get("question", "").strip()
                filter_doc_id = payload.get("filter_doc_id", "").strip() or None
                api_key = payload.get("api_key", "").strip() or os.getenv("GEMINI_API_KEY")

                result = engine.answer_question(question, filter_doc_id=filter_doc_id, api_key=api_key)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()


# ==============================================================================
# 4. KHỚI TẠO VÀ CHẠY SERVER
# ==============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
dataset_base = os.path.join(project_root, "preProject", "dataset", "dataset")
index_file = os.path.join(script_dir, "knowledge_base_index.json")

engine = MultiDocumentRAGEngine(index_file)


def find_free_port(start_port: int = 8088, max_tries: int = 10) -> int:
    """
    [TỐI ƯU 6] Tự động tìm cổng trống.
    Nếu 8088 đang bị chiếm (do server cũ chưa tắt hẳn), thử 8089, 8090...
    Tránh lỗi 'OSError: Address already in use' khi chạy lại sau Ctrl+C.
    """
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def main():
    print("----------------------------------------------------------------------")
    print("KHỚI ĐỘNG DEDOCU AI SCHOLAR - MULTI-DOC RAG & RECOMMENDER (SCRUM-7)")
    print("----------------------------------------------------------------------")

    engine.load_or_build_index(dataset_base, api_key=os.getenv("GEMINI_API_KEY"))
    print(f">> Sẵn sàng: {len(engine.documents)} tài liệu giáo trình, {len(engine.chunks)} vector chunks.")

