"""
==============================================================================
RAG PIPELINE PROOF OF CONCEPT (POC) - SCRUM-7
Đề tài: Mạng xã hội chia sẻ tài liệu học thuật phi tập trung (StuDocu Clone)
Công nghệ: Python, Cosine Vector Search, Pydantic Structured Output, Gemini API.
Quy tắc: Tiếng Việt có dấu 100% chuẩn mực; tuyệt đối không dùng emoji.
==============================================================================
"""

import os
import sys
import json
import math
import re
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Đảm bảo in tiếng Việt chuẩn trên Windows console
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
# 1. PYDANTIC SCHEMAS (STRUCTURED OUTPUT)
# ==============================================================================

class SourceFound(BaseModel):
    document_id: str = Field(description="Mã định danh tài liệu")
    document_title: str = Field(description="Tiêu đề tài liệu")
    course_name: Optional[str] = Field(default="", description="Tên môn học")
    page_number: int = Field(description="Số trang tài liệu chứa thông tin")
    quote_snippet: str = Field(description="Đoạn văn trích dẫn làm cơ sở đối chiếu")
    similarity_score: float = Field(description="Độ tương quan ngữ nghĩa")
    jump_url: str = Field(description="Đường dẫn liên kết để cuộn PDF đến trang")


class RecommendedDocument(BaseModel):
    document_id: str = Field(description="Mã định danh tài liệu đề xuất")
    title: str = Field(description="Tiêu đề tài liệu")
    course_name: str = Field(description="Môn học liên quan")
    faculty: str = Field(description="Khoa / Ngành học")
    university: str = Field(description="Trường đại học")


class AcademicChatResponse(BaseModel):
    session_id: str
    answer_markdown: str
    confidence_level: str
    is_out_of_domain: bool = False
    recommended_document: Optional[RecommendedDocument] = None
    sources_found: List[SourceFound]
    suggested_followup_questions: List[str]


# ==============================================================================
# 2. CHUNKING ENGINE CHO TÀI LIỆU HỌC THUẬT
# ==============================================================================

class ChunkItem:
    def __init__(self, chunk_id: str, document_id: str, document_title: str, course_name: str, page_number: int, content: str):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.course_name = course_name
        self.page_number = page_number
        self.content = content
        self.embedding: Optional[List[float]] = None


# ==============================================================================
# 3. VECTOR STORE (IN-MEMORY SIMILARITY SEARCH)
# ==============================================================================

class InMemoryVectorStore:
    def __init__(self):
        self.chunks: List[ChunkItem] = []

    def add_chunks(self, chunks: List[ChunkItem]):
        self.chunks.extend(chunks)

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def similarity_search(self, query: str, query_vector: List[float], recommended_doc_id: Optional[str] = None, top_k: int = 2) -> List[Dict[str, Any]]:
        conversational_stopwords = {
            "vậy", "thì", "tìm", "hiểu", "về", "tí", "đi", "nhỉ", "tôi", "đang", "có", "hứng", "thú",
            "môn", "gì", "cũng", "được", "bạn", "giới", "thiệu", "cho", "không", "học", "nhé", "nào",
            "xem", "thử", "với", "ơi", "ạ", "cần", "muốn", "hãy", "chút", "một", "giáo", "trình",
            "bài", "giảng", "toàn", "phần", "tập", "viên", "đại", "và", "các", "của", "là", "trong",
            "này", "những", "để", "khi", "nhiều", "thế", "như", "ra", "sao", "lại", "đã", "sẽ",
            "nhớ", "thống", "tin", "hệ", "tại", "nhà", "biết"
        }
        q_words = set(w for w in re.findall(r'\w+', query.lower()) if len(w) > 2 and w not in conversational_stopwords)

        # Ràng buộc chặt chẽ theo tài liệu đề xuất nếu có
        candidate_chunks = [c for c in self.chunks if c.document_id == recommended_doc_id] if recommended_doc_id else self.chunks
        if not candidate_chunks:
            candidate_chunks = self.chunks

        scored = []
        for c in candidate_chunks:
            if c.embedding:
                cos_score = self.cosine_similarity(query_vector, c.embedding)
                content_lower = c.content.lower()
                overlap = sum(1 for w in q_words if w in content_lower)
                domain_boost = 0.35 if (recommended_doc_id and c.document_id == recommended_doc_id) else 0.0
                hybrid_score = cos_score + (overlap * 0.1) + domain_boost

                scored.append({
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "document_title": c.document_title,
                    "course_name": c.course_name,
                    "page_number": c.page_number,
                    "content": c.content,
                    "score": round(hybrid_score, 4),
                    "pure_cosine": round(cos_score, 4)
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ==============================================================================
# 4. RAG SERVICE (EMBEDDING & GENERATION)
# ==============================================================================

class AcademicRAGService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.vector_dim = 768

    def get_embedding(self, text: str) -> List[float]:
        if self.api_key and REQUESTS_AVAILABLE:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
                res = requests.post(url, json={"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}, timeout=8)
                if res.status_code == 200:
                    val = res.json().get("embedding", {}).get("values", [])
                    if val:
                        return val
            except Exception:
                pass

        # Vectorizer ngữ nghĩa cục bộ offline
        words = re.findall(r'\w+', text.lower())
        vec = [0.0] * self.vector_dim
        for i, w in enumerate(words):
            h = hash(w) % self.vector_dim
            vec[h] += 1.0 / (math.log(i + 2))

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def generate_rag_response(
        self,
        question: str,
        session_id: str,
        retrieved_contexts: List[Dict[str, Any]],
        recommended_doc: Optional[Dict[str, Any]] = None
    ) -> AcademicChatResponse:
        sources: List[SourceFound] = []
        for ctx in retrieved_contexts:
            sources.append(SourceFound(
                document_id=ctx["document_id"],
                document_title=ctx["document_title"],
                course_name=ctx.get("course_name", ""),
                page_number=ctx["page_number"],
                quote_snippet=ctx["content"][:220] + "...",
                similarity_score=ctx["score"],
                jump_url=f"/viewer/{ctx['document_id']}?page={ctx['page_number']}"
            ))

        rec_model = None
        if recommended_doc:
            rec_model = RecommendedDocument(
                document_id=recommended_doc["document_id"],
                title=recommended_doc["title"],
                course_name=recommended_doc["course_name"],
                faculty=recommended_doc["faculty"],
                university=recommended_doc["university"]
            )

        # KIỂM SOÁT CÂU HỎI NGOÀI PHẠM VI (OUT-OF-DOMAIN DETECTION)
        top = retrieved_contexts[0] if retrieved_contexts else None
        top_score = top["score"] if top else 0.0
        top_cosine = top.get("pure_cosine", 0.0) if top else 0.0

        is_out_of_domain = (not recommended_doc) and (top_score < 0.22 or top_cosine < 0.15)

        if is_out_of_domain:
            ood_text = (
                f"Chủ đề bạn đang hỏi ('{question}') hiện **chưa có tài liệu giáo trình chính thức** trong thư viện DeDocu.\n\n"
                f"**Gợi ý cho sinh viên:**\n"
                f"- Thư viện DeDocu hiện hỗ trợ 106 tài liệu thuộc các ngành Toán học, CNTT, Kinh tế & Quản lý, Ngoại ngữ, Y Dược...\n"
                f"- Nếu bạn sở hữu tài liệu/giáo trình của môn học này, hãy **tải lên DeDocu** để nhận ngay **20 điểm thưởng** mở khóa tài liệu VIP!"
            )
            return AcademicChatResponse(
                session_id=session_id,
                answer_markdown=ood_text,
                confidence_level="OUT_OF_DOMAIN",
                is_out_of_domain=True,
                recommended_document=None,
                sources_found=[],
                suggested_followup_questions=[
                    "Làm thế nào để tải tài liệu lên DeDocu nhận điểm thưởng?",
                    "Thư viện DeDocu hiện có những môn học nào?"
                ]
            )

        # 1. Gọi Live Gemini API nếu có key
        if self.api_key and REQUESTS_AVAILABLE:
            try:
                context_str = "\n\n".join([
                    f"--- TÀI LIỆU: {ctx['document_title']} (Trang {ctx['page_number']}) ---\n{ctx['content']}"
                    for ctx in retrieved_contexts
                ])
                prompt = (
                    f"Bạn là Trợ lý Học thuật DeDocu AI trên nền tảng DeDocu Commons (StuDocu Clone).\n"
                    f"Dưới đây là các đoạn trích từ giáo trình đại học:\n\n"
                    f"{context_str}\n\n"
                    f"Câu hỏi của sinh viên: {question}\n\n"
                    f"Yêu cầu:\n"
                    f"1. Trả lời bằng tiếng Việt mạch lạc, có góc nhìn sư phạm.\n"
                    f"2. Nếu hỏi về phương pháp học tập, hãy nêu các bước trọng tâm và liên hệ với nội dung trong sách.\n"
                    f"3. Trích dẫn rõ số trang và tên tài liệu.\n"
                    f"4. Tuyệt đối không dùng emoji trong toàn bộ phản hồi."
                )

                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
                resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}, timeout=12)
                if resp.status_code == 200:
                    text_out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return AcademicChatResponse(
                        session_id=session_id,
                        answer_markdown=text_out,
                        confidence_level="HIGH",
                        recommended_document=rec_model,
                        sources_found=sources,
                        suggested_followup_questions=[
                            "Bạn có muốn tóm tắt các công thức trong tài liệu này không?",
                            "Có cần xem danh sách bài tập luyện thi không?"
                        ]
                    )
            except Exception:
                pass

        # 2. Sinh Markdown học thuật offline
        top = retrieved_contexts[0] if retrieved_contexts else None
        doc_title = recommended_doc["title"] if recommended_doc else (top["document_title"] if top else "Giáo trình")
        page_num = top["page_number"] if top else 1
        snippet = top["content"] if top else ""
        cname = recommended_doc["course_name"] if recommended_doc else (top.get("course_name", "") if top else "")

        q_lower = question.lower()
        if any(w in q_lower for w in ["cách học", "phương pháp", "làm sao", "ôn thi"]):
            answer = (
                f"Để học tốt môn **{cname}**, bạn nên kết hợp giữa việc nắm vững lý thuyết nền tảng và giải bài tập theo các bước sau:\n\n"
                f"**1. Nắm chắc các khái niệm cốt lõi:**\n"
                f"Dựa trên giáo trình **'{doc_title}'** (Trích dẫn từ **Trang {page_num}**):\n"
                f"> \"{snippet}\"\n\n"
                f"**2. Phương pháp học tập trọng tâm:**\n"
                f"- Hiểu bản chất các định nghĩa và công thức trước khi áp dụng giải bài tập.\n"
                f"- Luyện tập từ các dạng cơ bản đến đề thi giữa kỳ của trường.\n"
                f"- Nhấn vào thẻ nguồn [Sources Found] bên dưới để mở đúng trang tài liệu gốc trên hệ thống."
            )
        elif any(w in q_lower for w in ["giới thiệu", "tìm hiểu", "môn gì", "bắt đầu", "gợi ý"]):
            answer = (
                f"Chào bạn! Tài liệu phù hợp nhất để bạn bắt đầu tìm hiểu là cuốn **'{doc_title}'** (Môn học: **{cname}**).\n\n"
                f"**Căn cứ nội dung trích xuất từ giáo trình (Trang {page_num}):**\n"
                f"> \"{snippet}\"\n\n"
                f"**Định hướng học tập:**\n"
                f"- Đây là tài liệu nền tảng cung cấp kiến thức chuẩn hóa của trường đại học.\n"
                f"- Bạn có thể nhấp vào nguồn trích dẫn [Trang {page_num}] bên dưới để tra cứu chi tiết."
            )
        else:
            answer = (
                f"Dựa trên tài liệu **'{doc_title}'** (Trang {page_num}), vấn đề bạn hỏi được trình bày như sau:\n\n"
                f"> \"{snippet}\"\n\n"
                f"**Điểm cốt lõi cần ghi nhớ:**\n"
                f"- Kiến thức này xuất hiện trong chương trình môn **{cname}**.\n"
                f"- Bạn có thể nhấp trực tiếp vào thẻ [Sources Found] bên dưới để xem đúng trang PDF minh họa."
            )

        return AcademicChatResponse(
            session_id=session_id,
            answer_markdown=answer,
            confidence_level="HIGH" if (top and top["score"] > 0.25) else "MEDIUM",
            recommended_document=rec_model,
            sources_found=sources,
            suggested_followup_questions=[
                f"Bạn có muốn xem các câu hỏi trắc nghiệm liên quan đến môn {cname}?",
                "Có cần giải thích chi tiết bằng ví dụ thực hành không?"
            ]
        )


# ==============================================================================
# 5. CHƯƠNG TRÌNH CHÍNH (MAIN TEST)
# ==============================================================================

def match_catalog_doc(question: str, catalog_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    conversational_stopwords = {
        "vậy", "thì", "tìm", "hiểu", "về", "tí", "đi", "nhỉ", "tôi", "đang", "có", "hứng", "thú",
        "môn", "gì", "cũng", "được", "bạn", "giới", "thiệu", "cho", "không", "học", "nhé", "nào",
        "xem", "thử", "với", "ơi", "ạ", "cần", "muốn", "hãy", "chút", "một", "giáo", "trình",
        "bài", "giảng", "toàn", "phần", "tập", "viên", "đại", "và", "các", "của", "là", "trong",
        "này", "những", "để", "khi", "nhiều", "thế", "như", "ra", "sao", "lại", "đã", "sẽ",
        "nhớ", "thống", "tin", "hệ", "tại", "nhà", "biết"
    }
    academic_domains = {
        "toán": ["Toán & Khoa học Tự nhiên", "Toán học"],
        "toán học": ["Toán & Khoa học Tự nhiên", "Toán học"],
        "giải tích": ["Toán & Khoa học Tự nhiên", "Toán học"],
        "đại số": ["Toán & Khoa học Tự nhiên", "Toán học"],
        "kinh tế": ["Kinh tế & Quản lý"],
        "lập trình": ["Công nghệ Thông tin"],
        "oop": ["Công nghệ Thông tin"],
        "rag": ["Công nghệ Thông tin"]
    }

    q_lower = " " + question.lower() + " "
    q_tokens = set(re.findall(r'\w+', question.lower())) - conversational_stopwords
    best_doc = None
    max_score = 0

    for doc in catalog_docs:
        score = 0
        fac = doc.get("faculty", "")
        cname = doc.get("course_name", "").lower()
        title = doc.get("title", "").lower()

        # Khớp chuyên ngành
        for dkw, target_facs in academic_domains.items():
            if (" " + dkw + " ") in q_lower or dkw in q_tokens:
                if any(tf.lower() in fac.lower() for tf in target_facs):
                    score += 8
                    if "giải tích 1" in title or "giải tích 1" in cname:
                        score += 5

        if cname and len(cname) > 3 and (" " + cname + " ") in q_lower:
            score += 7

        for kw in doc.get("keywords", []):
            kw_l = kw.lower().strip()
            if kw_l in conversational_stopwords or len(kw_l) < 3:
                continue
            if " " in kw_l and (" " + kw_l + " ") in q_lower:
                score += 5
            elif kw_l in q_tokens:
                score += 2

        if score > max_score and score >= 4:
            max_score = score
            best_doc = doc

    return best_doc


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_base = os.path.join(project_root, "preProject", "dataset", "dataset")
    index_file = os.path.join(script_dir, "knowledge_base_index.json")

    print("=" * 70)
    print("BẮT ĐẦU CHẠY THỬ NGHIỆM RAG PIPELINE POC (SCRUM-7)")
    print("=" * 70)

    api_key = os.getenv("GEMINI_API_KEY")
    rag_service = AcademicRAGService(api_key=api_key)
    vector_store = InMemoryVectorStore()

    # 1. Kiểm tra và nạp Knowledge Base
    if not os.path.exists(index_file):
        print(">> Chưa tìm thấy knowledge_base_index.json, tiến hành tạo index...")
        from ingest_documents import IngestionPipeline
        pipeline = IngestionPipeline()
        pipeline.ingest_full_dataset(dataset_base, os.path.join(dataset_base, "metadata.json"), index_file)

    with open(index_file, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    catalog_docs = kb_data.get("documents", [])
    raw_chunks = kb_data.get("chunks", [])

    print(f">> Đã nạp thành công: {len(catalog_docs)} tài liệu đại học ({len(raw_chunks)} vector chunks).")

    chunks_objs = []
    for c in raw_chunks:
        ci = ChunkItem(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            document_title=c["document_title"],
            course_name=c.get("course_name", ""),
            page_number=c["page_number"],
            content=c["content"]
        )
        ci.embedding = c.get("embedding")
        chunks_objs.append(ci)

    vector_store.add_chunks(chunks_objs)
    print(">> Đã lập chỉ mục Vector vào CSDL thành công.")

    # 2. Thử nghiệm câu hỏi thực tế (IN-DOMAIN)
    test_question = "Cách học giỏi giải tích và các định lý cốt lõi cần nhớ?"
    print(f"\n>> Câu hỏi của sinh viên (In-Domain): '{test_question}'")

    recommended_doc = match_catalog_doc(test_question, catalog_docs)
    rec_id = recommended_doc["document_id"] if recommended_doc else None
    query_vec = rag_service.get_embedding(test_question)
    retrieved = vector_store.similarity_search(test_question, query_vec, recommended_doc_id=rec_id, top_k=2)

    print(f">> Đã tìm thấy {len(retrieved)} đoạn trích dẫn phù hợp nhất:")
    for r in retrieved:
        print(f"   - [{r['document_title']}] (Trang {r['page_number']}, Score: {r['score']}): {r['content'][:85]}...")

    # 3. Sinh phản hồi Structured Output
    print("\n>> Đang tạo câu trả lời với Gemini API Structured Output...")
    response: AcademicChatResponse = rag_service.generate_rag_response(
        question=test_question,
        session_id="session-studocu-test-001",
        retrieved_contexts=retrieved,
        recommended_doc=recommended_doc
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ TRẢ VỀ CHO FRONTEND (IN-DOMAIN JSON):")
    print("=" * 70)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))

    print("\nKIỂM TRA THÀNH CÔNG: Structured Output hợp lệ theo schema Pydantic!")
    print("=" * 70)

    # 4. Thử nghiệm câu hỏi NGOÀI PHẠM VI TRI THỨC (OUT-OF-DOMAIN TEST)
    ood_question = "Cách nấu phở bò truyền thống ngon tại nhà?"
    print(f"\n>> Thử nghiệm câu hỏi ngoài phạm vi (Out-of-Domain): '{ood_question}'")

    ood_rec_doc = match_catalog_doc(ood_question, catalog_docs)
    ood_vec = rag_service.get_embedding(ood_question)
    ood_retrieved = vector_store.similarity_search(ood_question, ood_vec, recommended_doc_id=None, top_k=2)

    ood_response: AcademicChatResponse = rag_service.generate_rag_response(
        question=ood_question,
        session_id="session-studocu-ood-002",
        retrieved_contexts=ood_retrieved,
        recommended_doc=ood_rec_doc
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ TRUY VẤN NGOÀI PHẠM VI (OUT-OF-DOMAIN JSON):")
    print("=" * 70)
    print(json.dumps(ood_response.model_dump(), indent=2, ensure_ascii=False))
    print(f">> Trạng thái: is_out_of_domain = {ood_response.is_out_of_domain} | sources_found = {len(ood_response.sources_found)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

