"""
==============================================================================
DEDOCU COMMONS - MULTI-DOCUMENT INGESTION & VECTOR INDEXING (SCRUM-7)
Mục đích: Quét, bóc tách chữ, phân mảnh (chunking) và lập chỉ mục Vector
          cho toàn bộ 106 tài liệu học thuật từ metadata.json vào Knowledge Base.
Công nghệ: PyMuPDF, Cosine Semantic Vectorizer, JSON Index Store.
Quy tắc: Không sử dụng emoji trong mã nguồn và log hiển thị.
==============================================================================
"""

import os
import sys
import json
import math
import re
from typing import List, Dict, Any, Optional

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
# 1. BỘ PHÂN MẢNH VĂN BẢN (TEXT CHUNKER)
# ==============================================================================

class DocumentChunker:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_meta: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return []

        chunks = []
        start = 0
        idx = 1
        text_len = len(cleaned)

        doc_id = doc_meta.get("document_id", "DOC")
        doc_title = doc_meta.get("title", "Tài liệu học thuật")
        faculty = doc_meta.get("faculty", "Đại cương")
        course_name = doc_meta.get("course_name", doc_title)
        university = doc_meta.get("university", "Đại học")

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                last_period = cleaned.rfind('. ', start, end)
                if last_period != -1 and last_period > start + (self.chunk_size // 2):
                    end = last_period + 1

            chunk_text = cleaned[start:end].strip()
            if len(chunk_text) > 40:
                cid = f"{doc_id}_p{page_num}_c{idx}"
                chunks.append({
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "faculty": faculty,
                    "course_name": course_name,
                    "university": university,
                    "page_number": page_num,
                    "content": chunk_text
                })
                idx += 1

            if end >= text_len:
                break
            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks


# ==============================================================================
# 2. VECTORIZER & INGESTION PIPELINE
# ==============================================================================

class IngestionPipeline:
    def __init__(self, vector_dim: int = 768):
        self.vector_dim = vector_dim
        self.chunker = DocumentChunker(chunk_size=600, chunk_overlap=100)

    def compute_embedding(self, text: str, api_key: Optional[str] = None) -> List[float]:
        # 1. Gọi Gemini Embedding nếu có API Key
        if api_key and REQUESTS_AVAILABLE:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
                res = requests.post(url, json={"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}, timeout=8)
                if res.status_code == 200:
                    val = res.json().get("embedding", {}).get("values", [])
                    if val:
                        return [round(x, 6) for x in val]
            except Exception:
                pass

        # 2. Pseudo-semantic vectorizer (offline)
        words = re.findall(r'\w+', text.lower())
        vec = [0.0] * self.vector_dim
        for i, w in enumerate(words):
            h = hash(w) % self.vector_dim
            vec[h] += 1.0 / (math.log(i + 2))

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [round(x / norm, 6) for x in vec]
        return vec

    def ingest_full_dataset(
        self,
        base_dataset_dir: str,
        metadata_file_path: str,
        output_index_path: str,
        max_pages_per_doc: int = 3,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("Thư viện 'pymupdf' chưa được cài đặt. Vui lòng cài: pip install pymupdf")

        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"Không tìm thấy metadata.json tại: {metadata_file_path}")

        with open(metadata_file_path, "r", encoding="utf-8") as f:
            metadata_items = json.load(f)

        print("----------------------------------------------------------------------")
        print(f"BẮT ĐẦU NẠP VÀ LẬP CHỈ MỤC TOÀN BỘ {len(metadata_items)} TÀI LIỆU VÀO KNOWLEDGE BASE (SCRUM-7)")
        print(f"Giới hạn đọc: {max_pages_per_doc} trang/tài liệu (Tổng ~{len(metadata_items) * max_pages_per_doc} trang)")
        print("----------------------------------------------------------------------")

        indexed_documents = []
        all_chunks = []

        for idx, item in enumerate(metadata_items, start=1):
            rel_path = item.get("file_path", "")
            full_pdf_path = os.path.join(base_dataset_dir, rel_path)

            if not os.path.exists(full_pdf_path):
                print(f"[{idx}/{len(metadata_items)}] [CẢNH BÁO] Tệp không tồn tại: {rel_path}")
                continue

            doc_title = item.get("title", os.path.basename(rel_path))
            cname = item.get("course_name", item.get("faculty", "Môn học"))
            doc_id = item.get("document_id", f"DOC-{idx:03d}")

            try:
                doc = pymupdf.open(full_pdf_path)
                total_pdf_pages = len(doc)
                pages_to_read = min(max_pages_per_doc, total_pdf_pages)

                doc_chunks = []
                for p_idx in range(pages_to_read):
                    page_text = doc[p_idx].get_text("text")
                    page_chunks = self.chunker.chunk_text(page_text, item, p_idx + 1)
                    for c in page_chunks:
                        c["embedding"] = self.compute_embedding(c["content"], api_key)
                    doc_chunks.extend(page_chunks)

                doc.close()

                # Tạo danh sách từ khóa tìm kiếm phong phú từ tags + tên môn + tiêu đề
                tags = item.get("tags", [])
                title_words = [w.lower() for w in re.findall(r'\w+', doc_title) if len(w) > 2]
                course_words = [w.lower() for w in re.findall(r'\w+', cname) if len(w) > 2]
                combined_keywords = list(set([t.lower() for t in tags] + title_words + course_words))

                doc_entry = {
                    "document_id": doc_id,
                    "title": doc_title,
                    "faculty": item.get("faculty", "Chung"),
                    "course_id": item.get("course_id", ""),
                    "course_name": cname,
                    "university": item.get("university", "PTIT"),
                    "category": item.get("category", "GiaoTrinh"),
                    "keywords": combined_keywords,
                    "relative_path": rel_path,
                    "total_pages": total_pdf_pages,
                    "indexed_pages": pages_to_read,
                    "chunks_count": len(doc_chunks)
                }
                indexed_documents.append(doc_entry)
                all_chunks.extend(doc_chunks)

                if idx % 10 == 0 or idx == len(metadata_items):
                    print(f">> Tiến độ: {idx}/{len(metadata_items)} tài liệu | Tổng chunks: {len(all_chunks)}")
            except Exception as e:
                print(f"[{idx}/{len(metadata_items)}] [LỖI ĐỌC PDF] {doc_title}: {e}")

        # Lưu kết quả vào Knowledge Base Index
        output_data = {
            "version": "2.0",
            "scope": "FULL_PLATFORM_CATALOG",
            "total_documents": len(indexed_documents),
            "total_chunks": len(all_chunks),
            "documents": indexed_documents,
            "chunks": all_chunks
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_index_path)), exist_ok=True)
        with open(output_index_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print("----------------------------------------------------------------------")
        print(f"HOÀN TẤT LẬP CHỈ MỤC: {len(indexed_documents)} tài liệu, {len(all_chunks)} chunks.")
        print(f"Tệp chỉ mục đã được lưu tại: {output_index_path}")
        print("----------------------------------------------------------------------")

        return {
            "success": True,
            "total_documents": len(indexed_documents),
            "total_chunks": len(all_chunks),
            "output_file": output_index_path
        }


# ==============================================================================
# 3. CHẠY THỰC THI TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_base = os.path.join(project_root, "preProject", "dataset", "dataset")
    meta_file = os.path.join(dataset_base, "metadata.json")
    index_file = os.path.join(script_dir, "knowledge_base_index.json")

    api_key_env = os.getenv("GEMINI_API_KEY")

    pipeline = IngestionPipeline()
    pipeline.ingest_full_dataset(
        base_dataset_dir=dataset_base,
        metadata_file_path=meta_file,
        output_index_path=index_file,
        max_pages_per_doc=3,
        api_key=api_key_env
    )
