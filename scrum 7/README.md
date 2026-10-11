# SCRUM-7: Nghien cuu RAG Pipeline (Gemini API + Vector DB)

## 1. Gioi thieu Tong quan
Thu muc nay chua toan bo ket qua nghien cuu, tai lieu dac ta kien truc ky thuat va ma nguon thuc nghiem (PoC & Interactive Demo) cho task **SCRUM-7**: *"Nghien cuu RAG Pipeline (Gemini API + Vector DB)"* thuoc de tai **Mang xa hoi chia se tai lieu hoc thuat phi tap trung** (tham khao mo hinh StuDocu).

Muc tieu cot loi cua nghien cuu la xay dung tro ly AI Chatbot thong minh ben canh giao dien doc tai lieu PDF:
- Tra loi cac cau hoi hoc thuat dua tren noi dung thuc te cua tai lieu sinh vien dang xem hoac toan bo kho tai lieu cua mon hoc/truong dai hoc.
- Vua de xuat tai lieu lien quan trong kho DeDocu (Recommender Integration), vua trich dan chinh xac nguon thong tin (Document ID, Ten tai lieu, So trang, Doan trich dan).
- Toi uu chi phi va hieu nang bang cach ket hop co so du lieu Vector va mo hinh ngon ngu lon **Gemini API** (Google) voi kha nang xuat cau truc JSON chuan xac (**Structured Output** qua Pydantic).

---

## 2. Cau truc Thu muc `scrum 7`

```text
scrum 7/
│
├── README.md                      # [File nay] Huong dan tong quan va tom tat nghien cuu
├── RAG_Technical_Proposal.md      # Tai lieu de xuat giai phap ky thuat RAG chi tiet toan dien
├── rag_sequence_diagram.mermaid   # So do tuan tu luong goi RAG (Sequence Diagram)
├── rag_architecture.mermaid       # So do kien truc tong the luong Ingestion & Query
├── api_schema_specs.json          # Dac ta JSON Schema cho Request, Response & Gemini Output
├── slide_deck_sync.md             # Kich ban / Noi dung slide thuyet trinh Sync noi bo nhom
├── ingest_documents.py            # Script bóc tách, chunking và lập chỉ mục Vector đa môn học
├── knowledge_base_index.json      # Cơ sở tri thức Vector chứa 428 chunks (Toán, Kinh tế, CNTT, Bảo mật)
├── rag_pipeline_poc.py            # Mã nguồn PoC Python CLI chạy kiểm thử có Structured Output
└── rag_web_demo.py                # Ứng dụng Web Chat tương tác thời gian thực (DeDocu AI Scholar)
```

---

## 3. Danh muc Tai lieu trong Kho Tri thuc Da Mon hoc

| Document ID | Mon hoc / Khoa | Ten giao trinh dai hoc | So trang / Chunks |
| :--- | :--- | :--- | :--- |
| **REAL-MATH-001** | Toan & Khoa hoc Tu nhien | Giao trinh Giai tich Ham mot bien so (PTIT 2013) | 189 trang (50 chunks) |
| **REAL-ECON-027** | Kinh te & Quan ly | Giao trinh Kinh te Vi mo 1 (PTIT 2013) | 180 trang (88 chunks) |
| **REAL-IT-001** | Cong nghe Thong tin | Bai giang Lap trinh huong doi tuong Java & C++ (PTIT) | 250 trang (79 chunks) |
| **REAL-IT-003** | Cong nghe Thong tin | Giao trinh Co so An toan Thong tin (PTIT 2018) | 143 trang (99 chunks) |
| **REAL-RAG-001** | AI & Cong nghe Thong tin | Nghien cuu mo hinh RAG goc (Lewis et al., Meta AI 2020) | 19 trang (112 chunks) |

---

## 4. Huong dan Chay Thu nghiem

### Cach 1: Chay Giao dien Web Tuong tac Thoi gian thuc (Khuyen nghi cho Thuyet trinh / Demo)

Chay file `rag_web_demo.py` (su dung thu vien san co cua Python, tu dong mo trinh duyet tai `http://localhost:8088`):

```bash
python "preProject/scrum 7/rag_web_demo.py"
```

Tinh nang tren Web:
1. Chon pham vi tim kiem: Tim kiem toan san DeDocu hoac loc theo tung giao trinh cu the.
2. Go bat ky cau hoi nao (vi du: *"cach hoc gioi giai tich"*, *"kinh te vi mo nghien cuu gi"*, *"tinh da hinh trong java"*).
3. AI hien thi the **[Tai lieu de xuat tu thu vien DeDocu]**, dong thoi trich dan chinh xac **[Trang X]** cua giao trinh tuong ung.

### Cach 2: Chay Kiem thu CLI Pipeline PoC (Kiem tra Structured Output Pydantic)

```bash
python "preProject/scrum 7/rag_pipeline_poc.py"
```

Ket qua se in ra JSON Payload hoan chinh kem theo truong `recommended_document` va mang `sources_found` de Frontend de dang tieu thu qua API.

### Cach 3: Chay lai Ingestion Pipeline de cap nhat chi muc

Neu muon quet lai hoac nap them tai lieu tu kho dataset:

```bash
python "preProject/scrum 7/ingest_documents.py"
```

---

## 5. Phan cong Cong viec ke tiep cho cac Thanh vien

- **Backend Engineer:** 
  - Xem tai lieu `RAG_Technical_Proposal.md` va `api_schema_specs.json`.
  - Khoi tao extension `pgvector` tren PostgreSQL Docker: `CREATE EXTENSION IF NOT EXISTS vector;`.
  - Tao bang `document_chunks` theo DDL de luu tru lau dai thay cho file JSON.
- **Frontend Engineer:** 
  - Dua vao mau giao dien trong `rag_web_demo.py` de ghep vao giao dien React DeDocu.
  - Hien thi the `Recommended Document Card` va `Sources Found Badge` canh khung doc PDF.
- **SysOps / DevOps:** 
  - Dam bao image Docker PostgreSQL co kem extension `pgvector` (vi du: `pgvector/pgvector:pg16`).
