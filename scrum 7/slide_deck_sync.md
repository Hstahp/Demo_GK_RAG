# KỊCH BẢN THUYẾT TRÌNH SYNC NHÓM (10-15 PHÚT)
## CHỦ ĐỀ: BỨC TRANH TOÀN CẢNH VỀ RAG CHATBOT (GEMINI API + VECTOR DB)
*Dành cho toàn bộ thành viên dự án: Frontend, Backend, AI & QA Engineers*

---

### 🕒 SLIDE 1: TỔNG QUAN VÀ MỤC TIÊU (Thời lượng: 2 phút)
**Tiêu đề:** Trợ lý Học thuật StuDocu AI - Tại sao sinh viên cần nó?
- **Vấn đề thực tế:** Tài liệu slide và giáo trình đại học có thể dài hàng trăm trang. Sinh viên không thể đọc hết trong đêm trước kỳ thi hoặc muốn tìm nhanh câu trả lời chính xác.
- **Tại sao không dùng ChatGPT thông thường?** 
  - Mô hình bên ngoài không đọc được tài liệu nội bộ trong trường (PTIT, Bách Khoa...).
  - Dễ bị ảo giác (hallucination), bịa ra thông tin sai.
  - Không chỉ ra được: *"Thông tin này nằm ở trang mấy của slide cô giáo dạy?"*
- **Mục tiêu của chúng ta:** Xây dựng khung Chatbot AI bên cạnh PDF Viewer. Khi sinh viên hỏi, AI trả lời có dẫn chứng, kèm thẻ **"Sources Found"** - click vào là PDF tự động cuộn đến đúng trang đó!

> 💬 *Lời thoại cho người thuyết trình:* "Chào cả nhóm, hôm nay mình sẽ giải thích nhanh trong 10 phút về tính năng RAG Chatbot. Tính năng này chính là 'vũ khí sát thương' để đồ án của chúng ta ghi điểm tuyệt đối. Sinh viên không chỉ xem PDF đơn thuần mà còn có một gia sư AI túc trực bên cạnh trả lời mọi thắc mắc trích xuất trực tiếp từ bài giảng."

---

### 🕒 SLIDE 2: NGUYÊN LÝ RAG HOẠT ĐỘNG NHƯ THẾ NÀO? (Thời lượng: 3 phút)
**Tiêu đề:** RAG (Retrieval-Augmented Generation) là gì?
- **Bước 1 - Chunking (Cắt nhỏ):** File PDF 100 trang được cắt thành các đoạn nhỏ (mỗi đoạn 600-800 ký tự).
- **Bước 2 - Embedding (Vector hóa):** Dùng `text-embedding-004` để biến mỗi đoạn văn bản thành một chuỗi 768 con số đại diện cho ý nghĩa ngữ nghĩa.
- **Bước 3 - Vector Search (Tìm đoạn liên quan):** Khi bạn sinh viên đặt câu hỏi, ta vector hóa câu hỏi rồi tìm các đoạn có hướng vector gần nhất (Cosine Similarity).
- **Bước 4 - Augment & Generate (Bổ sung & Trả lời):** Đưa câu hỏi + 3-4 đoạn tìm được vào prompt cho **Gemini 1.5 Flash**. Gemini chỉ việc tổng hợp câu trả lời dựa trên sự thật có sẵn!

> 💬 *Lời thoại cho người thuyết trình:* "Các bạn có thể hình dung RAG giống như một kỳ thi 'Open Book'. Thay vì bắt AI phải học thuộc lòng cả triệu cuốn sách, ta cho AI xem đề thi (câu hỏi của user), lật đúng 3 trang sách liên quan nhất đưa cho AI đọc và bảo nó: 'Dựa vào 3 trang này, hãy giải thích cho sinh viên hiểu'."

---

### 🕒 SLIDE 3: CHỐT CÔNG NGHỆ (TECH STACK DECISION) (Thời lượng: 3 phút)
**Tiêu đề:** Tại sao chọn `PostgreSQL + pgvector` thay vì mở thêm DB mới?
- **Hiện trạng Stack:** Đề tài đã yêu cầu PostgreSQL trong Docker/K8s.
- **Nếu dùng ChromaDB / Pinecone độc lập:** Nhóm sẽ phải quản lý thêm 1 server nữa, việc đồng bộ giữa ID tài liệu trong SQL và Vector ID rất dễ bị lệch (desync).
- **Khi dùng `pgvector`:**
  - Cực kỳ gọn nhẹ: Chỉ cần 1 lệnh `CREATE EXTENSION vector;`.
  - Cùng chung 1 cơ sở dữ liệu: Xóa tài liệu ở bảng `documents`, toàn bộ vector tự động bị xóa theo (`CASCADE`).
  - Phân quyền (Security) siêu dễ: Ta có thể lọc sinh viên nào chỉ được xem 3 trang đầu ngay trong câu lệnh `WHERE` của SQL!

> 💬 *Lời thoại cho người thuyết trình:* "Thay vì cài thêm một Vector DB cồng kềnh, Backend của chúng ta chỉ cần cài extension `pgvector` vào ngay container PostgreSQL sẵn có. Điều này giúp hệ thống siêu nhẹ, dễ mang đi demo và cực kỳ ổn định."

---

### 🕒 SLIDE 4: PHỐI HỢP GIỮA FRONTEND VÀ BACKEND (Thời lượng: 4 phút)
**Tiêu đề:** API Contract & Giao diện hiển thị Thẻ Trích dẫn
- **Định dạng dữ liệu:** Gemini API hỗ trợ **Structured Output**, cam kết trả về JSON chuẩn 100%, không bao giờ bị vỡ format.
- **Frontend hiển thị cái gì?**
  1. Bong bóng tin nhắn câu trả lời (Markdown + Code highlight + Math formula).
  2. Bên dưới tin nhắn là các thẻ **[Sources Found]**:
     - Tên tài liệu + Trang số `X`.
     - Đoạn trích dẫn ngắn (Quote snippet).
     - Khi click vào thẻ: Bộ điều khiển PDF Viewer (PDF.js) lập tức cuộn (scroll) đến đúng trang `X` và bôi vàng đoạn trích!
- **Kiểm soát bản quyền (DRM Integration):**
  - Nếu tài liệu chưa được unlock, backend chặn không cho AI search vào các trang từ số 4 trở đi. Điều này tránh việc người dùng lách luật yêu cầu AI leak nội dung trang VIP.

> 💬 *Lời thoại cho người thuyết trình:* "Điểm đặc biệt nhất của StuDocu là thẻ trích dẫn bên dưới câu trả lời. Frontend và Backend sẽ bắt tay nhau qua một chuẩn JSON duy nhất đã được định nghĩa trong file `api_schema_specs.json`. Nhờ đó Frontend có thể bắt tay làm UI ngay hôm nay mà không cần chờ Backend hoàn thiện logic."

---

### 🕒 SLIDE 5: KẾ HOẠCH HÀNH ĐỘNG SPRINT TIẾP THEO (Thời lượng: 2 phút)
**Tiêu đề:** Việc cần làm ngay của từng vị trí
- **AI Engineer:** Tinh chỉnh prompt, kiểm thử các file PDF khó trong thư mục `dataset/`, tích hợp Gemini API key.
- **Backend Engineer:** Chạy migration SQL tạo bảng `document_chunks`, dựng endpoint `/api/v1/ai/chat`.
- **Frontend Engineer:** Dựng khung chat bên phải màn hình Viewer, làm component thẻ `SourcesFoundCard`.
- **QA Engineer:** Dựa vào schema để viết kịch bản kiểm thử (Test cases) về độ trễ, câu hỏi bẫy (jailbreak), và kiểm tra độ chính xác của trích dẫn.

---
*Hết kịch bản buổi Sync. Sẵn sàng giải đáp Q&A cho các thành viên!*
