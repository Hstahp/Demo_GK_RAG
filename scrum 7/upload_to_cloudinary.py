"""
==============================================================================
ACADEMIANET - UPLOAD DATASET PDF TO CLOUDINARY (SCRUM-7)
Tu dong tai 106 file PDF trong dataset len thu muc academianet_docs tren Cloudinary,
sinh file mapping cloudinary_pdf_urls.json de phuc vu demo giua ky va chia se Git.
==============================================================================
"""

import os
import sys
import json
import tempfile
import pymupdf
import cloudinary
import cloudinary.uploader

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Doc cau hinh Cloudinary tu bien moi truong hoac file cuc bo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(SCRIPT_DIR, "cloudinary_credentials.json")

CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")

if os.path.exists(CREDENTIALS_FILE):
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            cred = json.load(f)
            CLOUD_NAME = CLOUD_NAME or cred.get("cloud_name", "")
            API_KEY = API_KEY or cred.get("api_key", "")
            API_SECRET = API_SECRET or cred.get("api_secret", "")
    except Exception:
        pass

if CLOUD_NAME and API_KEY and API_SECRET:
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )
elif os.getenv("CLOUDINARY_URL"):
    cloudinary.config()

curr = SCRIPT_DIR
while curr and os.path.basename(curr).lower() != "studocu":
    parent = os.path.dirname(curr)
    if parent == curr:
        break
    curr = parent

DATASET_BASE = os.path.join(curr, "preProject", "dataset", "dataset")
METADATA_FILE = os.path.join(DATASET_BASE, "metadata.json")
OUTPUT_MAP_FILE = os.path.join(SCRIPT_DIR, "cloudinary_pdf_urls.json")

MAX_FILE_BYTES = 10 * 1024 * 1024  # Gioi han 10MB cua goi Free Cloudinary


def main():
    print("----------------------------------------------------------------------")
    print("ACADEMIANET - BAT DAU UPLOAD DATASET PDF LEN CLOUDINARY")
    print("----------------------------------------------------------------------")

    if not os.path.exists(METADATA_FILE):
        print(f"[Loi] Khong tim thay metadata.json tai: {METADATA_FILE}")
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    # Doc file mapping da co neu muon chay tiep tuc
    url_map = {}
    if os.path.exists(OUTPUT_MAP_FILE):
        try:
            with open(OUTPUT_MAP_FILE, "r", encoding="utf-8") as f:
                url_map = json.load(f)
            print(f">> Da tim thay {len(url_map)} file da upload tu truoc.")
        except Exception:
            url_map = {}

    total = len(meta_list)
    success_count = len(url_map)

    for idx, item in enumerate(meta_list, 1):
        doc_id = item.get("document_id")
        title = item.get("title", "")
        rel_path = item.get("file_path", "")
        abs_path = os.path.join(DATASET_BASE, rel_path)

        if doc_id in url_map:
            continue

        if not os.path.exists(abs_path):
            print(f"[{idx}/{total}] [Bo qua] Khong tim thay file: {rel_path}")
            continue

        file_size = os.path.getsize(abs_path)
        upload_path = abs_path
        is_temp = False
        temp_path = None

        # Xu ly toi uu neu file vuot qua 10MB
        if file_size >= MAX_FILE_BYTES:
            try:
                doc = pymupdf.open(abs_path)
                sub = pymupdf.open()
                max_pages = min(35, len(doc))
                sub.insert_pdf(doc, from_page=0, to_page=max_pages - 1)
                temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
                os.close(temp_fd)
                sub.save(temp_path, garbage=4, deflate=True)
                upload_path = temp_path
                is_temp = True
                print(f"[{idx}/{total}] Toi uu file {doc_id} tu {file_size/(1024*1024):.1f}MB xuong {os.path.getsize(temp_path)/(1024*1024):.1f}MB")
            except Exception as e:
                print(f"[{idx}/{total}] Loi toi uu {doc_id}: {e}")

        try:
            public_id = f"{doc_id}"
            res = cloudinary.uploader.upload(
                upload_path,
                folder="academianet_docs",
                public_id=public_id,
                resource_type="raw",
                overwrite=True
            )
            secure_url = res.get("secure_url")
            url_map[doc_id] = {
                "document_id": doc_id,
                "title": title,
                "course_name": item.get("course_name"),
                "faculty": item.get("faculty"),
                "total_pages": item.get("total_pages", 8),
                "cloudinary_url": secure_url
            }
            success_count += 1
            print(f"[{idx}/{total}] Upload thanh cong [{doc_id}] -> {secure_url}")

            # Ghi lai ngay de khong mat tien do neu bi gian doan
            with open(OUTPUT_MAP_FILE, "w", encoding="utf-8") as f:
                json.dump(url_map, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[{idx}/{total}] [Loi upload {doc_id}]: {e}")
        finally:
            if is_temp and temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    print("----------------------------------------------------------------------")
    print(f">> HOAN TAT: Da upload thanh cong {success_count}/{total} tai lieu.")
    print(f">> File ket qua luu tai: {OUTPUT_MAP_FILE}")
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
