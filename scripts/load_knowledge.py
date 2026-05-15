"""
Script sekali jalan: ekstrak semua PDF dari folder jurnal → knowledge_base.json.
Jalankan sebelum start server pertama kali, atau saat ada jurnal baru.

Contoh:
    python scripts/load_knowledge.py \\
        --folder /path/ke/jurnal \\
        --output app/data/knowledge_base.json
"""
import os
import json
import argparse


def extract_pdfs(folder_path: str, output_path: str) -> None:
    try:
        import fitz  # pymupdf
    except ImportError:
        raise SystemExit("Install pymupdf dulu: pip install pymupdf")

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    print(f"{len(pdf_files)} PDF ditemukan di {folder_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    semua_jurnal: dict = {}

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        nama = os.path.splitext(filename)[0]

        try:
            doc = fitz.open(path)
            teks_per_halaman = []

            for i, page in enumerate(doc):
                teks = page.get_text()
                if teks.strip():
                    teks_per_halaman.append({"halaman": i + 1, "isi": teks})

            semua_jurnal[nama] = {
                "judul": nama,
                "total_halaman": len(doc),
                "konten": teks_per_halaman,
            }
            print(f"  OK: {filename} ({len(teks_per_halaman)} halaman)")
            doc.close()

        except Exception as e:
            print(f"  GAGAL: {filename} — {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(semua_jurnal, f, ensure_ascii=False, indent=2)

    print(f"\nSelesai. {len(semua_jurnal)} jurnal disimpan ke {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder berisi file PDF jurnal")
    parser.add_argument("--output", default="app/data/knowledge_base.json")
    args = parser.parse_args()
    extract_pdfs(args.folder, args.output)
