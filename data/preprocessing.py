# data/preprocessing.py

import os
import shutil
import pandas as pd
import requests
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging

# === CONFIGURATION ===

SOURCE_DIR = "data/RFP data/Website data/web_content/batch_1"
OUTPUT_DIR = "data/VectorDB-Data-Folder"
PRODUCT_CSV_PATH = "data/sku_series_datasheet.csv"
PRODUCT_PDF_DIR = os.path.join(OUTPUT_DIR, "products_data")
SUPPORT_PDF_PATH = "data/Support-RFP-FAQ.pdf"
SUPPORT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "support_data")


# === STEP 1: Load Text Files ===

def get_text_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.txt')]


# === STEP 2: Categorize Filenames by Hierarchy ===

def categorize_files_by_depth(file_names, depth=3):
    categorized = {}
    for filename in file_names:
        parts = filename.split('_')
        current = categorized
        for i in range(min(depth, len(parts))):
            part = parts[i].replace('.txt', '')
            if i < depth - 1:
                current = current.setdefault(part, {})
            else:
                current.setdefault(part, []).append(filename)
    return categorized


# === STEP 3: Pretty Print Keys ===

def print_dict_keys(d, level=0, max_depth=3):
    if level > max_depth:
        return
    for key, value in d.items():
        count = len(value) if isinstance(value, list) else len(value.keys())
        print("  " * level + f"L{level + 1}: {key} ({count})")
        if isinstance(value, dict):
            print_dict_keys(value, level + 1, max_depth)


# === STEP 4: Extract Key Paths and Copy Files ===

def save_files_for_keys(categorized, keys, output_dir, source_path):
    def extract_files(d, key_parts, current=""):
        if not key_parts or not isinstance(d, dict):
            return {}
        head, *tail = key_parts
        if head in d:
            new_path = os.path.join(current, head)
            if not tail:
                return {new_path: d[head]}
            return extract_files(d[head], tail, new_path)
        return {}

    def copy_recursive(d, rel_path):
        if isinstance(d, list):
            for fname in d:
                src = os.path.join(source_path, fname)
                dest = os.path.join(output_dir, rel_path, fname)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest):
                    shutil.copy(src, dest)
                    print(f"[COPIED] {fname}")
        elif isinstance(d, dict):
            for k, v in d.items():
                copy_recursive(v, os.path.join(rel_path, k))

    for key_path in keys:
        parts = key_path.split('/')
        files = extract_files(categorized, parts)
        for rel_path, contents in files.items():
            copy_recursive(contents, rel_path)


# === STEP 5: Download Product PDFs ===

def download_product_pdfs(csv_path, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if "url" not in df.columns or "name" not in df.columns:
        raise ValueError("CSV must contain 'url' and 'name' columns.")

    for _, row in df.iterrows():
        url, name = row["url"], row["name"]
        folder = os.path.join(base_dir, name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name}.pdf")
        if not os.path.exists(path):
            r = requests.get(url)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"[DOWNLOADED] {path}")
            else:
                print(f"[FAILED] {url}")


# === STEP 6: Extract Text from PDF ===

def process_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if (p := page.extract_text()):
                text += p
            for table in page.extract_tables():
                for row in table:
                    text += "\t".join(cell or "" for cell in row) + "\n"

    for img in convert_from_path(pdf_path):
        text += "\n" + pytesseract.image_to_string(img)

    return text


# === STEP 7: Convert All PDFs in Directory to .txt ===

def process_pdf_folder(input_dir, output_dir):
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            src_path = os.path.join(root, file)
            rel = os.path.relpath(root, input_dir)
            target_dir = os.path.join(output_dir, rel)
            os.makedirs(target_dir, exist_ok=True)
            txt_path = os.path.join(target_dir, file.replace(".pdf", ".txt"))
            if not os.path.exists(txt_path):
                text = process_pdf(src_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                shutil.copy2(src_path, os.path.join(target_dir, file))
                print(f"[PROCESSED] {file}")


# === STEP 8: Process a Single Support PDF ===

def process_single_support_pdf(pdf_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "support_rfp_faq.txt")
    pdf_out = os.path.join(out_dir, "support_rfp_faq.pdf")

    if not os.path.exists(txt_path):
        shutil.copy(pdf_path, pdf_out)
        text = process_pdf(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[SUPPORT SAVED] {txt_path}")


# === STEP 9: Print Folder Tree ===

def list_folders_only(base_path):
    for root, dirs, _ in os.walk(base_path):
        level = root.replace(base_path, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}📂 {os.path.basename(root)}/")


# === MAIN RUNNER ===

if __name__ == "__main__":
    # Step 1–2
    file_names = get_text_files(SOURCE_DIR)
    categorized = categorize_files_by_depth(file_names)
    print_dict_keys(categorized)

    # Step 3–4
    keys_to_copy = [
        'arista/support/product',
        'arista/support/customer',
        'arista/support/software',
        'arista/company',
        'arista/products',
        'arista/advisories',
        'arista/solutions',
        'arista/partner',
        'arista/tech',
        'arista/news'
    ]
    save_files_for_keys(categorized, keys_to_copy, OUTPUT_DIR, SOURCE_DIR)

    # Step 5
    download_product_pdfs(PRODUCT_CSV_PATH, os.path.join("RFP_data", "Products-Pdf"))

    # Step 6–7
    process_pdf_folder(os.path.join("RFP_data", "Products-Pdf"), PRODUCT_PDF_DIR)

    # Step 8
    process_single_support_pdf(SUPPORT_PDF_PATH, SUPPORT_OUTPUT_DIR)

    # Step 9
    list_folders_only(OUTPUT_DIR)
