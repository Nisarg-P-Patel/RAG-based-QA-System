# vectorstore/index.py

import os
import pickle
import shutil
import numpy as np
import faiss
import tqdm
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

from vectorstore.embedding import embed_documents

def build_index(main_data_folder: str, index_path: str, metadata_path: str, drive_backup_dir: str):
    documents, metadatas = [], []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    for root, _, files in os.walk(main_data_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                modified_time = datetime.fromtimestamp(os.stat(file_path).st_mtime).isoformat()

                with open(file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()

                chunks = text_splitter.split_text(full_text)

                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "source": os.path.relpath(file_path, main_data_folder),
                        "filename": os.path.basename(file_path),
                        "file_modified_time": modified_time,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "chunk_char_start": full_text.find(chunk),
                        "chunk_char_end": full_text.find(chunk) + len(chunk),
                        "file_type": ".txt",
                        "content_preview": chunk[:50] + ("..." if len(chunk) > 50 else "")
                    })

    print(f"[DEBUG] Encoding {len(documents)} document chunks...")
    embeddings = embed_documents(documents)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump({"documents": documents, "metadatas": metadatas}, f)

    # Ensure backup directory exists
    os.makedirs(drive_backup_dir, exist_ok=True)

    shutil.copy(index_path, os.path.join(drive_backup_dir, os.path.basename(index_path)))
    shutil.copy(metadata_path, os.path.join(drive_backup_dir, os.path.basename(metadata_path)))

    print("[DEBUG] FAISS index and metadata saved successfully.")

# from vectorstore.index import build_index

# build_index(
#     main_data_folder="data/raw_documents",
#     index_path="faiss_index.idx",
#     metadata_path="metadata.pkl",
#     drive_backup_dir="/content/drive/MyDrive"
# )
