# vectorstore/retriever.py

import os
import faiss
import pickle
import numpy as np

from vectorstore.embedding import get_cached_embedding


class FAISSRetriever:
    def __init__(
        self,
        index_paths=None,
        metadata_paths=None,
        top_k=3
    ):
        self.index_paths = index_paths or [
            "faiss_index.idx",
            "/content/drive/MyDrive/faiss_index.idx"
        ]
        self.metadata_paths = metadata_paths or [
            "metadata.pkl",
            "/content/drive/MyDrive/metadata.pkl"
        ]
        self.top_k = top_k

        self.index = self._load_index()
        self.documents, self.metadatas = self._load_metadata()

    def _load_index(self):
        for path in self.index_paths:
            if os.path.exists(path):
                print(f"[INFO] Loading FAISS index from: {path}")
                return faiss.read_index(path)
        raise FileNotFoundError("FAISS index file not found in expected locations.")

    def _load_metadata(self):
        for path in self.metadata_paths:
            if os.path.exists(path):
                print(f"[INFO] Loading metadata from: {path}")
                with open(path, "rb") as f:
                    store = pickle.load(f)
                return store["documents"], store["metadatas"]
        raise FileNotFoundError("Metadata file not found in expected locations.")

    def hybrid_search(self, queries):
        """
        Perform vector search using FAISS for a list of queries.
        Returns list of (query, document index) tuples.
        """
        embeddings = [np.array(get_cached_embedding(q)) for q in queries]
        results = []

        for i, q_emb in enumerate(embeddings):
            D, I = self.index.search(np.array([q_emb]).astype("float32"), self.top_k)
            for idx in I[0]:
                results.append((queries[i], idx))

        return list(set(results))  # deduplicate

    def rerank(self, query, query_idx_list):
        """
        Rerank retrieved document chunks using cosine similarity.
        Returns list of (score, document_text, metadata).
        """
        query_vec = np.array(get_cached_embedding(query))
        scored = []

        for _, idx in query_idx_list:
            doc = self.documents[idx]
            doc_vec = np.array(get_cached_embedding(doc))
            similarity = self.cosine_similarity(query_vec, doc_vec)
            scored.append((similarity, doc, self.metadatas[idx]))

        return sorted(scored, key=lambda x: x[0], reverse=True)

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



# from vectorstore.retriever import FAISSRetriever

# retriever = FAISSRetriever(top_k=5)
# query = "What are the evaluation criteria for RFP submissions?"

# initial_hits = retriever.hybrid_search([query])
# ranked_results = retriever.rerank(query, initial_hits)

# for score, doc, meta in ranked_results[:3]:
#     print(f"[{score:.4f}] {meta['filename']}: {doc[:100]}...")
