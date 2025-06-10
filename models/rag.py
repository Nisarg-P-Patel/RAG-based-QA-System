# models/rag.py

import numpy as np
from collections import Counter

from models.similar_query import QueryParaphraser
from models.classifier import QueryClassifier
from models.summarizer import Summarizer
from vectorstore.retriever import FAISSRetriever
from vectorstore.embedding import get_cached_embedding


class RAGPipeline:
    def __init__(self, qa_chain, llm, retriever=None, summarizer=None):
        self.qa_chain = qa_chain
        self.llm = llm
        self.retriever = retriever or FAISSRetriever()
        self.summarizer = summarizer or Summarizer()
        self.paraphraser = QueryParaphraser()
        self.classifier = QueryClassifier()

    def run(self, user_query: str, summarize_docs=False):
        print("\n[DEBUG] Starting RAG pipeline")
        print(f"[DEBUG] User Query: {user_query}")

        # Step 1: Query Expansion
        expansions = self.paraphraser.generate(user_query)
        all_queries = [user_query] + expansions

        # Step 2: Classification
        label_conf_pairs = self.classifier.classify_batch(all_queries)
        labels = [label for label, _ in label_conf_pairs]
        majority_label = Counter(labels).most_common(1)[0][0]
        confidences = [conf for (label, conf) in label_conf_pairs if label == majority_label]
        avg_classification_conf = sum(confidences) / len(confidences) if confidences else 0

        filtered_queries = [q for q, (label, _) in zip(all_queries, label_conf_pairs) if label == majority_label or q == user_query]

        print("\n[DEBUG] Predicted Categories:")
        for i, (q, (label, conf)) in enumerate(zip(all_queries, label_conf_pairs)):
            print(f"  Q{i+1}: '{q}' -> Category: {label} (Conf: {round(conf, 3)})")
        print(f"[DEBUG] Final Category: {majority_label}")
        print(f"[DEBUG] Avg Classification Confidence: {round(avg_classification_conf, 4)}")

        # Step 3: Retrieval
        retrieved_docs = self.retriever.hybrid_search(filtered_queries)
        print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents.")

        # Step 4: Reranking
        reranked = self.retriever.rerank(user_query, retrieved_docs)
        top_similarities = [sim for sim, _, _ in reranked[:5]]
        avg_similarity = sum(top_similarities) / len(top_similarities)

        # Step 5: Deduplication
        seen_docs = set()
        unique_docs = []
        for score, doc, meta in reranked:
            if doc not in seen_docs:
                seen_docs.add(doc)
                unique_docs.append((score, doc, meta))

        print(f"[DEBUG] Unique Documents After Deduplication: {len(unique_docs)}")

        # Step 6: Context Building
        top_k = 3
        context = "\n\n".join([
            self.summarizer.summarize_if_needed(doc) if summarize_docs else doc
            for _, doc, _ in unique_docs[:top_k]
        ])
        print(f"[DEBUG] Context length: {len(context)}")

        # Step 7: Answer Generation
        print("[DEBUG] Invoking LLM QA Chain...")
        answer = self.qa_chain.invoke({
            "query": user_query,
            "context": context,
            "category": majority_label
        })['text']

        print(f"[DEBUG] Answer: {answer}")

        # Step 8: Confidence Score Calculation
        answer_vec = np.array(get_cached_embedding(answer))
        doc_sims = [
            FAISSRetriever.cosine_similarity(answer_vec, np.array(get_cached_embedding(doc)))
            for _, doc, _ in unique_docs[:top_k]
        ]
        avg_answer_alignment = sum(doc_sims) / len(doc_sims)
        query_vec = np.array(get_cached_embedding(user_query))
        query_answer_sim = FAISSRetriever.cosine_similarity(query_vec, answer_vec)

        final_confidence = round((
            0.25 * avg_classification_conf +
            0.25 * avg_similarity +
            0.30 * avg_answer_alignment +
            0.20 * query_answer_sim
        ) * 100, 2)

        # Step 9: Metadata + Highlight Preparation
        top_sources = [meta for _, _, meta in unique_docs[:top_k]]
        highlighted_chunks = [
            f"""
            <div style="
                border: 1px solid #e0e0e0;
                padding: 16px;
                margin: 12px 0;
                background-color: #f9f9fb;
                border-left: 5px solid #2a6ecf;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 14px;
                line-height: 1.6;
                color: #333;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <strong style="font-size: 15px; color: #2a6ecf;">
                    Chunk {meta['chunk_index']} from <em>{meta['filename']}</em>:
                </strong>
                <pre style="
                  white-space: pre-wrap;
                  font-family: 'Consolas', 'Courier New', monospace;
                  background-color: #fff;
                  color: #333;
                  border: 1px solid #ddd;
                  padding: 10px;
                  margin-top: 8px;
                  border-radius: 4px;
                  overflow-x: auto;
                ">{doc}</pre>
            </div>
            """
            for _, doc, meta in unique_docs[:top_k]
        ]

        return {
            "answer": answer,
            "context": context,
            "category": majority_label,
            "sources": top_sources,
            "retrieved_chunks": unique_docs,
            "confidence_score": final_confidence,
            "highlights": highlighted_chunks
        }
    

