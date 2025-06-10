# app/app.py

import os
import gradio as gr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from app.local_llm_reader import get_llm
from models.rag import RAGPipeline
from vectorstore.index import build_index
from app.helper import (
    handle_query,
    apply_suggestion,
    generate_document,
    handle_feedback,
    handle_vote
)

from visualization.visualize import generate_rag_flowchart

# === CONFIGURATION ===
DATA_DIR = "data/VectorDB-Data-Folder"
INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "metadata.pkl"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive" if os.path.exists("/content/drive") else "/backup_data"

MODEL_CHOICES = ["tinyllama", "mistral"] 
model_cache = {}
pipeline_cache = {}

# === SETUP ===

def initialize_vector_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("[INFO] Building FAISS index...")
        build_index(
            main_data_folder=DATA_DIR,
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            drive_backup_dir=DRIVE_BACKUP_DIR
        )
    else:
        print("[INFO] FAISS index and metadata found.")

def build_prompt_template():
    return PromptTemplate(
        input_variables=["query", "category", "context"],
        template=(
            "You are a domain expert and helpful assistant. Answer the following question concisely and clearly, based only on the provided context.\n\n"
            "Category: {category}\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer:"
        )
    )

def preload_models():
    for name in MODEL_CHOICES:
        try:
            llm = get_llm(name)
            prompt = build_prompt_template()
            qa_chain = LLMChain(llm=llm, prompt=prompt)
            rag = RAGPipeline(qa_chain=qa_chain, llm=llm)
            model_cache[name] = llm
            pipeline_cache[name] = {"rag": rag, "qa_chain": qa_chain}
            print(f"[INFO] Loaded model: {name}")
        except Exception as e:
            print(f"[ERROR] Could not load {name}: {e}")

# === GRADIO UI ===

def launch_ui():
    with gr.Blocks() as demo:
        state = gr.State({})

        gr.Markdown("<h1 style='text-align:center;'>📘 RAG Assistant</h1>")

        with gr.Row():
            model_selection = gr.Dropdown(label="Select LLM Model", choices=MODEL_CHOICES, value="tinyllama")
            summarize_flag = gr.Checkbox(label="Summarize documents before answering", value=True)

        user_query = gr.Textbox(label="💬 Your Question")
        submit_btn = gr.Button("🔍 Submit Query")

        output_display = gr.Markdown()
        suggestion = gr.Textbox(label="✍️ Suggest improvements", visible=False)
        apply_btn = gr.Button(" Improve Answer", visible=False)
        final_display = gr.Markdown(visible=False)

        with gr.Row():
            upvote_btn = gr.Button("👍 Upvote", visible=False)
            downvote_btn = gr.Button("👎 Downvote", visible=False)

        vote_ack = gr.Markdown(visible=False)
        retrieved_chunks_display = gr.HTML(visible=False)

        gr.Markdown("## 💬 Feedback")
        feedback_input = gr.Textbox(label="📢 Your Feedback", placeholder="Tell us how we did")
        submit_feedback_btn = gr.Button("📩 Submit Feedback")
        feedback_ack = gr.Markdown()

        generate_btn = gr.Button("📄 Generate Word Document")
        download_file = gr.File(label="⬇️ Download Document")

        # === FUNCTION WIRING ===

        submit_btn.click(
            fn=lambda q, s, sm, m: handle_query(q, s, sm, m, pipeline_cache),
            inputs=[user_query, state, summarize_flag, model_selection],
            outputs=[
                output_display,
                state,
                upvote_btn,
                downvote_btn,
                vote_ack,
                suggestion,
                apply_btn,
                final_display,
                retrieved_chunks_display
            ]
        )

        apply_btn.click(
            fn=lambda sug, st: apply_suggestion(sug, st, pipeline_cache),
            inputs=[suggestion, state],
            outputs=[final_display, state]
        )

        generate_btn.click(fn=generate_document, outputs=download_file)
        submit_feedback_btn.click(fn=handle_feedback, inputs=[feedback_input, state], outputs=[feedback_ack])

        upvote_btn.click(fn=lambda st: handle_vote("up", st), inputs=[state], outputs=[vote_ack])
        downvote_btn.click(fn=lambda st: handle_vote("down", st), inputs=[state], outputs=[vote_ack])

    demo.queue().launch(debug=True, share=True)

# === MAIN ===

def main():
    print("[INFO] Starting RAG Assistant...")
    initialize_vector_store()
    preload_models()
    
    launch_ui()

    # Optional: change filename or disable Drive copy
    generate_rag_flowchart(output_name="my_rag_flowchart")

if __name__ == "__main__":
    main()
