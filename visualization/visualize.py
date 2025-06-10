# visualization/visualize.py

import os
from graphviz import Digraph
import shutil

def generate_rag_flowchart(output_name="rag_pipeline", save_to_drive=False):
    """
    Generates and saves a RAG pipeline flowchart as a PNG file.
    
    Args:
        output_name (str): Name of the output file (without extension).
        save_to_drive (bool): If True, copies the image to Google Drive (must be mounted separately).
    Returns:
        str: Path to the rendered PNG file.
    """

    # Create a Digraph object
    dot = Digraph(format='png', engine='dot')
    dot.attr(rankdir='TB', size='10,16')

    # Node colors
    bg_color = '#AED6F1'
    process_color = '#A9DFBF'
    user_color = '#F9E79F'
    rerank_color = '#FADBD8'

    # Background System Column
    with dot.subgraph(name='cluster_bg') as c:
        c.attr(label='Background System', style='dashed')
        c.node('Docs', 'Text Documents', style='filled', fillcolor=bg_color, shape='cylinder')
        c.node('MetaIndex', 'Metadata & Indexing\n(TextSplitter + Timestamps)', style='filled', fillcolor=bg_color, shape='cylinder')
        c.node('VectorDB', 'Vector Database (FAISS)\nEmbeddings via MiniLM-L6-v2', style='filled', fillcolor=bg_color, shape='cylinder')
        c.edge('Docs', 'MetaIndex')
        c.edge('MetaIndex', 'VectorDB')

    # RAG Pipeline Column
    with dot.subgraph(name='cluster_rag') as c:
        c.attr(label='RAG Pipeline', style='dashed')
        c.node('UserQuery', 'User Query', style='filled', fillcolor=user_color, shape='box')
        c.node('ExpandQuery', 'Expand to K Queries\n(t5-paraphraser)', style='filled', fillcolor=process_color, shape='box')
        c.node('FewShot', 'Few-Shot Classification\n(DeBERTa-v3 Model)', style='filled', fillcolor=process_color, shape='box')
        c.node('MajorityLabel', 'Majority Label\n(from few-shot results)', style='filled', fillcolor=process_color, shape='box')
        c.node('Retrieval', 'FAISS Retrieval\n(FAISS + MiniLM Embeddings)', style='filled', fillcolor=process_color, shape='box')
        c.node('Dedup', 'Remove Duplicates\n(Set-based)', style='filled', fillcolor=rerank_color, shape='box')
        c.node('Rerank', 'Rerank by Similarity\n(Cosine Score)', style='filled', fillcolor=rerank_color, shape='box')
        c.node('TopK', 'Top-K for Context', style='filled', fillcolor=process_color, shape='box')
        c.node('SummarizeOpt', 'Summarize Documents? (Optional)\n(DistilBART)', style='filled', fillcolor=process_color, shape='box')
        c.node('GenerateAnswer', 'Generate Answer (LLM)\n(Mistral or TinyLlama via LangChain)', style='filled', fillcolor=process_color, shape='box')
        c.node('Confidence', 'Score Confidence\n(Composite Metric)', style='filled', fillcolor=process_color, shape='box')
        c.node('FinalAnswer', 'Final Answer', style='filled', fillcolor=user_color, shape='box')

    # User Interaction Column
    with dot.subgraph(name='cluster_ui') as c:
        c.attr(label='User Interaction', style='dashed')
        c.node('Feedback', 'System Feedback Form', style='filled', fillcolor=user_color, shape='box')
        c.node('Upvote', 'Upvote', style='filled', fillcolor=user_color, shape='box')
        c.node('Downvote', 'Downvote', style='filled', fillcolor=user_color, shape='box')
        c.node('Suggestion', 'Suggest Refinement\n(Align with Domain)', style='filled', fillcolor=user_color, shape='box')
        c.node('RefinedAnswer', 'New Answer via Suggestion\n(Query + Answer + Context + Suggestion)', style='filled', fillcolor=process_color, shape='box')

    # Define edges
    dot.edge('UserQuery', 'ExpandQuery', label='initial query')
    dot.edge('ExpandQuery', 'FewShot', label='K rephrased queries')
    dot.edge('FewShot', 'MajorityLabel', label='predicted labels')
    dot.edge('FewShot', 'Retrieval', label='label-filtered queries')
    dot.edge('VectorDB', 'Retrieval', label='vector search')
    dot.edge('Retrieval', 'Dedup', label='retrieved docs (top-N)')
    dot.edge('Dedup', 'Rerank', label='unique docs')
    dot.edge('Rerank', 'TopK', label='ranked docs')
    dot.edge('TopK', 'SummarizeOpt', label='context docs (if large)', style='dashed')
    dot.edge('SummarizeOpt', 'GenerateAnswer', label='summarized context', style='dashed')
    dot.edge('TopK', 'GenerateAnswer', label='raw context (if not summarized)')
    dot.edge('GenerateAnswer', 'Confidence', label='generated text')
    dot.edge('Confidence', 'FinalAnswer', label='scored answer')
    dot.edge('FinalAnswer', 'Suggestion', label='refine?')
    dot.edge('FinalAnswer', 'Downvote', label='negative vote')
    dot.edge('FinalAnswer', 'Upvote', label='positive vote')
    dot.edge('FinalAnswer', 'Feedback', label='free-form input')
    dot.edge('Suggestion', 'RefinedAnswer', label='suggestion + context')
    dot.edge('RefinedAnswer', 'FinalAnswer', label='new version')

    # Render and save diagram
    output_path = dot.render(filename=output_name, cleanup=True)
    
    if save_to_drive:
        drive_path = f"/content/drive/MyDrive/{os.path.basename(output_path)}"
        shutil.copy(output_path, drive_path)
        print(f"Saved to Google Drive: {drive_path}")

    return output_path

# Example usage (can be removed in production)
# if __name__ == "__main__":
#     path = generate_rag_flowchart(output_name="my_rag_flowchart")
#     print(f"Diagram saved at: {path}")

# from visualization.visualize import generate_rag_flowchart

# # Optional: change filename or disable Drive copy
# generate_rag_flowchart(output_name="my_rag_flowchart")