# models/similar_query.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class QueryParaphraser:
    def __init__(self, model_name="prithivida/parrot_paraphraser_on_T5", device="auto"):
        print(f"[INFO] Loading paraphraser model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.paraphraser = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            num_beams=5
        )

    def generate(self, query: str, num_questions: int = 5) -> list[str]:
        """
        Generate multiple paraphrased versions of the input query.
        """
        prompt = f"paraphrase the question in different ways: {query}:"

        print("[DEBUG] Generating paraphrases...")
        responses = self.paraphraser(prompt, num_return_sequences=num_questions)
        queries = [resp["generated_text"].strip() for resp in responses]

        print("[DEBUG] Paraphrased Queries:")
        for i, q in enumerate(queries, 1):
            print(f"{i}. {q}")

        return queries


# from models.similar_query import QueryParaphraser

# paraphraser = QueryParaphraser()
# paraphrased_queries = paraphraser.generate("What are the submission guidelines for the proposal?", num_questions=4)
