# models/summarizer.py

from transformers import pipeline


class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        print(f"[INFO] Loading summarization model: {model_name}")
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_if_needed(self, text: str, max_words: int = 512) -> str:
        """
        Conditionally summarize long documents to fit within context window limits.

        Args:
            text (str): Document text to evaluate.
            max_words (int): Word threshold to trigger summarization.

        Returns:
            str: Original or summarized version of the text.
        """
        words = text.split()
        if len(words) > max_words:
            print(f"[INFO] Text exceeds {max_words} words. Summarizing...")
            summary = self.summarizer(
                text,
                max_length=512,
                min_length=64,
                do_sample=False
            )[0]['summary_text']
            return summary
        return text


# from models.summarizer import Summarizer

# summarizer = Summarizer()

# long_text = """Your very long document string..."""  # Assume >512 words
# condensed = summarizer.summarize_if_needed(long_text)

# print(condensed)
