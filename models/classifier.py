# models/classifier.py

from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# Define candidate categories for classification
CATEGORIES = ["general", "legal", "finance", "product", "feature", "news", "collaboration"]

# Few-shot examples per category to guide the model
FEW_SHOT_EXAMPLES = {
    "general": (
        "Q: How does Arista differentiate itself in the enterprise cloud networking space?\n"
        "A: general\n\n"
        "Q: What are the key advantages of Arista's network architecture compared to Cisco and Juniper?\n"
        "A: general"
    ),
    "legal": (
        "Q: Share Arista's standard Master Services Agreement and its terms regarding uptime SLAs.\n"
        "A: legal\n\n"
        "Q: Does your licensing agreement include data privacy clauses?\n"
        "A: legal"
    ),
    "finance": (
        "Q: Outline the projected total cost of ownership over 3 years.\n"
        "A: finance\n\n"
        "Q: Provide a breakdown of recurring vs one-time expenses of service.\n"
        "A: finance"
    ),
    "product": (
        "Q: Explain the hardware architecture and ASIC choices behind the Arista 7500R Series.\n"
        "A: product\n\n"
        "Q: What are the power and port density specs of Arista’s latest 7050X3 switches?\n"
        "A: product"
    ),
    "feature": (
        "Q: Does EOS support BGP EVPN for VXLAN and how is it configured?\n"
        "A: feature\n\n"
        "Q: What built-in telemetry and real-time analytics does CloudVision provide?\n"
        "A: feature"
    ),
    "news": (
        "Q: When is the EOS 4.32 release scheduled and what major changes are included?\n"
        "A: news\n\n"
        "Q: What is Arista’s end-of-sale timeline for the 7280SR platform?\n"
        "A: news"
    ),
    "collaboration": (
        "Q: Which hyperscalers or CSPs did Arista partner with in the past year?\n"
        "A: collaboration\n\n"
        "Q: Describe Arista’s approach to co-engineering engagements with large enterprises.\n"
        "A: collaboration"
    )
}


class QueryClassifier:
    def __init__(self, model_name="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"):
        print(f"[INFO] Loading zero-shot classification model: {model_name}")
        self.classifier = pipeline("zero-shot-classification", model=model_name)

    def build_prompt(self, query: str) -> str:
        examples_prompt = "\n\n".join(example for example in FEW_SHOT_EXAMPLES.values())
        return (
            f"{examples_prompt}\n\n"
            f"Based on the above examples, return only the category of this query.\n\n"
            f"Q: {query}\nA:"
        )

    def classify(self, query: str) -> tuple[str, float]:
        prompt = self.build_prompt(query)
        result = self.classifier(prompt, candidate_labels=CATEGORIES)
        return result['labels'][0], result['scores'][0]

    def classify_batch(self, queries: list[str]) -> list[tuple[str, float]]:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.classify, queries))


# from models.classifier import QueryClassifier

# classifier = QueryClassifier()
# label, score = classifier.classify("What are the contract terms for SLA compliance?")
# print(f"Predicted Category: {label} (Score: {score:.2f})")

# batch_results = classifier.classify_batch([
#     "What’s the cost breakdown for this proposal?",
#     "Who are your recent cloud partners?"
# ])
# for label, score in batch_results:
#     print(f"{label} ({score:.2f})")
