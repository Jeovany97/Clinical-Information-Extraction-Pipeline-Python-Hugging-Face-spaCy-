from transformers import pipeline

class SDOHClassifier:
    def __init__(self, model_name="roberta-large-mnli"):
        self.classifier = pipeline("text-classification", model=model_name)
        self.hypotheses = {
            "tobacco_use": "The patient uses tobacco.",
            "alcohol_use": "The patient consumes alcohol.",
            "employment": "The patient is unemployed.",
            "housing": "The patient has housing insecurity.",
            "social_support": "The patient lacks social support."
        }

    def classify(self, premise):
        results = {}
        for category, hypothesis in self.hypotheses.items():
            # Format for RoBERTa NLI
            nli_input = f"{premise} </s></s> {hypothesis}"
            prediction = self.classifier(nli_input)[0]
            results[category] = {
                "label": prediction["label"],
                "score": round(prediction["score"], 3)
            }
        return results