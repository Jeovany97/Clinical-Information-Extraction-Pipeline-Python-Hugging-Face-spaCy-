from transformers import pipeline

class MedicalNER:
    def __init__(self, model_name="d4data/biomedical-ner-all"):
        self.ner_pipeline = pipeline(
            "ner", 
            model=model_name, 
            aggregation_strategy="simple"
        )

    def extract(self, text):
        entities = self.ner_pipeline(text)
        return [
            {
                "text": ent["word"],
                "label": ent["entity_group"],
                "score": round(ent["score"], 3)
            }
            for ent in entities
        ]