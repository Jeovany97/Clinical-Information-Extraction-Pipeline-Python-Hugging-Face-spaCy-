from data_loader import load_sdoh_dataset, get_sample_note
from preprocessing import sentence_split
from sdoh_classifier import SDOHClassifier
from ner_extractor import MedicalNER

def run_clinical_pipeline():
    # Initialize components
    print("Initializing models...")
    sdoh_tool = SDOHClassifier()
    ner_tool = MedicalNER()
    dataset = load_sdoh_dataset()
    
    # Get data
    raw_note = get_sample_note(dataset)
    
    # Process
    print("Processing clinical note...")
    sentences = sentence_split(raw_note)
    sdoh_results = sdoh_tool.classify(raw_note)
    entities = ner_tool.extract(raw_note)
    
    # Final Output
    output = {
        "raw_note": raw_note,
        "sentence_count": len(sentences),
        "sdoh_classification": sdoh_results,
        "extracted_medical_entities": entities
    }
    
    return output

if __name__ == "__main__":
    result = run_clinical_pipeline()
    import json
    print(json.dumps(result, indent=4))