# Clinical-Information-Extraction-Pipeline-Python-Hugging-Face-spaCy-
This project demonstrates an end-to-end clinical NLP pipeline for extracting:
* **Social Determinants of Health (SDoH)** - Tobacco use, Alcohol use, Employment status, Housing insecurity, Social support
The project is designed to mirror real-world healthcare NLP systems and explicitly highlights experience with:
1) Hugging Face Transformers
2) Clinical / biomedical NLP
3) Natural Language Inference (NLI)
4) Named Entity Recognition (NER)
5) Healthcare data considerations

# Dataset
The SDOH-NLI (Hugging Face) dataset is loaded directly using the Hugging Face datasets library.
* Source: tasksource/SDOH-NLI
* Domain: Clinical social history notes
* Format:
    1) **premise:** clinical note text
    2) **hypothesis:** SDoH-related statement
    3) **label:** entailment / neutral / contradiction

# Models Used

1) SDoH Classification (NLI)
   * **Model:** roberta-large-mnli
   * **Task:** Text classification (Natural Language Inference)
   * **Purpose:** Determine whether a clinical note implies specific SDoH categories
  
2) Medical Entity Extraction (NER)
   * **Model:** d4data/biomedical-ner-all
   * **Task:** Named Entity Recognition
   * **Entities:** Medications, diseases, symptoms, procedures
  
3) Text Processing
   * **Library:** spaCy(en_core_web_sm)
   * **Purpose:** Sentence segmentation and clinical-text-safe preprocessing
  
# How the Pipeline Works
### 1) Load the data
A clinical or social history note is used as input.

### 2) SDoH Classification (NLI)
Each SDoH category is represented as a hypothesis. The NLI model predicts whether each hypothesis is entailed, neutral, or contradicted by the note.

### 3) Medical Entity Extraction
A biomedical NER model extracts Medications, Diagonses, and Symptoms

### 4) Output
The pipeline outputs structured JSON suitable for downstream analytics or clinical coding.

