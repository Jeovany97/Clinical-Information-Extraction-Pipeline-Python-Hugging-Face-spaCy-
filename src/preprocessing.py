import spacy

# Load spaCy model globally within the module
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model isn't downloaded
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def sentence_split(text):
    """Clean and split text into individual sentences."""
    return [sent.text.strip() for sent in nlp(text).sents]