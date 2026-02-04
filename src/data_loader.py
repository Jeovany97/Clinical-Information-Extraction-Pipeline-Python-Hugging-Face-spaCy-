from datasets import load_dataset

def load_sdoh_dataset():
    """Load the SDOH-NLI dataset from Hugging Face."""
    return load_dataset("tasksource/SDOH-NLI")

def get_sample_note(dataset, split="train", index=0):
    """Retrieve a specific clinical note for testing."""
    return dataset[split][index]["premise"]