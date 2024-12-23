from datasets import DatasetDict, load_from_disk

# Load your dataset
dataset_path = '../data/eswiki-processed-v3_short'
dataset = load_from_disk(dataset_path)

# Save it as a Hugging Face dataset
dataset.push_to_hub("PanditaInfernal/eswiki-processed-v3-short", private=False)