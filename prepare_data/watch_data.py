from datasets import load_from_disk
from transformers import GPT2Tokenizer
import json


# Load the saved dataset
dataset = load_from_disk("../data/eswiki-processed-v2")
with open(f"../data/eswiki-processed-v2/label_map.json", "r") as f:
    label_map = json.load(f)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")

# Print a few examples to verify the dataset structure
for i in range(10):
    input_text = tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)
    label = dataset[i]['label']  # Multi-class label: 0 = space, 1 = ., 2 = ,, etc.
    masked_position = dataset[i]['masked_position']
    print(f"Input: {input_text}")
    print(f"Masked Position: {masked_position} -> Label (class): {label}")
    print("----------------------------------")

print(dataset[0].keys())
print(dataset[10])