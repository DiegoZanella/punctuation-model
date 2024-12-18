from datasets import load_from_disk
from transformers import GPT2Tokenizer

# Load the saved dataset
dataset = load_from_disk("../data/eswiki-processed-v2")
print(f""
# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")

# Print a few examples to verify the dataset structure
for i in range(5):
    input_text = tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)
    label = dataset[i]['label']  # Multi-class label: 0 = space, 1 = ., 2 = ,, etc.
    masked_position = dataset[i]['masked_position']
    print(f"Input: {input_text}")
    print(f"Masked Position: {masked_position} -> Label (class): {label}")