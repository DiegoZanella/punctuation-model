# Load the data and print a few results to verify it
from datasets import load_from_disk
from transformers import GPT2Tokenizer

# Load the saved dataset
dataset = load_from_disk("../data/eswiki-processed")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")

# Print the decoded inputs and targets
for i in range(5):
    input_text = tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)
    target_text = tokenizer.decode(dataset[i]['labels'], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Target: {target_text}")