from datasets import load_dataset
import random
import re
import argparse
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")
    # Remove special characters like Unicode symbols (e.g., km² -> km2)
    text = re.sub(r"[^\w\s.,!?¿¡]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_punctuation(text):
    # Remove all punctuation while keeping spaces and alphanumeric characters
    return re.sub(r"[^\w\s]", "", text)

def preprocess(example):
    text = normalize_text(example['text'])  # Normalize text
    unpunctuated = remove_punctuation(text)  # Remove punctuation
    return {'input_text': unpunctuated, 'target_text': text}

def tokenize_data(example, tokenizer):
    # Tokenize input and target
    inputs = tokenizer(example['input_text'], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(example['target_text'], max_length=512, truncation=True, padding="max_length")
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    }

if __name__ == '__main__':
    input_file = '../data/eswiki-train.txt'
    output_file = '../data/eswiki-processed'


    dataset = load_dataset("text", data_files="../data/eswiki-train.txt")
    sample_size = 1000000
    sampled_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))

    print(dataset)
    print(dataset['train'][0])  # Print the first record
    print(f"Number of training examples: {len(dataset['train'])}")

    for i in range(15):
        print(dataset['train'][i])

    random_numbers = [random.randint(1, len(dataset['train'])) for _ in range(10)]
    for i in random_numbers:
        print(dataset['train'][i])

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")

    # Normalize and preprocess
    processed_dataset = sampled_dataset.map(preprocess)
    # Tokenize
    tokenized_dataset = processed_dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text', 'input_text', 'target_text'])

    # Save processed dataset
    tokenized_dataset.save_to_disk(output_file)
    print(f"Processed dataset saved to {output_file}")