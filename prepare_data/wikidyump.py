from datasets import load_dataset
import random
import re
from transformers import GPT2Tokenizer

def normalize_text(text):
    text = text.lower()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^\w\s.,!?¿¡]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(example):
    text = normalize_text(example['text'])
    return {'normalized_text': text}

def create_classification_example(text):
    # Randomly pick a position to mask (either punctuation or whitespace)
    positions = [i for i, char in enumerate(text) if char in " .,!?¿¡"]  # Eligible positions
    if not positions:
        return None  # Skip samples with no punctuation or spaces

    masked_position = random.choice(positions)
    char_at_position = text[masked_position]

    # Map characters to class labels (e.g., space=0, .=1, ,=2, etc.)
    label_map = {' ': 0, '.': 1, ',': 2, '!': 3, '?': 4, '¿': 5, '¡': 6}
    label = label_map[char_at_position]

    # Replace the masked position with a special token (e.g., <mask>)
    masked_text = text[:masked_position] + "<mask>" + text[masked_position + 1:]

    return {
        'input_text': masked_text,
        'masked_position': masked_position,
        'label': label
    }

def tokenize_data(example, tokenizer):
    inputs = tokenizer(example['input_text'], max_length=512, truncation=True, padding="max_length")
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'label': example['label'],
        'masked_position': example['masked_position']
    }

if __name__ == '__main__':
    input_file = '../data/eswiki-train.txt'
    output_file = '../data/eswiki-processed-v2'

    # Load the dataset
    dataset = load_dataset("text", data_files=input_file)['train']

    # Normalize text
    normalized_dataset = dataset.map(preprocess)

    # Create classification examples
    classification_dataset = normalized_dataset.map(
        lambda x: create_classification_example(x['normalized_text']),
        remove_columns=['text'],
    ).filter(lambda x: x is not None)  # Remove None values

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")

    # Tokenize data
    tokenized_dataset = classification_dataset.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=False
    )
    tokenized_dataset = tokenized_dataset.remove_columns(['input_text', 'normalized_text'])

    # Save processed dataset
    tokenized_dataset.save_to_disk(output_file)
    print(f"Processed dataset saved to {output_file}")