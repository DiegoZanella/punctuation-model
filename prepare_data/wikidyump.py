from datasets import load_dataset
import random
import re
from transformers import GPT2Tokenizer
import json
import os


def normalize_text(text):
    text = text.lower()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^\w\s.,!?¿¡]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(example):
    text = normalize_text(example['text'])
    return {'normalized_text': text}

def create_classification_example(example):
    text = example['normalized_text']
    label_map = {' ': 0, '.': 1, ',': 2, '!': 3, '?': 4, '¿': 5, '¡': 6}
    chars = list(label_map.keys())

    # Choose a label with equal probability from the 7 possible characters
    chosen_char = random.choice(chars)

    # Randomly pick a position to mask (either punctuation or whitespace)
    positions = [i for i, char in enumerate(text) if char == chosen_char]  # Eligible positions
    # If no positions found, try picking another label or skip this example
    # For simplicity, let's try up to a few times:
    retries = 3
    while not positions and retries > 0:
        chosen_char = random.choice(chars)
        positions = [i for i, char in enumerate(text) if char == chosen_char]
        retries -= 1

    if not positions:
        example['input_text'] = text
        example['masked_position'] = 0
        example['label'] = 0  # Dummy label
        example['should_drop'] = True

        return example


    masked_position = random.choice(positions)
    char_at_position = text[masked_position]

    # Map characters to class labels (e.g., space=0, .=1, ,=2, etc.)
    label_map = {' ': 0, '.': 1, ',': 2, '!': 3, '?': 4, '¿': 5, '¡': 6}
    label = label_map[char_at_position]

    # Replace the masked position with a special token (e.g., <mask>)
    masked_text = text[:masked_position] + "<mask>" + text[masked_position + 1:]
    example['input_text'] = masked_text
    example['masked_position'] = masked_position
    example['label'] = label
    example['should_drop'] = False

    return example

def tokenize_data(example, tokenizer):
    inputs = tokenizer(example['input_text'], max_length=512, truncation=True, padding="max_length")
    masked_position = example['masked_position']

    if masked_position >= len(inputs['input_ids']):
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'label': example['label'],
            'masked_position': example['masked_position'],
            'should_drop': True
        }

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'label': example['label'],
        'masked_position': example['masked_position'],
        'should_drop': False
    }

if __name__ == '__main__':
    input_file = '../data/eswiki-train.txt'
    output_file = '../data/eswiki-processed-v3_long'
    print("Current directory: ", os.getcwd())
    # Load the dataset
    # dataset = load_dataset("text", data_files=input_file)['train']
    # dataset = dataset.select(range(10000))
    dataset = load_dataset("daqc/wikipedia-txt-spanish", split="train")
    print(f"Number of training examples: {len(dataset)}")
    dataset = dataset.shuffle(seed=42).select(range(100000))

    # Normalize text
    normalized_dataset = dataset.map(preprocess)

    # Create classification examples
    classification_dataset = normalized_dataset.map(
        create_classification_example,
    ).filter(lambda x: x['should_drop'] == False)  # Remove None values
    classification_dataset = classification_dataset.remove_columns(['should_drop'])

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")

    # Tokenize data
    tokenized_dataset = classification_dataset.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=False
    )
    tokenized_dataset = tokenized_dataset.filter(lambda x: x['should_drop'] == False)
    tokenized_dataset = tokenized_dataset.remove_columns(['input_text', 'normalized_text', 'should_drop'])

    # Save processed dataset
    tokenized_dataset.save_to_disk(output_file)
    print(f"Processed dataset saved to {output_file}")

    label_map = {' ': 0, '.': 1, ',': 2, '!': 3, '?': 4, '¿': 5, '¡': 6}
    with open(f"{output_file}/label_map.json", "w") as f:
        json.dump(label_map, f)

    print(f"Number of examples: {len(tokenized_dataset)}")