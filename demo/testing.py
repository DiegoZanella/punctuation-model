from transformers import GPT2Tokenizer, GPT2Model
from torch import nn
import torch
import json

# Define paths
MODEL_PATH = "../models/punctuation_model_v3_short/checkpoint-90"  # Latest checkpoint directory
TOKENIZER_PATH = "datificate/gpt2-small-spanish"
LABEL_MAP_PATH = "../data/eswiki-processed-v3_short/label_map.json"

# Load label map
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    label_map_inverse = {v: k for k, v in label_map.items()}

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)


# Define the Punctuation Classifier
class PunctuationClassifier(nn.Module):
    def __init__(self, model, num_classes):
        super(PunctuationClassifier, self).__init__()
        self.gpt2 = model
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)  # For punctuation classification

    def forward(self, input_ids, attention_mask, masked_position):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        masked_states = hidden_states[range(hidden_states.size(0)), masked_position]  # Shape: [batch_size, hidden_size]
        logits = self.classifier(masked_states)  # Shape: [batch_size, num_classes]
        return logits


# Load base GPT-2 model
base_model = GPT2Model.from_pretrained(MODEL_PATH)  # Load from the safetensors file

# Create the classification model
model = PunctuationClassifier(base_model, num_classes=len(label_map))

# Set model to evaluation mode
model.eval()


# Function to test the model
def test_model(input_text, masked_position):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Perform inference
    logits = model(input_ids=input_ids, attention_mask=attention_mask, masked_position=[masked_position])
    prediction = logits.argmax(dim=-1).item()  # Get the predicted class index
    return label_map_inverse[prediction]


# Example usage
test_sentence = "Hola<mask>como estas"  # Example sentence with a mask
masked_position = 1  # Position of the <mask> token (adjust as needed)

print(f"Prediction for '{test_sentence}' at position {masked_position}: {test_model(test_sentence, masked_position)}")

while True:
    try:
        sentence = input("Introduce a sentence with mask: ")
        masked_position = 1
        print(f"Prediction for '{test_sentence}' at position {masked_position}: {test_model(test_sentence, masked_position)}")
    except:
        continue
