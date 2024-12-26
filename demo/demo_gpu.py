#!/usr/bin/env python3

import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer
import json
import os

################################################################################
# Define the same PunctuationClassifier class used during training
################################################################################
class PunctuationClassifier(nn.Module):
    def __init__(self, model, num_classes):
        super(PunctuationClassifier, self).__init__()
        self.gpt2 = model
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)
        self.config = self.gpt2.config

    def forward(self, input_ids, attention_mask, masked_position, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        # Select the hidden state at the masked position
        masked_states = hidden_states[range(hidden_states.size(0)), masked_position]
        logits = self.classifier(masked_states)

        # If labels are provided, compute the loss (not really used for pure inference)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        return logits


################################################################################
# Inference (Testing) Script
################################################################################
def main():
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    # Specify the model you pushed to Hugging Face
    HF_MODEL_NAME = "PanditaInfernal/punctuation_model_v3-long"

    # Load tokenizer 
    # (It could be the same 'datificate/gpt2-small-spanish' if that's how you tokenized)
    tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")

    # Load the label_map.json (make sure you have it locally or adapt to pull from HF, if hosted)
    with open("punctuation-model/data/eswiki-processed-v3_long/label_map.json", "r") as f:
        label_map = json.load(f)
    # Invert the label map to get punctuation from predicted numeric class
    label_map_inverse = {v: k for k, v in label_map.items()}
    print(f"Label map: {label_map}")
    print(f"Label map inverse: {label_map_inverse}")

    # Load the base GPT2 model from the Hugging Face Hub
    print("Loading base GPT2 model from Hugging Face...")
    base_model = GPT2Model.from_pretrained(HF_MODEL_NAME)

    # Instantiate the custom classifier and move to device
    NUM_CLASSES = 7  # match whatever you used during training
    model = PunctuationClassifier(base_model, NUM_CLASSES).to(device)

    # Define a helper function for testing
    def test_model(input_text, masked_position):
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        # Move to GPU/CPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass: get logits
        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                masked_position=torch.tensor([masked_position], device=device)
            )

        # Argmax to get predicted class
        prediction_id = logits.argmax(dim=-1).item()
        predicted_punct = label_map_inverse[prediction_id]
        return predicted_punct

    ############################################################################
    # Example usage: propose a few test inputs with known <mask> positions
    ############################################################################
    test_examples = [
        ("hola<mask>como estas", 4),
        ("El gato<mask>negro", 3),
        ("Buenos<mask>días, señor", 1),
        ("bienvenida de vuelta, hermosa<mask>Espero que todo haya ido bien.", 9)
    ]

    for text_input, mask_pos in test_examples:
        prediction = test_model(text_input, mask_pos)
        print(f"Input: '{text_input}'")
        print(f"Masked position: {mask_pos}")
        print(f"Predicted punctuation: '{prediction}'")
        # If it's a single punctuation character, you can see its ASCII code
        if len(prediction) == 1:
            print(f"ASCII value of the prediction: {ord(prediction)}")
        print("------")


if __name__ == "__main__":
    main()