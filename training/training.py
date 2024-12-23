from transformers import GPT2Model, Trainer, TrainingArguments, GPT2Tokenizer
from datasets import load_from_disk
import json
from torch import nn
from transformers import Trainer
from torch.nn.functional import cross_entropy
import numpy as np


# Paths and Parameters
DATASET_PATH = "../data/eswiki-processed-v3_short"  # Path to tokenized dataset
TOKENIZER_PATH = "datificate/gpt2-small-spanish"
OUTPUT_DIR = "../models/punctuation_model_v3_short"  # Directory to save the model
BATCH_SIZE = 2   # Batch size for training
EPOCHS = 1  # Number of training epochs
LEARNING_RATE = 5e-5  # Learning rate
FREEZE_LAYERS = 6
NUM_CLASSES = 7




# Load the dataset
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(10))  # Take only the first 200 examples
print("Finished loading dataset...")

# Manually create a train-test split
test_size = 0.1  # Use 10% of the data for validation
split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
print("Finished splitting dataset...")
# Access train and validation datasets
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Load tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)

# Load label map for reference
with open(f"{DATASET_PATH}/label_map.json", "r") as f:
    label_map = json.load(f)
    label_map_inverse = {v: k for k, v in label_map.items()}  # For decoding labels

# Load GPT-2 model
print("Initializing model...")
base_model = GPT2Model.from_pretrained("datificate/gpt2-small-spanish")

class PunctuationClassifier(nn.Module):
    def __init__(self, model, num_classes):
        super(PunctuationClassifier, self).__init__()
        self.gpt2 = model
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)  # Binary classification (space or punctuation)

    def forward(self, input_ids, attention_mask, masked_position, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the hidden states of the masked position
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        masked_states = hidden_states[range(hidden_states.size(0)), masked_position]  # Shape: [batch_size, hidden_size]
        logits = self.classifier(masked_states)  # Shape: [batch_size, 2]

        # Ensure logits always have consistent shape
        if logits.dim() != 2:
            raise ValueError(f"Logits should have 2 dimensions [batch_size, num_classes], got {logits.shape}.")

        # If labels are provided, compute the loss
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits


        return logits

# Wrap the GPT-2 model
model = PunctuationClassifier(base_model, NUM_CLASSES)

# Freeze the first few layers of GPT-2
for layer_index in range(FREEZE_LAYERS):
    for param in model.gpt2.h[layer_index].parameters():
        param.requires_grad = False


# Define training arguments
print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_total_limit=2,  # Keep only the 2 most recent model checkpoints
    logging_dir=f"{OUTPUT_DIR}/logs",  # Directory for logs
    logging_steps=100,  # Log training metrics every 100 steps
)

# Custom Trainer to handle masked_position
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Extract labels
        masked_position = inputs.pop("masked_position")  # Extract masked_position
        logits = model(**inputs, masked_position=masked_position)  # Pass remaining inputs to the model
        loss = cross_entropy(logits, labels)  # Compute cross-entropy loss
        return (loss, logits) if return_outputs else loss


# Define metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Convert logits to predictions
    # Flatten both arrays to ensure matching shapes
    predictions = predictions.flatten()
    labels = labels.flatten()
    # Compute accuracy
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# Initialize Trainer
print("Initializing Trainer...")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
  #  compute_metrics=compute_metrics,
)

print(f"Begining training...")
# Train the model
trainer.train()
print("Finished training...")

# Save the final model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# Testing the model
def test_model(input_text, masked_position):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], masked_position=[masked_position])
    prediction = logits.argmax(axis=-1).item()
    return label_map_inverse[prediction]

# Test input
test_input = "hola<mask>como estas"
test_position = 4  # Position of the mask
print(f"Prediction for '{test_input}' at position {test_position}: {test_model(test_input, test_position)}")