from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Tokenizer
from datasets import load_from_disk, DatasetDict

# Paths and Parameters
DATASET_PATH = "../data/eswiki-processed"  # Path to tokenized dataset
TOKENIZER_PATH = "datificate/gpt2-small-spanish"
OUTPUT_DIR = "../models/punctuation_model_v1"  # Directory to save the model
BATCH_SIZE = 2   # Batch size for training
EPOCHS = 1  # Number of training epochs
LEARNING_RATE = 5e-5  # Learning rate
FREEZE_LAYERS = 6





# Load the dataset
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.select(range(200))  # Take only the first 200 examples
print("Finished loading dataset...")

# Manually create a train-test split
test_size = 0.1  # Use 10% of the data for validation
split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
print("Finished splitting dataset...")
# Access train and validation datasets
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Load the GPT2-small-spanish model
model = GPT2LMHeadModel.from_pretrained("datificate/gpt2-small-spanish")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)

print("Model created...")
# Freeze the first few layers
for layer_index in range(FREEZE_LAYERS):
    for param in model.transformer.h[layer_index].parameters():
        param.requires_grad = False  # Disable gradient computation for this layer

# Define training arguments
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

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print(f"Begining training...")
# Train the model
trainer.train()
print("Finished training...")

# Save the final model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")


# Testing: Load a sample text and run it through the fine-tuned model
def test_model(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    # Decode the output tokens
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


# Example test input
test_input = "hola como estas espero que todo este bien"
print("\nTesting the model on unseen text...")
print(f"Input: {test_input}")
print(f"Prediction: {test_model(test_input)}")