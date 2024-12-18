from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Paths
MODEL_PATH = "../models/punctuation_model_v1"  # Path to the fine-tuned model
TOKENIZER_PATH = "datificate/gpt2-small-spanish"  # Path to the original tokenizer
FREEZE_LAYERS = 6  # Number of early layers to freeze (e.g., first 6 layers)

# Load the fine-tuned model and tokenizer
print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)


# Testing function
def test_model(input_text):
    """
    Generate punctuation for the given input text using the fine-tuned model.
    """
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


# Example test inputs
test_inputs = [
    "hola como estas espero que todo este bien",
    "mañana iremos al parque si no llueve",
    "es un placer conocerte",
]

print("\nTesting the model on unseen text...")
for text in test_inputs:
    print(f"Input: {text}")
    print(f"Prediction: {test_model(text)}")
    print("-" * 40)