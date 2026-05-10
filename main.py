from transformers import pipeline
# Load Hugging Face Model Pipeline into memory
try:
    print("Loading transformer model... This might take a few seconds.")
    sentiment_analyzer = pipeline(
        "text-classification",
        model="./saved_airline_model",
        tokenizer="./saved_airline_model"
    )
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model. Error: {str(e)}")