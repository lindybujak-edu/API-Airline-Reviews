from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load Hugging Face Model Pipeline into memory
try:
    print("Loading transformer model... This might take a few seconds.")
    # pipeline() high level API connecting to model
    sentiment_analyzer = pipeline(
        "text-classification",
        model="lindybujak/airline-review-modified", 
        tokenizer="lindybujak/airline-review-modified" 
    )
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model. Error: {str(e)}")

# Initialize API
app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="A REST API serving a fine-tuned DistilBERT transformer model.",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema (what incoming data should look like)
class ReviewRequest(BaseModel):
    review_text: str

# Endpoint
@app.post("/predict/sentiment")
async def predict_sentiment(request: ReviewRequest):

    if len(request.review_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    # Run raw text through transformer pipeline
    result = sentiment_analyzer(request.review_text)[0]

    raw_label = result['label']
    confidence = round(result['score'] * 100, 2)

    # Map Hugging Face labels to sentiments
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    sentiment_label = label_map.get(raw_label, raw_label)

    # Return JSON
    return {
        "status": "success",
        "input_text": request.review_text,
        "prediction": sentiment_label,
        "confidence_score": f"{confidence}%",
        "model_type": "DistilBERT Transformer"
    }

# To load: uvicorn main:app --reload
# To stop: ctrl + C