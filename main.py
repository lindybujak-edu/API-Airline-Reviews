from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig

# To load: fastapi dev main.py  | or |  fastapi run main.py
# In second terminal: streamlit run app.py
# To stop: ctrl + C

# Load Hugging Face Model Pipeline into memory
try:
    print("Loading transformer model... This might take a few seconds.")
    # pipeline() high level API connecting to model
    sentiment_analyzer = pipeline(
        "text-classification",
        model="lindybujak/airline-review-modified", 
        tokenizer="lindybujak/airline-review-modified" 
    )
    # Load settings file
    config = AutoConfig.from_pretrained("lindybujak/airline-review-modified").to_dict()
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

    sentiment_label = result['label']
    confidence = round(result['score'] * 100, 2)

    # Return JSON
    return {
        "status": "success",
        "input_text": request.review_text,
        "prediction": sentiment_label,
        "confidence_score": f"{confidence}%",
        "model_type": "DistilBERT Transformer"
    }

# Get method for model details 
# Note: Simple - can be improved upon by adding different model options for user to select
@app.get("/model/info")
async def read_model_info():
    return {
        "status":200,
        "data": config
    }
