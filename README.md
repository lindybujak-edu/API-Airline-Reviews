# ✈️ Airline Review Sentiment Classifier

A full-stack AI web application that classifies the sentiment of airline passenger reviews. This project uses a fine-tuned DistilBERT transformer model served via a REST API, paired with an interactive, user-friendly frontend.

## Project Architecture
This application is split into two distinct parts:
* **Backend (`main.py`):** A lightweight FastAPI server that loads the Hugging Face transformer model into memory and exposes a `/predict/sentiment` POST endpoint for user's review submission, as well as a `/predict/sentiment` GET to retrieve model information. 
* **Frontend (`app.py`):** A Streamlit UI that captures user reviews, communicates with the backend via HTTP requests, and displays the sentiment class and confidence score.

## Tech Stack
* **Machine Learning:** Hugging Face `transformers` (DistilBERT)
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Requests

## Prerequisites
Make sure you have Python installed, install the required dependencies, and run main and app (in separate terminals):

```bash
pip install -r requirements.txt
fastapi run main.py
streamlit run app.py
```

## 💌 Message from the Author
**Thanks for your interest in my project!** I had a lot of fun working on this. This started as building a transformer model in my Stats 507 class, and eventually scaled upwards into this application. Through this process, I learned so much about **fine-tuning NLP models with Hugging Face**, **designing RESTful APIs with FastAPI**, and **building interactive web frontends using Streamlit**. 

Feel free to reach out or connect if you have any questions or feedback!