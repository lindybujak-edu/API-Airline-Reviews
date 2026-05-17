import streamlit as st
import requests

# Introduction
st.markdown('''## Airline Review Sentiment Classification :airplane:  
Have you ever traveled and had a great experience with the airline? What about a negative experience?  
Test out this airline review classifier! Below, write your airline related review, and this BERT style transformer will classify your review. For example:''')
st.markdown(''' > "I had the worst flight ever. All the flight attendants were rude, and our food was served cold. Do better next time."
            ''', text_alignment="center")
st.markdown("This should come out as :red[negative]. Give your ideas a try!")

# Text box for user input
review_txt = st.text_area(
    "Your Review:",
    "",
    height=2
)

# Map Hugging Face labels to sentiments
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Button to run review through model & print output
if st.button("Submit"):
    if review_txt.strip() == "":
        st.warning("Please type something first.")
    else:
        api_url = "http://127.0.0.1:8000/predict/sentiment"
        # JSON Structure for Pydantic BaseModel
        payload = {"review_text": review_txt}

        try:
            with st.spinner("Communicating to FastAPI backend..."):
                response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                label = data['prediction']
                score = data['confidence_score']
                # st.write(f'''
                #          Sentiment class: {label}\n
                #          Confidence level: {score}''')
                col1, col2 = st.columns(2)
                col1.metric("Sentiment", label)
                col2.metric("Confidence", f"{score}")
            else: 
                st.write(f"API Returned error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.write("Could not connect to API. Ensure API is running (uvicorn main.app --reload)")

        