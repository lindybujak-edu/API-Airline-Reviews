import streamlit as st

st.markdown('''## Airline Review Sentiment Classification  
Have you ever traveled and had a great experience with the airline? What about a negative experience?  
Test out this airline review classifier! Below, write your airline related review, and this BERT style transformer will classify your review. For example:''')
st.markdown(''' *I had the worst flight ever. All the flight attendants were rude, and our food was served cold. Do better next time.*
            ''', text_alignment="center")
st.markdown("This should come out as :red[negative]. Give your ideas a try!")

