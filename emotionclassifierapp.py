import streamlit as st
from transformers import pipeline

st.title("Emotion Classifier")

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

user_input = st.text_area("How are you feeling today?")

if st.button("Analyze Emotion"):
    if user_input:
        result = classifier(user_input)[0][0]
        st.markdown(f"*Predicted Emotion:* {result['label']} (score: {result['score']:.2f})")
    else:
        st.warning("Please enter some text.")