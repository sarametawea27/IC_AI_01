import streamlit as st
import joblib

# ==========================
# Load model and vectorizer
# ==========================
model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Customer Support Chatbot")
st.markdown("Ask a question, and the bot will try to guess which company it's related to!")

# User input
user_input = st.text_input("💬 Type your message here:")

if user_input:
    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)[0]
    st.success(f"🤖 Reply: This message seems related to **{prediction}**")
