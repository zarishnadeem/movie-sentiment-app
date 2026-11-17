import re
import joblib
import streamlit as st

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = text.replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'(re|ve|ll|d|m|s)\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    svc = joblib.load("models/linear_svc_model.pkl")
    nb = joblib.load("models/naive_bayes_model.pkl")
    return tfidf, {"Linear SVC": svc, "Naive Bayes": nb}

tfidf, models = load_models()

st.title("🎬 Movie Review Sentiment Analysis")
model_name = st.selectbox("Model", list(models.keys()))
review = st.text_area("Review likho:", height=150)

if st.button("Predict"):
    if not review.strip():
        st.warning("Pehle review likho.")
    else:
        proba = models[model_name].predict_proba(tfidf.transform([clean_text(review)]))[0]
        if proba[1] >= 0.5:
            st.success(f"Positive 😊 ({proba[1]:.0%})")
        else:
            st.error(f"Negative 😞 ({proba[0]:.0%})")