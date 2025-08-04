"""
Streamlit web application for movie review sentiment analysis.

This app loads a pre-trained sentiment classifier and its associated TF-IDF
vectorizer and exposes a simple web interface for classifying new movie
reviews.  Users can enter free-form text, click **Predict**, and the app
will display whether the review is positive or negative along with the
predicted probabilities.  To run the app, execute::

    streamlit run app.py

Ensure that `sentiment_model.joblib` and `sentiment_vectorizer.joblib`
are present in the same directory as this script.  These files are
produced by `train_sentiment.py`.
"""

import joblib
import streamlit as st
from typing import Tuple

@st.cache_resource
def load_artifacts(model_path: str, vectorizer_path: str) -> Tuple[object, object]:
    """
    Load the classifier and vectorizer from disk. Cached so it only runs once.
    """
    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return classifier, vectorizer

def predict_sentiment(
    text: str, classifier: object, vectorizer: object
) -> Tuple[str, float, float]:
    """
    Given a piece of text, return:
      - label: "positive" or "negative"
      - pos_proba: probability assigned to the "positive" class
      - neg_proba: probability assigned to the "negative" class
    """
    X_vec = vectorizer.transform([text])
    proba = classifier.predict_proba(X_vec)[0]
    classes = classifier.classes_
    pos_proba = proba[list(classes).index("positive")]
    neg_proba = proba[list(classes).index("negative")]
    label = "positive" if pos_proba >= neg_proba else "negative"
    return label, float(pos_proba), float(neg_proba)

def main() -> None:
    # --- Page setup ---
    st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")
    st.title("🎬 Movie Review Sentiment Analysis")
    st.write("Enter a movie review below and click **Predict** to classify it.")

    # --- Load model & vectorizer once ---
    classifier, vectorizer = load_artifacts(
        model_path="sentiment_model.joblib",
        vectorizer_path="sentiment_vectorizer.joblib",
    )

    # --- Input area (prefill from query params if available) ---
    initial_review = st.query_params.get("review", [""])[0]
    user_input = st.text_area(
        "Movie Review",
        value=initial_review,
        placeholder="Type or paste a movie review here...",
        height=150,
    )

    # --- Prediction button ---
    if st.button("Predict", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text before prediction.")
        else:
            with st.spinner("Analyzing sentiment..."):
                label, pos_proba, neg_proba = predict_sentiment(
                    user_input, classifier, vectorizer
                )

            st.markdown("### Result")
            if label == "positive":
                st.success(f"Positive 👍 ({pos_proba:.2%}); Negative: {neg_proba:.2%}")
            else:
                st.error(f"Negative 👎 ({neg_proba:.2%}); Positive: {pos_proba:.2%}")

            # --- Update URL query params so you can share the input & result ---
            st.query_params.clear()
            st.query_params["review"] = user_input
            st.query_params["sentiment"] = label

            # Rerun to apply changes
            st.rerun()

    # --- Example reviews expander ---
    with st.expander("Try an example review"):
        examples = [
            "This movie was absolutely wonderful, with superb acting and a touching story.",
            "I fell asleep halfway through — the plot was dull and the characters were flat.",
            "An unexpected delight! It made me laugh and cry in equal measure.",
            "Terrible. Waste of time and money. Avoid at all costs.",
        ]
        for ex in examples:
            if st.button(f"Use example: {ex}"):
                st.query_params.clear()
                st.query_params["review"] = ex
                st.rerun()

if __name__ == "__main__":
    main()
