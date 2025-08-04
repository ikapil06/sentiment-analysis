"""
Train a sentiment analysis model using the movie review polarity dataset.

This script loads a CSV dataset containing movie reviews and their sentiment
labels (positive or negative), performs standard text preprocessing using
TF‑IDF features, trains a Logistic Regression classifier and persists both
the vectorizer and the classifier to disk using ``joblib``.  After
training, it prints the accuracy of the model on a held‑out test set.

Usage::

    python train_sentiment.py --data movie_reviews_dataset.csv \
        --model_output sentiment_model.joblib \
        --vectorizer_output sentiment_vectorizer.joblib

The default values for the file paths correspond to the dataset and
output locations created in this repository.  Feel free to adjust
``max_features`` or other hyper‑parameters as needed.

"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for training.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with dataset and output paths.
    """
    parser = argparse.ArgumentParser(description="Train a sentiment analysis classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default="movie_reviews_dataset.csv",
        help="Path to the CSV dataset with 'review' and 'sentiment' columns.",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="sentiment_model.joblib",
        help="File path to save the trained classifier.",
    )
    parser.add_argument(
        "--vectorizer_output",
        type=str,
        default="sentiment_vectorizer.joblib",
        help="File path to save the fitted TF‑IDF vectorizer.",
    )
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing movie reviews and labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``review`` and ``sentiment``.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file '{csv_path}' not found.")
    df = pd.read_csv(csv_path)
    # Drop rows with missing values
    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    return df


def train_model(df: pd.DataFrame, model_output: str, vectorizer_output: str) -> None:
    """Train a sentiment classifier and persist it along with its vectorizer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text and labels.
    model_output : str
        Path to save the trained classifier.
    vectorizer_output : str
        Path to save the fitted TF‑IDF vectorizer.
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    # Create TF‑IDF vectorizer. You can tune ``max_features`` and ``ngram_range``.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),
        max_features=5000,
    )
    # Fit the vectorizer on training data and transform both train and test
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Logistic Regression classifier. Increase max_iter to ensure convergence.
    classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
    classifier.fit(X_train_vec, y_train)

    # Evaluate on the test set
    y_pred = classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Persist the model and vectorizer
    joblib.dump(classifier, model_output)
    joblib.dump(vectorizer, vectorizer_output)
    print(f"Saved classifier to {model_output}")
    print(f"Saved vectorizer to {vectorizer_output}")


def main() -> None:
    args = parse_args()
    df = load_data(args.data)
    train_model(df, args.model_output, args.vectorizer_output)


if __name__ == "__main__":
    main()