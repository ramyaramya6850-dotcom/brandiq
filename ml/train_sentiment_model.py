from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from preprocess import validate_and_prepare


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


LABELS = ["GOOD", "NEUTRAL", "BAD"]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(ROOT / "data/raw/consumer_reviews.csv"),
    )

    args = parser.parse_args()

    path = Path(args.input)

    if not path.exists():
        raise SystemExit(
            f"Real Kaggle dataset not found: {path}"
        )

    print("=" * 70)
    print("SENTIMENT MODEL TRAINING")
    print("=" * 70)

    print(f"Dataset: {path}")

    # ---------------------------------------------------------
    # 1. Load REAL dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(f"Raw rows: {len(df)}")

    # ---------------------------------------------------------
    # 2. Prepare data
    # ---------------------------------------------------------

    df, validation_info = validate_and_prepare(df)

    print(f"Prepared rows: {len(df)}")

    if len(df) < 100:
        raise SystemExit(
            "Not enough valid rows for sentiment model training."
        )

    print()
    print("Training sentiment distribution:")
    print(
        df["sentiment"]
        .value_counts()
        .to_dict()
    )

    # ---------------------------------------------------------
    # 3. Check classes
    # ---------------------------------------------------------

    for label in LABELS:

        count = int(
            (df["sentiment"] == label).sum()
        )

        if count < 5:

            raise SystemExit(
                f"Sentiment class '{label}' has only "
                f"{count} rows. At least 5 are required."
            )

    # ---------------------------------------------------------
    # 4. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"],
        df["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["sentiment"],
    )

    print()
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    # ---------------------------------------------------------
    # 5. TF-IDF
    # ---------------------------------------------------------

    print()
    print("Building TF-IDF features...")

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=120000,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=60000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    X_train_features = features.fit_transform(
        X_train
    )

    X_test_features = features.transform(
        X_test
    )

    print(
        "Feature matrix:",
        X_train_features.shape,
    )

    # ---------------------------------------------------------
    # 6. REAL Logistic Regression
    #
    # IMPORTANT:
    # liblinear does NOT support 3-class multiclass
    # classification directly.
    #
    # lbfgs supports multiclass classification.
    # ---------------------------------------------------------

    print()
    print("Training Logistic Regression...")

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1500,
        class_weight="balanced",
        C=2.0,
        random_state=42,
    )

    model.fit(
        X_train_features,
        y_train,
    )

    # ---------------------------------------------------------
    # 7. Prediction
    # ---------------------------------------------------------

    print()
    print("Evaluating model...")

    predictions = model.predict(
        X_test_features
    )

    # ---------------------------------------------------------
    # 8. Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    metrics = {

        "accuracy": float(
            accuracy
        ),

        "precision_weighted": float(
            precision
        ),

        "recall_weighted": float(
            recall
        ),

        "f1_weighted": float(
            f1
        ),

        "classification_report": report,

        "confusion_matrix": matrix.tolist(),

        "labels": LABELS,

        "training_rows": int(
            len(X_train)
        ),

        "testing_rows": int(
            len(X_test)
        ),

        "total_prepared_rows": int(
            len(df)
        ),

        "training_distribution": (
            y_train
            .value_counts()
            .to_dict()
        ),

        "testing_distribution": (
            y_test
            .value_counts()
            .to_dict()
        ),

        "model": "TF-IDF Word + Character Ngrams + Logistic Regression",

        "solver": "lbfgs",

        "real_dataset": True,

        "dataset_path": str(path),

    }

    # ---------------------------------------------------------
    # 9. Save REAL model
    # ---------------------------------------------------------

    model_path = (
        MODELS / "sentiment_model.pkl"
    )

    vectorizer_path = (
        MODELS / "sentiment_vectorizer.pkl"
    )

    metadata_path = (
        MODELS / "sentiment_metadata.json"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        features,
        vectorizer_path,
    )

    metadata_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 10. Display results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SENTIMENT MODEL TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    print()
    print(
        f"Model saved     : {model_path}"
    )

    print(
        f"Vectorizer saved: {vectorizer_path}"
    )

    print(
        f"Metadata saved  : {metadata_path}"
    )

    print()
    print("REAL DATASET:")
    print(path)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()