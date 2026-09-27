from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from preprocess import (
    validate_and_prepare,
    reputation_aggregate,
)


ROOT = Path(__file__).resolve().parents[1]

MODELS = ROOT / "models"

MODELS.mkdir(
    exist_ok=True
)


FEATURES = [
    "review_count",
    "avg_rating",
    "good_pct",
    "neutral_pct",
    "bad_pct",
    "rating_trend",
    "sentiment_trend",
    "negative_change",
    "volume_change_pct",
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(
            ROOT
            / "data"
            / "raw"
            / "consumer_reviews.csv"
        ),
    )

    args = parser.parse_args()

    path = Path(
        args.input
    )

    if not path.exists():

        raise SystemExit(
            f"Real Kaggle dataset not found: {path}"
        )

    print()
    print("=" * 70)
    print("REPUTATION MODEL TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD REAL DATA
    # ---------------------------------------------------------

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(
        f"Raw rows: {len(df)}"
    )

    # ---------------------------------------------------------
    # PREPARE
    #
    # For training, rating-derived sentiment labels are used.
    # ---------------------------------------------------------

    df, validation_info = validate_and_prepare(
        df,
        use_ml_sentiment=False,
    )

    print(
        f"Prepared rows: {len(df)}"
    )

    # ---------------------------------------------------------
    # HISTORICAL FEATURES
    # ---------------------------------------------------------

    try:

        historical = reputation_aggregate(
            df
        )

    except ValueError as exc:

        raise SystemExit(
            str(exc)
        )

    if historical.empty:

        raise SystemExit(
            "No historical reputation data could be created."
        )

    print(
        f"Historical rows: {len(historical)}"
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        historical[
            "will_decline"
        ]
        .value_counts()
        .to_dict()
    )

    # ---------------------------------------------------------
    # FEATURES EXIST?
    # ---------------------------------------------------------

    missing_features = [
        c
        for c in FEATURES
        if c not in historical.columns
    ]

    if missing_features:

        raise SystemExit(
            "Missing reputation features: "
            + ", ".join(
                missing_features
            )
        )

    # ---------------------------------------------------------
    # REMOVE NaN / INF
    # ---------------------------------------------------------

    historical = historical.replace(
        [float("inf"), -float("inf")],
        pd.NA,
    )

    historical = historical.dropna(
        subset=FEATURES
    )

    if len(historical) < 20:

        raise SystemExit(
            "Not enough real historical observations "
            "for reputation model training."
        )

    # ---------------------------------------------------------
    # CHRONOLOGICAL SORT
    # ---------------------------------------------------------

    historical = (
        historical
        .sort_values("period")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # TARGET CHECK
    # ---------------------------------------------------------

    target_counts = (
        historical[
            "will_decline"
        ]
        .value_counts()
    )

    if len(target_counts) < 2:

        print()
        print(
            "WARNING:"
        )

        print(
            "The real dataset does not contain both "
            "increase/stable and decline examples."
        )

        print(
            "Reputation prediction model will not be "
            "created because doing so would require fake data."
        )

        # Remove old model so the application does not
        # accidentally use an old/stale model.
        old_model = (
            MODELS
            / "reputation_model.pkl"
        )

        old_metadata = (
            MODELS
            / "reputation_metadata.json"
        )

        old_model.unlink(
            missing_ok=True
        )

        old_metadata.unlink(
            missing_ok=True
        )

        return

    # ---------------------------------------------------------
    # CHRONOLOGICAL TRAIN / TEST SPLIT
    # ---------------------------------------------------------

    n = len(
        historical
    )

    test_size = max(
        5,
        int(n * 0.25),
    )

    test_size = min(
        test_size,
        n - 10,
    )

    split = n - test_size

    train = historical.iloc[
        :split
    ].copy()

    test = historical.iloc[
        split:
    ].copy()

    print()
    print(
        "Chronological split:"
    )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Testing rows : {len(test)}"
    )

    print()
    print(
        "Training classes:"
    )

    print(
        train[
            "will_decline"
        ]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "Testing classes:"
    )

    print(
        test[
            "will_decline"
        ]
        .value_counts()
        .to_dict()
    )

    # ---------------------------------------------------------
    # TRAINING MUST HAVE BOTH CLASSES
    # ---------------------------------------------------------

    if train[
        "will_decline"
    ].nunique() < 2:

        raise SystemExit(
            "Chronological training portion contains only "
            "one class. More real historical variation "
            "is required."
        )

    # ---------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print()
    print(
        "Training Random Forest..."
    )

    model.fit(
        train[FEATURES],
        train[
            "will_decline"
        ].astype(int),
    )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    predictions = model.predict(
        test[FEATURES]
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            test["will_decline"],
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            test["will_decline"],
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    metrics = {

        "model": "Random Forest",

        "accuracy": float(
            accuracy_score(
                test["will_decline"],
                predictions,
            )
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

        "precision_macro": float(
            macro_precision
        ),

        "recall_macro": float(
            macro_recall
        ),

        "f1_macro": float(
            macro_f1
        ),

        "classification_report": classification_report(
            test["will_decline"],
            predictions,
            output_dict=True,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            test["will_decline"],
            predictions,
            labels=[0, 1],
        ).tolist(),

        "labels": [
            "STABLE_OR_IMPROVING",
            "MAY_DECLINE",
        ],

        "training_rows": int(
            len(train)
        ),

        "test_rows": int(
            len(test)
        ),

        "features": FEATURES,

        "target_definition": (
            "1 when next real brand-month reputation "
            "score is lower than current reputation score; "
            "0 otherwise."
        ),
    }

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    model_path = (
        MODELS
        / "reputation_model.pkl"
    )

    metadata_path = (
        MODELS
        / "reputation_metadata.json"
    )

    joblib.dump(
        model,
        model_path,
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
    # RESULT
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("REPUTATION MODEL COMPLETE")
    print("=" * 70)

    print(
        f"Accuracy          : {metrics['accuracy']:.4f}"
    )

    print(
        f"Weighted Precision : {metrics['precision_weighted']:.4f}"
    )

    print(
        f"Weighted Recall    : {metrics['recall_weighted']:.4f}"
    )

    print(
        f"Weighted F1        : {metrics['f1_weighted']:.4f}"
    )

    print(
        f"Macro F1           : {metrics['f1_macro']:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        metrics[
            "confusion_matrix"
        ]
    )

    print()
    print(
        "Saved:"
    )

    print(
        model_path
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()