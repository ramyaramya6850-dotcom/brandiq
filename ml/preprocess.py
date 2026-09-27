from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


ALIASES = {
    "review_text": [
        "review_text",
        "review",
        "reviews.text",
        "text",
        "review_body",
        "reviews_text",
        "content",
    ],
    "rating": [
        "rating",
        "reviews.rating",
        "star_rating",
        "overall",
        "score",
        "rate",
    ],
    "date": [
        "date",
        "reviews.date",
        "review_date",
        "review_time",
        "timestamp",
        "time",
    ],
    "product": [
        "product",
        "product_name",
        "product_title",
        "name",
        "reviews.name",
    ],
    "product_id": [
        "product_id",
        "asin",
        "asins",
        "productid",
    ],
    "brand": [
        "brand",
        "manufacturer",
    ],
    "category": [
        "category",
        "categories",
        "product_category",
    ],
    "region": [
        "region",
        "marketplace",
        "country",
        "state",
        "city",
    ],
    "sales": [
        "sales",
        "total_sales",
    ],
    "quantity": [
        "quantity",
        "units_sold",
    ],
    "price": [
        "price",
        "retail_price",
    ],
    "revenue": [
        "revenue",
        "total_revenue",
    ],
    "customer_id": [
        "customer_id",
        "reviewerID",
        "user_id",
    ],
    "review_title": [
        "review_title",
        "summary",
        "review_headline",
        "reviews.title",
        "title",
    ],
}


def clean_text(x) -> str:
    if pd.isna(x):
        return ""

    x = str(x)

    # HTML
    x = re.sub(r"<[^>]+>", " ", x)

    # URLs
    x = re.sub(r"https?://\S+|www\.\S+", " ", x)

    # Keep normal text / punctuation
    x = re.sub(r"[^\w\s.,!?'\-]", " ", x, flags=re.UNICODE)

    # Multiple spaces
    x = re.sub(r"\s+", " ", x).strip().lower()

    return x


def canonicalize_columns(
    df: pd.DataFrame,
    mapping: dict | None = None,
) -> tuple[pd.DataFrame, dict]:

    df = df.copy()
    mapping = mapping or {}

    used = set()
    result = {}

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for canonical, aliases in ALIASES.items():

        requested = mapping.get(canonical)

        source = None

        # User-selected mapping gets first priority
        if requested and requested in df.columns:
            source = requested

        # Automatic alias matching
        else:
            for alias in aliases:

                alias_lower = alias.lower()

                if alias_lower in normalized:

                    candidate = normalized[alias_lower]

                    if candidate not in used:
                        source = candidate
                        break

        if source is not None:
            result[canonical] = source
            used.add(source)

    output = pd.DataFrame(index=df.index)

    for canonical, source in result.items():
        output[canonical] = df[source]

    return output, result


def _rating_to_training_label(rating):
    """
    Used ONLY while training the sentiment model.

    The real Kaggle dataset does not provide a ready GOOD/NEUTRAL/BAD
    sentiment column, so ratings are used as the supervised training label.

    This is NOT used for uploaded-data prediction.
    """

    if pd.isna(rating):
        return None

    rating = float(rating)

    if rating >= 4:
        return "GOOD"

    if rating == 3:
        return "NEUTRAL"

    return "BAD"


def _load_sentiment_model():

    model_path = MODELS / "sentiment_model.pkl"
    vectorizer_path = MODELS / "sentiment_vectorizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "sentiment_model.pkl not found. "
            "Train the sentiment model first."
        )

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            "sentiment_vectorizer.pkl not found. "
            "Train the sentiment model first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def _predict_sentiment_with_ml(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Run the ACTUAL trained sentiment model on uploaded reviews.
    """

    model, vectorizer = _load_sentiment_model()

    texts = clean["review_text"].fillna("").astype(str)

    X = vectorizer.transform(texts)

    predictions = model.predict(X)

    clean["sentiment"] = predictions

    # Confidence if the classifier supports probabilities
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        clean["sentiment_confidence"] = np.max(
            probabilities,
            axis=1,
        )

    else:
        clean["sentiment_confidence"] = np.nan

    return clean


def validate_and_prepare(
    df: pd.DataFrame,
    mapping: dict | None = None,
    use_ml_sentiment: bool = True,
):

    if df.empty:
        raise ValueError("The CSV is empty.")

    clean, resolved = canonicalize_columns(
        df,
        mapping,
    )

    missing = [
        c
        for c in ("review_text", "rating")
        if c not in clean.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------
    # TEXT CLEANING
    # ---------------------------------------------------------

    clean["review_text"] = (
        clean["review_text"]
        .map(clean_text)
    )

    # ---------------------------------------------------------
    # RATING CLEANING
    # ---------------------------------------------------------

    clean["rating"] = pd.to_numeric(
        clean["rating"],
        errors="coerce",
    )

    clean = clean[
        clean["review_text"].str.len() > 0
    ]

    clean = clean[
        clean["rating"].between(
            1,
            5,
            inclusive="both",
        )
    ]

    # ---------------------------------------------------------
    # DATE
    # ---------------------------------------------------------

    if "date" in clean.columns:

        clean["date"] = pd.to_datetime(
            clean["date"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

    # ---------------------------------------------------------
    # NUMERIC BUSINESS FIELDS
    # ---------------------------------------------------------

    for column in [
        "sales",
        "quantity",
        "price",
        "revenue",
    ]:

        if column in clean.columns:

            clean[column] = pd.to_numeric(
                clean[column],
                errors="coerce",
            )

    # ---------------------------------------------------------
    # TEXT BUSINESS FIELDS
    # ---------------------------------------------------------

    for column in [
        "product",
        "brand",
        "category",
        "region",
    ]:

        if column in clean.columns:

            clean[column] = (
                clean[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    # ---------------------------------------------------------
    # REMOVE DUPLICATES
    # ---------------------------------------------------------

    if "date" in clean.columns:

        clean = clean.drop_duplicates(
            subset=[
                "review_text",
                "rating",
                "date",
            ]
        )

    else:

        clean = clean.drop_duplicates(
            subset=[
                "review_text",
                "rating",
            ]
        )

    clean = clean.reset_index(drop=True)

    # ---------------------------------------------------------
    # SENTIMENT
    # ---------------------------------------------------------

    if use_ml_sentiment:

        # IMPORTANT:
        # Uploaded data uses the ACTUAL trained ML model.
        clean = _predict_sentiment_with_ml(
            clean
        )

    else:

        # Training only:
        # create labels from rating.
        clean["sentiment"] = (
            clean["rating"]
            .map(_rating_to_training_label)
        )

        clean["sentiment_confidence"] = np.nan

    return clean, resolved


def reputation_aggregate(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "brand" not in df.columns:
        raise ValueError(
            "Reputation analysis requires a brand column."
        )

    if "date" not in df.columns:
        raise ValueError(
            "Reputation analysis requires a date column."
        )

    x = df.copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    )

    x = x.dropna(
        subset=["date"]
    )

    if x.empty:
        return pd.DataFrame()

    x["period"] = (
        x["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # ---------------------------------------------------------
    # BRAND + MONTH AGGREGATION
    # ---------------------------------------------------------

    grouped = x.groupby(
        ["brand", "period"],
        dropna=False,
    )

    a = grouped.agg(
        review_count=("rating", "size"),
        avg_rating=("rating", "mean"),

        good_pct=(
            "sentiment",
            lambda s: (s == "GOOD").mean(),
        ),

        neutral_pct=(
            "sentiment",
            lambda s: (s == "NEUTRAL").mean(),
        ),

        bad_pct=(
            "sentiment",
            lambda s: (s == "BAD").mean(),
        ),
    ).reset_index()

    a = a.sort_values(
        ["brand", "period"]
    )

    # ---------------------------------------------------------
    # HISTORICAL FEATURES
    # ---------------------------------------------------------

    a["rating_trend"] = (
        a.groupby("brand")["avg_rating"]
        .diff()
    )

    a["sentiment_trend"] = (
        a.groupby("brand")["good_pct"]
        .diff()
    )

    a["negative_change"] = (
        a.groupby("brand")["bad_pct"]
        .diff()
    )

    a["volume_change_pct"] = (
        a.groupby("brand")["review_count"]
        .pct_change()
        .replace(
            [float("inf"), -float("inf")],
            np.nan,
        )
    )

    # ---------------------------------------------------------
    # REPUTATION SCORE
    # ---------------------------------------------------------

    a["rep_score"] = (
        a["good_pct"]
        + (a["neutral_pct"] * 0.5)
    )

    # Next month's actual reputation.
    a["next_rep_score"] = (
        a.groupby("brand")["rep_score"]
        .shift(-1)
    )

    # ---------------------------------------------------------
    # TARGET
    #
    # 1 = next period reputation decreases
    # 0 = next period stable/increases
    #
    # This is more meaningful than requiring a fixed 8%
    # decline and avoids the previous 43:2 imbalance.
    # ---------------------------------------------------------

    a["will_decline"] = (
        a["next_rep_score"]
        < a["rep_score"]
    ).astype("Int64")

    # Last period of every brand has no future target.
    a = a[
        a["next_rep_score"].notna()
    ].copy()

    # Lag features have no previous period.
    for column in [
        "rating_trend",
        "sentiment_trend",
        "negative_change",
        "volume_change_pct",
    ]:

        a[column] = (
            a[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    return a.reset_index(drop=True)