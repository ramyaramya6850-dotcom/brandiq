from __future__ import annotations

import json
import uuid
import sys
from pathlib import Path
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from .config import *
from .db import get_db, User, Dataset
from .auth import (
    hash_password,
    verify_password,
    create_token,
    get_user_from_token,
)

# ---------------------------------------------------------
# ML IMPORT
# ---------------------------------------------------------

sys.path.append(str(ROOT / "ml"))

from preprocess import (
    validate_and_prepare,
    reputation_aggregate,
)


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

app = FastAPI(
    title="BrandIQ API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

class AuthIn(BaseModel):
    username: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=8, max_length=128)


class MappingIn(BaseModel):
    mapping: dict[str, str] = {}


ALLOWED_EXT = {".csv"}


# ---------------------------------------------------------
# AUTH
# ---------------------------------------------------------

def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    token = authorization[7:]

    user = get_user_from_token(db, token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session.",
        )

    return user


# ---------------------------------------------------------
# DATASET HELPERS
# ---------------------------------------------------------

def dataset_for(
    dataset_id: int,
    user: User,
    db: Session,
):
    ds = db.get(Dataset, dataset_id)

    if not ds or ds.user_id != user.id:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return ds


def processed_path(ds: Dataset):
    return DATA_PROCESSED / f"dataset_{ds.id}.csv"


def load_dataset(ds: Dataset):

    path = processed_path(ds)

    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail="CSV mapping has not been completed yet.",
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    # -----------------------------------------------------
    # DATE FIX
    # -----------------------------------------------------

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

    # -----------------------------------------------------
    # RATING FIX
    # -----------------------------------------------------

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce",
        )

    # -----------------------------------------------------
    # SENTIMENT ML INFERENCE
    # -----------------------------------------------------

    # Processed datasets created before the ML integration may contain
    # rating-derived sentiment only. When confidence is absent, rerun the
    # real trained sentiment model so analytics/reviews use ML predictions.
    if "review_text" in df.columns:
        if "sentiment_confidence" not in df.columns:
            df = apply_sentiment_model(df)
        else:
            df["sentiment"] = (
                df["sentiment"]
                .astype(str)
                .str.upper()
                .str.strip()
            )
            df["sentiment_confidence"] = pd.to_numeric(
                df["sentiment_confidence"],
                errors="coerce",
            )

    return df


# ---------------------------------------------------------
# MODEL CHECK
# ---------------------------------------------------------

def model_ready(kind):

    files = {
        "sentiment": [
            MODELS / "sentiment_model.pkl",
            MODELS / "sentiment_vectorizer.pkl",
        ],
        "reputation": [
            MODELS / "reputation_model.pkl",
        ],
    }

    return all(
        p.exists() and p.stat().st_size > 0
        for p in files[kind]
    )


def apply_sentiment_model(df):
    """
    Run the saved trained sentiment model on review text in small batches.

    The uploaded dataset is processed with the real trained TF-IDF +
    classifier model. Batching keeps peak memory lower on Render Free while
    preserving the same ML predictions.
    """
    if "review_text" not in df.columns:
        raise HTTPException(
            status_code=409,
            detail="Sentiment prediction requires a review_text column.",
        )

    if not model_ready("sentiment"):
        raise HTTPException(
            status_code=409,
            detail=(
                "Sentiment ML model is not available. "
                "Run train_sentiment_model.py first."
            ),
        )

    try:
        # Work on the existing dataframe instead of making another full copy.
        # Render Free has limited RAM, so avoid duplicate dataframes while the
        # TF-IDF model is running.
        result = df
        texts = result["review_text"].fillna("").astype(str).tolist()

        model = joblib.load(MODELS / "sentiment_model.pkl")
        vectorizer = joblib.load(MODELS / "sentiment_vectorizer.pkl")

        # Smaller batches reduce peak memory on Render Free.
        batch_size = 100
        predictions_all = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            features = vectorizer.transform(batch_texts)

            # The real trained classifier is still used. We skip predict_proba
            # because its extra array can cause a memory spike on Free.
            predictions = model.predict(features)
            predictions_all.extend(
                pd.Series(predictions)
                .astype(str)
                .str.upper()
                .str.strip()
                .tolist()
            )

            del features
            del predictions

        result["sentiment"] = predictions_all
        result["sentiment_confidence"] = np.nan

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment ML inference failed: {e}",
        )


# ---------------------------------------------------------
# SAFE FLOAT
# ---------------------------------------------------------

def safe_float(value):

    try:

        if value is None:
            return None

        if pd.isna(value):
            return None

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except Exception:
        return None


# ---------------------------------------------------------
# FILTER
# ---------------------------------------------------------

def filter_df(df, q: dict):

    x = df.copy()

    if q.get("product") and "product" in x.columns:
        x = x[
            x["product"].astype(str)
            == str(q["product"])
        ]

    if q.get("region") and "region" in x.columns:
        x = x[
            x["region"].astype(str)
            == str(q["region"])
        ]

    if q.get("category") and "category" in x.columns:
        x = x[
            x["category"].astype(str)
            == str(q["category"])
        ]

    if (
        q.get("sentiment")
        and q["sentiment"] != "ALL"
        and "sentiment" in x.columns
    ):
        x = x[
            x["sentiment"].astype(str).str.upper()
            == str(q["sentiment"]).upper()
        ]

    if q.get("rating") and "rating" in x.columns:

        try:

            x = x[
                x["rating"]
                == float(q["rating"])
            ]

        except Exception:
            pass

    if "date" in x.columns:

        x["date"] = pd.to_datetime(
            x["date"],
            errors="coerce",
        )

        if q.get("start"):

            try:
                x = x[
                    x["date"]
                    >= pd.to_datetime(q["start"])
                ]
            except Exception:
                pass

        if q.get("end"):

            try:

                end_date = (
                    pd.to_datetime(q["end"])
                    + pd.Timedelta(days=1)
                )

                x = x[
                    x["date"] < end_date
                ]

            except Exception:
                pass

        if (
            q.get("year")
            and q["year"] != "ALL"
        ):

            try:
                x = x[
                    x["date"].dt.year
                    == int(q["year"])
                ]
            except Exception:
                pass

        if (
            q.get("month")
            and q["month"] != "ALL"
        ):

            try:
                x = x[
                    x["date"].dt.month
                    == int(q["month"])
                ]
            except Exception:
                pass

        if (
            q.get("day")
            and q["day"] != "ALL"
        ):

            try:
                x = x[
                    x["date"].dt.day
                    == int(q["day"])
                ]
            except Exception:
                pass

    return x


# ---------------------------------------------------------
# QUERY
# ---------------------------------------------------------

def q_from_request(
    product=None,
    region=None,
    category=None,
    sentiment=None,
    rating=None,
    start=None,
    end=None,
    year=None,
    month=None,
    day=None,
):

    return {
        "product": product,
        "region": region,
        "category": category,
        "sentiment": sentiment,
        "rating": rating,
        "start": start,
        "end": end,
        "year": year,
        "month": month,
        "day": day,
    }


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "models": {
            "sentiment": model_ready("sentiment"),
            "reputation": model_ready("reputation"),
        },
    }


# ---------------------------------------------------------
# REGISTER
# ---------------------------------------------------------

@app.post("/api/auth/register")
def register(
    payload: AuthIn,
    db: Session = Depends(get_db),
):

    username = payload.username.strip()

    existing = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Username already exists.",
        )

    user = User(
        username=username,
        password_hash=hash_password(
            payload.password
        ),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "access_token": create_token(user.id),
        "username": user.username,
    }


# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------

@app.post("/api/auth/login")
def login(
    payload: AuthIn,
    db: Session = Depends(get_db),
):

    username = payload.username.strip()

    user = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    if not verify_password(
        payload.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return {
        "access_token": create_token(user.id),
        "username": user.username,
    }


# ---------------------------------------------------------
# ME
# ---------------------------------------------------------

@app.get("/api/auth/me")
def me(
    user: User = Depends(current_user),
):

    return {
        "id": user.id,
        "username": user.username,
    }


# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    ext = Path(
        file.filename or ""
    ).suffix.lower()

    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    data = await file.read()

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_MB} MB.",
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV is empty.",
        )

    temp = (
        DATA_RAW
        / f"upload_{uuid.uuid4().hex}.csv"
    )

    temp.write_bytes(data)

    try:

        raw = pd.read_csv(
            temp,
            low_memory=False,
        )

    except Exception as e:

        temp.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {e}",
        )

    if raw.empty:

        temp.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail="The CSV contains no rows.",
        )

    ds = Dataset(
        user_id=user.id,
        original_filename=(
            file.filename
            or "upload.csv"
        ),
        stored_path=str(temp),
        row_count=len(raw),
        columns_json=json.dumps(
            [str(c) for c in raw.columns]
        ),
        mapping_json="{}",
    )

    db.add(ds)
    db.commit()
    db.refresh(ds)

    import preprocess

    aliases = {
        c: list(v)
        for c, v in preprocess.ALIASES.items()
    }

    auto_mapping = (
        preprocess
        .canonicalize_columns(raw)[1]
    )

    return {
        "dataset_id": ds.id,
        "filename": ds.original_filename,
        "file_size": len(data),
        "row_count": len(raw),
        "columns": [
            str(c)
            for c in raw.columns
        ],
        "aliases": aliases,
        "auto_mapping": auto_mapping,
    }


# ---------------------------------------------------------
# MAP DATASET
# ---------------------------------------------------------

@app.post("/api/datasets/{dataset_id}/map")
def map_dataset(
    dataset_id: int,
    payload: MappingIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    ds = dataset_for(
        dataset_id,
        user,
        db,
    )

    raw = pd.read_csv(
        ds.stored_path,
        low_memory=False,
    )

    try:

        clean, resolved = (
            validate_and_prepare(
                raw,
                payload.mapping,
            )
        )

        # Keep the original row count before releasing the raw dataframe.
        raw_row_count = len(raw)

        # The raw dataframe is no longer needed after validation. Release it
        # before loading the ML model so peak memory stays lower on Render.
        del raw

        # validate_and_prepare creates rating-derived labels for training
        # compatibility. For application data, replace those labels with
        # predictions from the real trained sentiment model.
        clean = apply_sentiment_model(clean)

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Dataset validation failed: {e}",
        )

    out = processed_path(ds)

    clean.to_csv(
        out,
        index=False,
    )

    ds.mapping_json = json.dumps(
        resolved
    )

    ds.row_count = len(clean)

    db.commit()

    return {
        "dataset_id": ds.id,
        "row_count": len(clean),
        "resolved_mapping": resolved,
        "available_columns": list(
            clean.columns
        ),
        "duplicates_removed": int(
            raw_row_count - len(clean)
        ),
    }


# ---------------------------------------------------------
# DATASET INFO
# ---------------------------------------------------------

@app.get("/api/datasets/{dataset_id}")
def dataset_info(
    dataset_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    ds = dataset_for(
        dataset_id,
        user,
        db,
    )

    return {
        "id": ds.id,
        "filename": ds.original_filename,
        "row_count": ds.row_count,
        "columns": json.loads(
            ds.columns_json
        ),
        "mapping": json.loads(
            ds.mapping_json
        ),
        "mapped": processed_path(ds).exists(),
    }


# =========================================================
# ANALYTICS
# =========================================================

@app.get("/api/datasets/{dataset_id}/analytics")
def analytics(
    dataset_id: int,
    product=None,
    region=None,
    category=None,
    sentiment=None,
    rating=None,
    start=None,
    end=None,
    year=None,
    month=None,
    day=None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    try:

        ds = dataset_for(
            dataset_id,
            user,
            db,
        )

        df = load_dataset(ds)

        if df.empty:
            return {
                "summary": {
                    "total_reviews": 0,
                    "good_reviews": 0,
                    "neutral_reviews": 0,
                    "bad_reviews": 0,
                    "good_pct": 0,
                    "neutral_pct": 0,
                    "bad_pct": 0,
                    "average_rating": None,
                    "total_products": 0,
                    "total_regions": 0,
                },
                "sentiment_distribution": [],
                "rating_distribution": [],
                "time_series": [],
                "products": [],
                "regions": [],
                "sales": None,
                "fields": {},
                "filter_options": {},
                "years": [],
            }

        # -------------------------------------------------
        # FILTER
        # -------------------------------------------------

        x = filter_df(
            df,
            q_from_request(
                product,
                region,
                category,
                sentiment,
                rating,
                start,
                end,
                year,
                month,
                day,
            ),
        )

        # -------------------------------------------------
        # SENTIMENT SAFETY
        # -------------------------------------------------

        if "sentiment" not in x.columns:
            raise HTTPException(
                status_code=409,
                detail=(
                    "ML sentiment predictions are not available for this "
                    "dataset. Please remap the CSV after training the "
                    "sentiment model."
                ),
            )

        x["sentiment"] = (
            x["sentiment"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        if "sentiment_confidence" in x.columns:
            x["sentiment_confidence"] = pd.to_numeric(
                x["sentiment_confidence"],
                errors="coerce",
            )

        # -------------------------------------------------
        # RATING SAFETY
        # -------------------------------------------------

        if "rating" not in x.columns:
            x["rating"] = np.nan

        x["rating"] = pd.to_numeric(
            x["rating"],
            errors="coerce",
        )

        total = len(x)

        counts = (
            x["sentiment"]
            .value_counts()
            .to_dict()
        )

        avg = safe_float(
            x["rating"].mean()
        )

        def pct(label):

            if total == 0:
                return 0

            return round(
                counts.get(label, 0)
                / total
                * 100,
                2,
            )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        summary = {
            "total_reviews": int(total),
            "good_reviews": int(
                counts.get("GOOD", 0)
            ),
            "neutral_reviews": int(
                counts.get("NEUTRAL", 0)
            ),
            "bad_reviews": int(
                counts.get("BAD", 0)
            ),
            "good_pct": pct("GOOD"),
            "neutral_pct": pct("NEUTRAL"),
            "bad_pct": pct("BAD"),
            "average_rating": avg,
            "total_products": (
                int(
                    x["product"]
                    .nunique()
                )
                if "product" in x.columns
                else None
            ),
            "total_regions": (
                int(
                    x["region"]
                    .nunique()
                )
                if "region" in x.columns
                else None
            ),
        }

        # -------------------------------------------------
        # SENTIMENT DISTRIBUTION
        # -------------------------------------------------

        sentiment_distribution = [
            {
                "label": label,
                "value": int(
                    counts.get(label, 0)
                ),
            }
            for label in [
                "GOOD",
                "NEUTRAL",
                "BAD",
            ]
        ]

        # -------------------------------------------------
        # RATING DISTRIBUTION
        # -------------------------------------------------

        rating_distribution = []

        try:

            rating_group = (
                x.dropna(
                    subset=["rating"]
                )
                .groupby("rating")
                .size()
                .reset_index(
                    name="count"
                )
            )

            for r in rating_group.itertuples():

                rating_distribution.append(
                    {
                        "rating": safe_float(
                            r.rating
                        ),
                        "count": int(
                            r.count
                        ),
                    }
                )

        except Exception:
            rating_distribution = []

        # =================================================
        # TIME SERIES
        # =================================================

        time_series = []

        if (
            "date" in x.columns
            and x["date"].notna().any()
        ):

            try:

                t = x.dropna(
                    subset=["date"]
                ).copy()

                t["period"] = (
                    t["date"]
                    .dt
                    .to_period("M")
                    .dt
                    .to_timestamp()
                )

                grouped = (
                    t.groupby("period")
                    .agg(
                        review_count=(
                            "rating",
                            "size",
                        ),
                        avg_rating=(
                            "rating",
                            "mean",
                        ),
                        good_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s == "GOOD"
                            ).mean()
                            * 100,
                        ),
                        neutral_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s
                                == "NEUTRAL"
                            ).mean()
                            * 100,
                        ),
                        bad_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s == "BAD"
                            ).mean()
                            * 100,
                        ),
                    )
                    .reset_index()
                )

                for r in grouped.itertuples():

                    time_series.append(
                        {
                            "date": r.period.strftime(
                                "%Y-%m-%d"
                            ),
                            "review_count": int(
                                r.review_count
                            ),
                            "avg_rating": safe_float(
                                r.avg_rating
                            ),
                            "good_pct": round(
                                float(
                                    r.good_pct
                                ),
                                2,
                            ),
                            "neutral_pct": round(
                                float(
                                    r.neutral_pct
                                ),
                                2,
                            ),
                            "bad_pct": round(
                                float(
                                    r.bad_pct
                                ),
                                2,
                            ),
                        }
                    )

            except Exception:
                time_series = []

        # =================================================
        # PRODUCT ANALYTICS
        # =================================================

        products = []

        if "product" in x.columns:

            try:

                g = (
                    x.groupby("product")
                    .agg(
                        review_count=(
                            "rating",
                            "size",
                        ),
                        avg_rating=(
                            "rating",
                            "mean",
                        ),
                        good_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s == "GOOD"
                            ).mean()
                            * 100,
                        ),
                        neutral_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s
                                == "NEUTRAL"
                            ).mean()
                            * 100,
                        ),
                        bad_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s == "BAD"
                            ).mean()
                            * 100,
                        ),
                    )
                    .reset_index()
                    .sort_values(
                        "review_count",
                        ascending=False,
                    )
                    .head(20)
                )

                for r in g.itertuples():

                    products.append(
                        {
                            "product": str(
                                r.product
                            ),
                            "review_count": int(
                                r.review_count
                            ),
                            "avg_rating": safe_float(
                                r.avg_rating
                            ),
                            "good_pct": round(
                                float(
                                    r.good_pct
                                ),
                                2,
                            ),
                            "neutral_pct": round(
                                float(
                                    r.neutral_pct
                                ),
                                2,
                            ),
                            "bad_pct": round(
                                float(
                                    r.bad_pct
                                ),
                                2,
                            ),
                        }
                    )

            except Exception:
                products = []

        # =================================================
        # REGION ANALYTICS
        # =================================================

        regions = []

        if "region" in x.columns:

            try:

                g = (
                    x.groupby("region")
                    .agg(
                        review_count=(
                            "rating",
                            "size",
                        ),
                        avg_rating=(
                            "rating",
                            "mean",
                        ),
                        good_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s == "GOOD"
                            ).mean()
                            * 100,
                        ),
                        neutral_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s
                                == "NEUTRAL"
                            ).mean()
                            * 100,
                        ),
                        bad_pct=(
                            "sentiment",
                            lambda s:
                            (
                                s == "BAD"
                            ).mean()
                            * 100,
                        ),
                    )
                    .reset_index()
                    .sort_values(
                        "review_count",
                        ascending=False,
                    )
                    .head(20)
                )

                for r in g.itertuples():

                    regions.append(
                        {
                            "region": str(
                                r.region
                            ),
                            "review_count": int(
                                r.review_count
                            ),
                            "avg_rating": safe_float(
                                r.avg_rating
                            ),
                            "good_pct": round(
                                float(
                                    r.good_pct
                                ),
                                2,
                            ),
                            "neutral_pct": round(
                                float(
                                    r.neutral_pct
                                ),
                                2,
                            ),
                            "bad_pct": round(
                                float(
                                    r.bad_pct
                                ),
                                2,
                            ),
                        }
                    )

            except Exception:
                regions = []

        # =================================================
        # SALES
        # =================================================

        sales = None

        sales_columns = [
            c
            for c in [
                "sales",
                "revenue",
                "quantity",
            ]
            if c in x.columns
        ]

        if sales_columns:

            sales = {
                "total_sales": None,
                "total_revenue": None,
                "quantity_sold": None,
            }

            if "sales" in x.columns:
                sales["total_sales"] = safe_float(
                    pd.to_numeric(
                        x["sales"],
                        errors="coerce",
                    ).sum()
                )

            if "revenue" in x.columns:
                sales["total_revenue"] = safe_float(
                    pd.to_numeric(
                        x["revenue"],
                        errors="coerce",
                    ).sum()
                )

            if "quantity" in x.columns:
                sales["quantity_sold"] = safe_float(
                    pd.to_numeric(
                        x["quantity"],
                        errors="coerce",
                    ).sum()
                )

            if (
                "date" in x.columns
                and "revenue" in x.columns
            ):

                try:

                    t = x.dropna(
                        subset=["date"]
                    ).copy()

                    t["revenue"] = pd.to_numeric(
                        t["revenue"],
                        errors="coerce",
                    )

                    t["period"] = (
                        t["date"]
                        .dt
                        .to_period("M")
                        .dt
                        .to_timestamp()
                    )

                    sg = (
                        t.groupby("period")
                        ["revenue"]
                        .sum()
                        .reset_index()
                    )

                    sales["trend"] = [
                        {
                            "date": r.period.strftime(
                                "%Y-%m-%d"
                            ),
                            "revenue": safe_float(
                                r.revenue
                            ),
                        }
                        for r in sg.itertuples()
                    ]

                except Exception:

                    sales["trend"] = []

        # =================================================
        # FIELD AVAILABILITY
        # =================================================

        field_names = [
            "date",
            "product",
            "brand",
            "category",
            "region",
            "sales",
            "quantity",
            "price",
            "revenue",
        ]

        fields = {
            c: c in df.columns
            for c in field_names
        }

        # =================================================
        # FILTER OPTIONS
        # =================================================

        filter_options = {}

        for c in [
            "product",
            "region",
            "category",
        ]:

            if c in df.columns:

                values = (
                    df[c]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                filter_options[c] = sorted(
                    values
                )[:300]

            else:

                filter_options[c] = []

        # =================================================
        # YEARS
        # =================================================

        years = []

        if "date" in df.columns:

            try:

                dates = pd.to_datetime(
                    df["date"],
                    errors="coerce",
                )

                years = sorted(
                    dates.dropna()
                    .dt.year
                    .unique()
                    .tolist()
                )

            except Exception:

                years = []

        # =================================================
        # FINAL RESPONSE
        # =================================================

        return {
            "summary": summary,
            "sentiment_distribution":
                sentiment_distribution,
            "rating_distribution":
                rating_distribution,
            "time_series":
                time_series,
            "products":
                products,
            "regions":
                regions,
            "sales":
                sales,
            "fields":
                fields,
            "filter_options":
                filter_options,
            "years":
                years,
            "sentiment_model": {
                "source": "trained_ml_model",
                "model": "TF-IDF + Logistic Regression",
                "confidence_available": "sentiment_confidence" in df.columns,
            },
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "ANALYTICS ERROR:",
            repr(e),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analytics processing failed: {str(e)}",
        )


# =========================================================
# REVIEWS
# =========================================================

@app.get("/api/datasets/{dataset_id}/reviews")
def reviews(
    dataset_id: int,
    page: int = 1,
    page_size: int = 25,
    search: str = "",
    product=None,
    region=None,
    category=None,
    sentiment=None,
    rating=None,
    start=None,
    end=None,
    year=None,
    month=None,
    day=None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    ds = dataset_for(
        dataset_id,
        user,
        db,
    )

    df = load_dataset(ds)

    x = filter_df(
        df,
        q_from_request(
            product,
            region,
            category,
            sentiment,
            rating,
            start,
            end,
            year,
            month,
            day,
        ),
    )

    if search and "review_text" in x.columns:

        x = x[
            x["review_text"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    total = len(x)

    page = max(
        1,
        page,
    )

    page_size = min(
        max(1, page_size),
        100,
    )

    start_index = (
        page - 1
    ) * page_size

    x = x.iloc[
        start_index:
        start_index + page_size
    ]

    rows = []

    for _, r in x.iterrows():

        row = {
            "review_text": str(
                r.get(
                    "review_text",
                    "",
                )
            ),
            "rating": safe_float(
                r.get("rating")
            ),
            "sentiment": str(
                r.get(
                    "sentiment",
                    "",
                )
            ),
            "confidence": safe_float(
                r.get("sentiment_confidence")
            ),
        }

        for c in [
            "date",
            "product",
            "product_id",
            "brand",
            "category",
            "region",
            "review_title",
        ]:

            if c not in x.columns:
                continue

            value = r[c]

            if c == "date":

                if pd.notna(value):

                    try:
                        row[c] = pd.to_datetime(
                            value
                        ).strftime(
                            "%Y-%m-%d"
                        )
                    except Exception:
                        row[c] = None

                else:
                    row[c] = None

            else:

                row[c] = (
                    None
                    if pd.isna(value)
                    else str(value)
                )

        rows.append(row)

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "rows": rows,
    }


# =========================================================
# PREDICTION
# =========================================================

@app.get("/api/datasets/{dataset_id}/prediction")
def prediction(
    dataset_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    try:

        ds = dataset_for(
            dataset_id,
            user,
            db,
        )

        df = load_dataset(ds)

        if not model_ready("reputation"):

            raise HTTPException(
                status_code=409,
                detail=(
                    "Reputation ML model is not available. "
                    "Run train_reputation_model.py first."
                ),
            )

        required = [
            "brand",
            "date",
            "sentiment",
            "rating",
        ]

        missing = [
            c
            for c in required
            if c not in df.columns
        ]

        if missing:

            raise HTTPException(
                status_code=409,
                detail=(
                    "Prediction requires: "
                    + ", ".join(required)
                    + ". Missing: "
                    + ", ".join(missing)
                ),
            )

        a = reputation_aggregate(df)

        if a.empty:

            raise HTTPException(
                status_code=409,
                detail=(
                    "There is not enough historical "
                    "data for future reputation prediction."
                ),
            )

        model = joblib.load(
            MODELS / "reputation_model.pkl"
        )

        metadata_path = (
            MODELS
            / "reputation_metadata.json"
        )

        metadata = {}

        if metadata_path.exists():

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

        features = metadata.get(
            "features",
            [
                "review_count",
                "avg_rating",
                "good_pct",
                "neutral_pct",
                "bad_pct",
                "rating_trend",
                "sentiment_trend",
                "negative_change",
                "volume_change_pct",
            ],
        )

        # ---------------------------------------------
        # Missing feature safety
        # ---------------------------------------------

        missing_features = [
            c
            for c in features
            if c not in a.columns
        ]

        if missing_features:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Prediction feature mismatch. "
                    "Missing: "
                    + ", ".join(
                        missing_features
                    )
                ),
            )

        latest = (
            a.sort_values("period")
            .groupby(
                "brand",
                as_index=False,
            )
            .tail(1)
            .copy()
        )

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        X = latest[features].copy()

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.fillna(0)

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = (
                model.predict_proba(X)
            )

            if (
                hasattr(
                    model,
                    "classes_",
                )
                and 1 in model.classes_
            ):

                class_index = list(
                    model.classes_
                ).index(1)

                decline_prob = (
                    probabilities[
                        :,
                        class_index,
                    ]
                )

            else:

                decline_prob = np.zeros(
                    len(latest)
                )

        else:

            predictions = model.predict(X)

            decline_prob = (
                predictions
                .astype(float)
            )

        latest[
            "probability_decline"
        ] = decline_prob

        latest[
            "future_prediction"
        ] = np.where(
            latest[
                "probability_decline"
            ] >= 0.5,
            "MAY DECLINE",
            "STABLE",
        )

        # ---------------------------------------------
        # Current reputation
        # ---------------------------------------------

        sentiment_score = (
            df["sentiment"]
            .astype(str)
            .str.upper()
            .map(
                {
                    "GOOD": 1.0,
                    "NEUTRAL": 0.5,
                    "BAD": 0.0,
                }
            )
        )

        current_score = safe_float(
            sentiment_score.mean()
        )

        if current_score is None:
            current_score = 0.0

        if current_score >= 0.70:

            status = "GOOD / STABLE"

        elif current_score >= 0.45:

            status = "AT RISK"

        else:

            status = "POOR / DECLINING"

        # ---------------------------------------------
        # Brand results
        # ---------------------------------------------

        items = []

        for _, r in latest.iterrows():

            items.append(
                {
                    "brand": str(
                        r["brand"]
                    ),
                    "period": pd.to_datetime(
                        r["period"]
                    ).strftime(
                        "%Y-%m-%d"
                    ),
                    "decline_probability":
                        round(
                            float(
                                r[
                                    "probability_decline"
                                ]
                            )
                            * 100,
                            1,
                        ),
                    "future_prediction":
                        str(
                            r[
                                "future_prediction"
                            ]
                        ),
                    "avg_rating":
                        safe_float(
                            r.get(
                                "avg_rating"
                            )
                        ),
                    "good_pct":
                        round(
                            float(
                                r.get(
                                    "good_pct",
                                    0,
                                )
                            )
                            * 100,
                            1,
                        ),
                    "neutral_pct":
                        round(
                            float(
                                r.get(
                                    "neutral_pct",
                                    0,
                                )
                            )
                            * 100,
                            1,
                        ),
                    "bad_pct":
                        round(
                            float(
                                r.get(
                                    "bad_pct",
                                    0,
                                )
                            )
                            * 100,
                            1,
                        ),
                    "review_count":
                        int(
                            r.get(
                                "review_count",
                                0,
                            )
                        ),
                    "rating_trend":
                        safe_float(
                            r.get(
                                "rating_trend"
                            )
                        ),
                    "sentiment_trend":
                        safe_float(
                            r.get(
                                "sentiment_trend"
                            )
                        ),
                    "negative_change":
                        safe_float(
                            r.get(
                                "negative_change"
                            )
                        ),
                    "volume_change_pct":
                        safe_float(
                            r.get(
                                "volume_change_pct"
                            )
                        ),
                }
            )

        # ---------------------------------------------
        # Overall prediction
        # ---------------------------------------------

        if len(latest):

            weights = latest[
                "review_count"
            ].astype(float)

            probabilities = latest[
                "probability_decline"
            ].astype(float)

            if weights.sum() > 0:

                weighted = float(
                    np.average(
                        probabilities,
                        weights=weights,
                    )
                )

            else:

                weighted = float(
                    probabilities.mean()
                )

        else:

            weighted = 0.0

        overall_prediction = (
            "MAY DECLINE"
            if weighted >= 0.5
            else "STABLE"
        )

        # ---------------------------------------------
        # Explanation
        # ---------------------------------------------

        explanation = []

        recent = (
            a.sort_values("period")
            .groupby("brand")
            .tail(2)
            .sort_values("period")
        )

        for brand, group in recent.groupby(
            "brand"
        ):

            if len(group) < 2:
                continue

            before = group.iloc[0]
            after = group.iloc[-1]

            explanation.append(
                f"{brand}: good sentiment moved "
                f"from {before.good_pct * 100:.1f}% "
                f"to {after.good_pct * 100:.1f}%, "
                f"while average rating moved "
                f"from {before.avg_rating:.2f} "
                f"to {after.avg_rating:.2f}."
            )

        return {
            "current": {
                "status": status,
                "reputation_score": round(
                    current_score * 100,
                    1,
                ),
                "total_reviews": int(
                    len(df)
                ),
                "good_pct": round(
                    (
                        df["sentiment"]
                        == "GOOD"
                    ).mean()
                    * 100,
                    1,
                ),
                "neutral_pct": round(
                    (
                        df["sentiment"]
                        == "NEUTRAL"
                    ).mean()
                    * 100,
                    1,
                ),
                "bad_pct": round(
                    (
                        df["sentiment"]
                        == "BAD"
                    ).mean()
                    * 100,
                    1,
                ),
                "average_rating":
                    safe_float(
                        df["rating"].mean()
                    ),
            },
            "future": {
                "prediction":
                    overall_prediction,
                "decline_probability":
                    round(
                        weighted * 100,
                        1,
                    ),
                "model":
                    "Random Forest on chronological brand-month features",
            },
            "by_brand": items,
            "explanations":
                explanation[:8],
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "PREDICTION ERROR:",
            repr(e),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction processing failed: {str(e)}"
            ),
        )


# =========================================================
# EXPORT CSV
# =========================================================

@app.get("/api/datasets/{dataset_id}/export/csv")
def export_csv(
    dataset_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    ds = dataset_for(
        dataset_id,
        user,
        db,
    )

    df = load_dataset(ds)

    buffer = BytesIO()

    df.to_csv(
        buffer,
        index=False,
    )

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition":
                f"attachment; filename=brandiq_{ds.id}.csv"
        },
    )


# =========================================================
# EXPORT REPORT
# =========================================================

@app.get("/api/datasets/{dataset_id}/export/report")
def export_report(
    dataset_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):

    ds = dataset_for(
        dataset_id,
        user,
        db,
    )

    df = load_dataset(ds)

    path = (
        REPORTS
        / f"brandiq_report_{ds.id}.pdf"
    )

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
    )

    story = [
        Paragraph(
            "Brand Reputation Analysis Report",
            styles["Title"],
        ),
        Spacer(1, 12),
        Paragraph(
            f"Dataset: {ds.original_filename}",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    total_reviews = len(df)

    average_rating = (
        df["rating"].mean()
        if "rating" in df.columns
        else 0
    )

    good_count = (
        (
            df["sentiment"]
            == "GOOD"
        ).sum()
        if "sentiment" in df.columns
        else 0
    )

    neutral_count = (
        (
            df["sentiment"]
            == "NEUTRAL"
        ).sum()
        if "sentiment" in df.columns
        else 0
    )

    bad_count = (
        (
            df["sentiment"]
            == "BAD"
        ).sum()
        if "sentiment" in df.columns
        else 0
    )

    vals = [
        ["Metric", "Value"],
        [
            "Total Reviews",
            str(total_reviews),
        ],
        [
            "Average Rating",
            f"{average_rating:.2f}",
        ],
        [
            "Good",
            str(good_count),
        ],
        [
            "Neutral",
            str(neutral_count),
        ],
        [
            "Bad",
            str(bad_count),
        ],
    ]

    table = Table(
        vals,
        colWidths=[220, 180],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#16213e"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1, 18)
    )

    story.append(
        Paragraph(
            "This report contains calculated values "
            "from the uploaded dataset. Optional "
            "sales, product and region fields are "
            "included only when supplied.",
            styles["BodyText"],
        )
    )

    doc.build(story)

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
    )

# ---------------------------------------------------------
# FRONTEND STATIC FILES
# Serve the production React build from the FastAPI server.
# This allows BrandIQ frontend + backend to use one public URL.
# ---------------------------------------------------------

FRONTEND_DIST = ROOT / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

if FRONTEND_ASSETS.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_ASSETS),
        name="assets",
    )


@app.get("/{full_path:path}")
def frontend_routes(full_path: str):
    # Do not turn unknown API endpoints into the React page.
    if full_path.startswith("api/"):
        raise HTTPException(
            status_code=404,
            detail="API endpoint not found.",
        )

    index_file = FRONTEND_DIST / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend build not found. Run npm run build in frontend.",
        )

    return FileResponse(index_file)

