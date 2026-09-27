from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1]

def sentiment_predict(texts):
    model=joblib.load(ROOT/'models/sentiment_model.pkl'); vec=joblib.load(ROOT/'models/sentiment_vectorizer.pkl'); X=vec.transform(texts); return model.predict(X), model.predict_proba(X).max(axis=1)
