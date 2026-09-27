# Trained model artifacts

This folder intentionally contains no placeholder model files. After the real Kaggle dataset is placed in `data/raw/consumer_reviews.csv`, run:

`python ml/train_sentiment_model.py --input data/raw/consumer_reviews.csv`

and

`python ml/train_reputation_model.py --input data/raw/consumer_reviews.csv`

Those commands create actual `.pkl` model artifacts and metadata. The backend refuses to fabricate predictions when these files are absent.
