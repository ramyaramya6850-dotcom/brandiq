from pathlib import Path
import argparse, json, joblib, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from preprocess import validate_and_prepare, reputation_aggregate
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default=str(ROOT/'data/raw/consumer_reviews.csv')); args=ap.parse_args(); p=Path(args.input)
    if not p.exists(): raise SystemExit(f'Real dataset not found: {p}')
    df,_=validate_and_prepare(pd.read_csv(p,low_memory=False))
    out={}
    sm=ROOT/'models/sentiment_model.pkl'; sv=ROOT/'models/sentiment_vectorizer.pkl'
    if sm.exists() and sv.exists():
        from sklearn.model_selection import train_test_split
        Xtr,Xte,ytr,yte=train_test_split(df.review_text,df.sentiment,test_size=.2,random_state=42,stratify=df.sentiment)
        model=joblib.load(sm); vec=joblib.load(sv); pred=model.predict(vec.transform(Xte)); p1,r,f,_=precision_recall_fscore_support(yte,pred,average='weighted',zero_division=0); out['sentiment']={'accuracy':accuracy_score(yte,pred),'precision':p1,'recall':r,'f1':f,'confusion_matrix':confusion_matrix(yte,pred,labels=['GOOD','NEUTRAL','BAD']).tolist()}
    rm=ROOT/'models/reputation_model.pkl'
    if rm.exists() and 'brand' in df and 'date' in df:
        from train_reputation_model import FEATURES
        a=reputation_aggregate(df).sort_values('period'); split=max(int(len(a)*.8),1); te=a.iloc[split:]; model=joblib.load(rm); pred=model.predict(te[FEATURES]); p1,r,f,_=precision_recall_fscore_support(te.will_decline,pred,average='weighted',zero_division=0); out['reputation']={'accuracy':accuracy_score(te.will_decline,pred),'precision':p1,'recall':r,'f1':f,'confusion_matrix':confusion_matrix(te.will_decline,pred,labels=[0,1]).tolist()}
    print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
