import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer
from scipy.sparse import hstack


def get_tree_features(text, count_vectorizer, tfidf_vectorizer):
    count_features = count_vectorizer.transform([text])
    tfidf_features = tfidf_vectorizer.transform([text])
    return hstack([count_features, tfidf_features])

def get_bert_features(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    # Take the CLS token (first token) output as feature representation
    cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

def predict_with_ensemble(text, models):
    # Unpack models and vectorizers
    count_vectorizer, tfidf_vectorizer = models["rf_vecs"]
    xgb_count_vectorizer, xgb_tfidf_vectorizer = models["xgb_vecs"]
    rf_model = models["rf_model"]
    xgb_model = models["xgb_model"]
    bert_model = models["bert_model"]
    tokenizer = models["bert_tokenizer"]

    # Predict with Random Forest
    rf_features = get_tree_features(text, count_vectorizer, tfidf_vectorizer)
    rf_pred = rf_model.predict_proba(rf_features)[0][1]

    # Predict with XGBoost
    xgb_features = get_tree_features(text, xgb_count_vectorizer, xgb_tfidf_vectorizer)
    xgb_pred = xgb_model.predict_proba(xgb_features)[0][1]

    # Predict with BERT
    bert_features = get_bert_features(text, tokenizer, bert_model)
    # Example classifier: average the CLS embedding for demo (you might have a real classifier here)
    bert_pred = float(np.mean(bert_features))  # placeholder logic — replace if you have a BERT classifier

    # Combine predictions (simple average here, or use weighted average / voting)
    final_score = (rf_pred + xgb_pred + bert_pred) / 3
    label = "phishing" if final_score >= 0.5 else "safe"

    return {
        "label": label,
        "confidence": round(final_score, 4),
        "rf_score": round(rf_pred, 4),
        "xgb_score": round(xgb_pred, 4),
        "bert_score": round(bert_pred, 4)
    }
