import pickle
import torch

def load_models():
    # Load BERT model
    model = torch.load("models/model.safetensors", map_location=torch.device('cpu'), weights_only=False)

    # Load vectorizers and traditional ML models (in binary mode)
    with open("models/count_vectorizer.pkl", "rb") as f:
        count_vectorizer = pickle.load(f)

    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open("models/xgb_count_vectorizer.pkl", "rb") as f:
        xgb_count_vectorizer = pickle.load(f)

    with open("models/xgb_tfidf_vectorizer.pkl", "rb") as f:
        xgb_tfidf_vectorizer = pickle.load(f)

    with open("models/random_forest_model.pkl", "rb") as f:
        random_forest_model = pickle.load(f)

    with open("models/xgboost_model.pkl", "rb") as f:
        xgboost_model = pickle.load(f)

    return {
        "bert_model": model,
        "rf_vecs": (count_vectorizer, tfidf_vectorizer),
        "xgb_vecs": (xgb_count_vectorizer, xgb_tfidf_vectorizer),
        "rf_model": random_forest_model,
        "xgb_model": xgboost_model
    }
