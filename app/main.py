from fastapi import FastAPI
from pydantic import BaseModel
from app.model_loader import load_models
from app.ensemble_logic import predict_with_ensemble
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins (for testing / browser access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load all models once at startup
models = load_models()

# Root route for Render health checks
@app.get("/")
async def root():
    return {"status": "Backend is live!"}

# Request schema for prediction endpoint
class EmailRequest(BaseModel):
    email: str

# Prediction route
@app.post("/predict")
async def predict(request: EmailRequest):
    result = predict_with_ensemble(request.email, models)
    return result
