from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os

from models.ipa_predictor import IPAPredictorPipeline
from models.load_vocab import load_ipa_vocab

IPA_VOCAB = load_ipa_vocab(os.getenv("IPA_VOCAB_PATH", "/workspaces/Torran/backend/pronunciation-service/app/models/ipa_vocab.txt"))
MODEL_PATH = os.getenv("IPA_MODEL_PATH", "/models/checkpoints/ipa-predictor-v1.0/model.pt")

router = APIRouter()
pipeline = IPAPredictorPipeline(model_path=MODEL_PATH, ipa_vocab=IPA_VOCAB)

class IPARequest(BaseModel):
    text: str

class IPAResponse(BaseModel):
    ipa: str

@router.post("/predict-ipa", response_model=IPAResponse)
def predict_ipa(request: IPARequest):
    try:
        ipa = pipeline.predict(request.text)
        return IPAResponse(ipa=ipa)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
