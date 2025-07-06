from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os

from models.similarity_engine import SimilarityPipeline
from models.load_vocab import load_ipa_vocab

IPA_VOCAB = load_ipa_vocab(os.getenv("IPA_VOCAB_PATH", "/workspaces/Torran/backend/pronunciation-service/app/models/ipa_vocab.txt"))
MODEL_PATH = os.getenv("SIMILARITY_MODEL_PATH", "/models/checkpoints/similarity-engine-v1.0/model.pt")

router = APIRouter()
pipeline = SimilarityPipeline(model_path=MODEL_PATH, ipa_vocab=IPA_VOCAB)

class SimilarityRequest(BaseModel):
    user_ipa: str
    target_ipa: str

class SimilarityResponse(BaseModel):
    score: float

@router.post("/compare", response_model=SimilarityResponse)
def compare_pronunciation(request: SimilarityRequest):
    try:
        score = pipeline.compare(request.user_ipa, request.target_ipa)
        return SimilarityResponse(score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
