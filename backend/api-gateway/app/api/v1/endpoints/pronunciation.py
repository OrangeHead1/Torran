from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx
import os

router = APIRouter()

PRONUNCIATION_SERVICE_URL = os.getenv("PRONUNCIATION_SERVICE_URL", "http://pronunciation-service:8001/api/pronunciation")

class IPARequest(BaseModel):
    text: str

class IPAResponse(BaseModel):
    ipa: str

@router.post("/predict-ipa", response_model=IPAResponse)
async def predict_ipa(request: IPARequest):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{PRONUNCIATION_SERVICE_URL}/predict-ipa", json=request.dict())
            resp.raise_for_status()
            return IPAResponse(**resp.json())
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Pronunciation service error: {str(e)}")
