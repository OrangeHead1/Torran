from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ProgressRequest(BaseModel):
    user_id: str
    exercise_id: str
    score: float

class ProgressResponse(BaseModel):
    status: str

@router.post("/update", response_model=ProgressResponse)
async def update_progress(request: ProgressRequest):
    # TODO: Integrate with progress microservice or database
    # For now, return a dummy response
    return ProgressResponse(status="ok")
