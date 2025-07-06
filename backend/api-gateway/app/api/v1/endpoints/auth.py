from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class UserRegisterResponse(BaseModel):
    user_id: str
    username: str
    email: str

@router.post("/register", response_model=UserRegisterResponse)
async def register_user(request: UserRegisterRequest):
    # TODO: Integrate with user microservice or database
    # For now, return a dummy user
    return UserRegisterResponse(user_id="dummy-id", username=request.username, email=request.email)
