from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class AdminUserListResponse(BaseModel):
    users: list

@router.get("/users", response_model=AdminUserListResponse)
async def list_users():
    # TODO: Integrate with user microservice or database
    # For now, return a dummy list
    return AdminUserListResponse(users=[{"user_id": "1", "username": "demo"}])
