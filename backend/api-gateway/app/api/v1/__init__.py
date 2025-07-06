# API v1 router aggregation
from fastapi import APIRouter
from .endpoints import pronunciation, users, auth, progress, admin

api_router = APIRouter()
api_router.include_router(pronunciation.router, prefix="/pronunciation", tags=["pronunciation"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(progress.router, prefix="/progress", tags=["progress"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
