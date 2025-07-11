
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1 import api_router

app = FastAPI(title="Pronunciation Coach API Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
