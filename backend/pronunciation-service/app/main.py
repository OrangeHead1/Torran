from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.pronunciation import router as pronunciation_router
from api.analysis import router as analysis_router
from api.feedback import router as feedback_router

app = FastAPI(title="Pronunciation Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pronunciation_router, prefix="/api/pronunciation", tags=["pronunciation"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["feedback"])

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
