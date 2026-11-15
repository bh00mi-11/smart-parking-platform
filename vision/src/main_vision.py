"""
Standalone FastAPI application for Computer Vision & Fusion service.
Can be executed via:
    uvicorn vision.src.main_vision:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as vision_router

app = FastAPI(
    title="Smart Parking Vision & Fusion Microservice",
    description="Vision module providing slot occupancy classification, open-zone counting, and probabilistic uncertainty fusion.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(vision_router)


@app.get("/")
def root():
    return {
        "service": "Smart Parking Vision & Fusion API",
        "docs": "/docs",
        "health": "/vision/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vision.src.main_vision:app", host="0.0.0.0", port=8000, reload=False)

