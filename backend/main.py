"""
SkinProAI FastAPI Backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys

# Add project root to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routes import patients, lesions, analysis, chat

app = FastAPI(title="SkinProAI API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes — analysis must be registered BEFORE patients so the literal
# /gradcam route is not shadowed by the parameterised /{patient_id} route.
app.include_router(analysis.router, prefix="/api/patients", tags=["analysis"])
app.include_router(chat.router, prefix="/api/patients", tags=["chat"])
app.include_router(patients.router, prefix="/api/patients", tags=["patients"])
app.include_router(lesions.router, prefix="/api/patients", tags=["lesions"])

# Ensure upload directories exist
UPLOADS_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Serve uploaded images
if UPLOADS_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Serve React build (production)
BUILD_DIR = Path(__file__).parent.parent / "web" / "dist"
if BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(BUILD_DIR), html=True), name="static")


@app.on_event("shutdown")
async def shutdown_event():
    from backend.services.analysis_service import get_analysis_service
    svc = get_analysis_service()
    if svc.agent.mcp_client:
        svc.agent.mcp_client.stop()


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
