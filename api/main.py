"""
api/main.py
FastAPI app for PhishGuard ML backend.
Receives email data from Chrome extension, returns phishing analysis.
"""

import pathlib
import yaml
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from api.predictor import get_predictor

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class EmailRequest(BaseModel):
    subject:  Optional[str] = Field(default="", max_length=1000)
    from_:    Optional[str] = Field(default="", alias="from", max_length=500)
    replyTo:  Optional[str] = Field(default="", max_length=500)
    body:     Optional[str] = Field(default="", max_length=50000)
    urls:     Optional[List[str]] = Field(default_factory=list, max_items=50)

    class Config:
        populate_by_name = True

    @validator("urls", pre=True, each_item=True)
    def truncate_url(cls, v):
        return str(v)[:500] if v else ""


class SenderAnalysis(BaseModel):
    displayName: str = ""
    domain: str = ""
    replyTo: Optional[str] = None
    domainMismatch: bool = False


class UrlAnalysis(BaseModel):
    total: int = 0
    suspicious: int = 0
    urls: List[str] = []


class AnalysisResponse(BaseModel):
    score: int = Field(..., ge=0, le=100)
    label: str = Field(..., pattern="^(phishing|suspicious|safe)$")
    reasons: List[str]
    flags: List[str]
    senderAnalysis: SenderAnalysis
    urlAnalysis: UrlAnalysis
    confidence: float = Field(..., ge=0.0, le=1.0)
    bertScore: int
    structuralScore: int
    inferenceMs: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


# ─── App Setup ────────────────────────────────────────────────────────────────

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("🚀 PhishGuard API starting up...")
    try:
        predictor = get_predictor()
        logger.info(f"✅ Model ready on device: {predictor.device}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
    yield
    logger.info("PhishGuard API shutting down")


app = FastAPI(
    title="PhishGuard ML API",
    description="Real-time phishing email detection using DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Chrome extension to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─── Middleware ───────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    t0 = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - t0) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed}ms)")
    return response


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        predictor = get_predictor()
        model_loaded = predictor._loaded
        device = predictor.device
    except Exception:
        model_loaded = False
        device = "unknown"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=device,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_email(request: EmailRequest):
    """
    Analyze an email for phishing indicators.

    Accepts email components (subject, from, body, urls) and returns:
    - Phishing score (0-100)
    - Label (phishing/suspicious/safe)
    - Human-readable reasons
    - Sender and URL analysis
    """
    try:
        predictor = get_predictor()

        email_dict = {
            "subject": request.subject or "",
            "from": request.from_ or "",
            "replyTo": request.replyTo or "",
            "body": request.body or "",
            "urls": request.urls or [],
        }

        result = predictor.predict(email_dict)

        # Ensure response matches schema
        return AnalysisResponse(
            score=result["score"],
            label=result["label"],
            reasons=result.get("reasons", []),
            flags=result.get("flags", []),
            senderAnalysis=SenderAnalysis(**result.get("senderAnalysis", {})),
            urlAnalysis=UrlAnalysis(**result.get("urlAnalysis", {"total": 0, "suspicious": 0, "urls": []})),
            confidence=result.get("confidence", 0.5),
            bertScore=result.get("bertScore", 0),
            structuralScore=result.get("structuralScore", 0),
            inferenceMs=result.get("inferenceMs", 0.0),
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "PhishGuard ML API", "docs": "/docs", "health": "/health"}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        reload=False,
        log_level="info",
    )
