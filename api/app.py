# FastAPI application with resource initialization on startup.
from __future__ import annotations

import logging
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.routes import router
from config.settings import settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INDEX_DIR = _PROJECT_ROOT / "data" / "indices"
_MODELS_DIR = _PROJECT_ROOT / "data" / "models"
_STATS_PATH = _MODELS_DIR / "category_stats.pkl"
_PHOTOS_CSV = _PROJECT_ROOT / "data" / "raw" / "advertisement_photos.csv"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("=" * 60)
    logger.info("Starting Monobazar Pricing Agent...")
    logger.info("=" * 60)

    logger.info("Loading FAISS indices from %s", _INDEX_DIR)
    from search.faiss_index import FAISSIndexManager

    faiss_manager = FAISSIndexManager()
    if _INDEX_DIR.exists():
        faiss_manager.load_all(str(_INDEX_DIR))
        logger.info(
            "Loaded %d FAISS indices (%d total vectors)",
            len(faiss_manager.categories),
            faiss_manager.total_size,
        )
    else:
        logger.warning("Index directory not found: %s", _INDEX_DIR)

    app.state.faiss_manager = faiss_manager
    app.state.faiss_categories = faiss_manager.categories

    category_stats = {}
    if _STATS_PATH.exists():
        with open(_STATS_PATH, "rb") as f:
            category_stats = pickle.load(f)
        logger.info("Loaded category stats for %d categories", len(category_stats))
    else:
        logger.warning("Category stats not found: %s", _STATS_PATH)

    app.state.category_stats = category_stats

    photo_lookup: dict[str, list[str]] = {}
    if _PHOTOS_CSV.exists():
        import csv
        with open(_PHOTOS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ad_id = row.get("advertisement_id", "")
                s3_key = row.get("s3_key", "")
                if ad_id and s3_key:
                    photo_lookup.setdefault(ad_id, []).append(s3_key)
        logger.info("Loaded photo lookup: %d ads with photos", len(photo_lookup))
    else:
        logger.warning("Photos CSV not found: %s", _PHOTOS_CSV)

    app.state.photo_lookup = photo_lookup

    logger.info("Loading LightGBM models from %s", _MODELS_DIR)
    from ml.predictor import PricingPredictor

    predictors = {}
    for cat_id in faiss_manager.categories:
        model_path = _MODELS_DIR / f"{cat_id}.pkl"
        if model_path.exists():
            try:
                cat_stats = category_stats.get(cat_id, {})
                predictor = PricingPredictor.from_saved(_MODELS_DIR, cat_id, cat_stats)
                predictors[cat_id] = predictor
                logger.info("  Loaded model for category %d", cat_id)
            except Exception as e:
                logger.error("  Failed to load model for category %d: %s", cat_id, e)
        else:
            logger.warning("  No model file for category %d", cat_id)

    app.state.predictors = predictors
    logger.info("Loaded %d pricing models", len(predictors))

    logger.info("Loading embedding model: %s", settings.embedding_model_name)
    from search.embedding import TextEmbedder

    embedder = TextEmbedder(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )
    embedder.load()
    app.state.embedder = embedder
    app.state.embedder_ready = True
    logger.info("Embedder loaded (dim=%d)", embedder.embedding_dim)

    from anthropic import AsyncAnthropic
    from search.olx_search import OLXSearcher
    from agent.phase1 import ClarificationAgent

    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    olx_searcher = OLXSearcher(
        timeout=5.0,
        max_results=settings.olx_num_results,
    )
    clarification_agent = ClarificationAgent(
        anthropic_client=anthropic_client,
    )

    app.state.anthropic_client = anthropic_client
    app.state.olx_searcher = olx_searcher
    app.state.clarification_agent = clarification_agent

    logger.info("=" * 60)
    logger.info("Monobazar Pricing Agent READY")
    logger.info("  Categories: %s", faiss_manager.categories)
    logger.info("  Models: %s", sorted(predictors.keys()))
    logger.info("=" * 60)

    yield

    logger.info("Shutting down Monobazar Pricing Agent...")
    app.state.embedder_ready = False


def create_app() -> FastAPI:
    app = FastAPI(
        title="Monobazar Pricing Agent",
        description=(
            "AI-powered pricing recommendations for the Monobazar marketplace. "
            "Analyzes photos and descriptions to suggest optimal listing prices."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    _UI_DIR = _PROJECT_ROOT / "ui"

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(
            str(_UI_DIR / "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
