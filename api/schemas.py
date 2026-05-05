# Pydantic schemas for API requests and responses.
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PricingRequest(BaseModel):
    description: str = Field(
        default="",
        description="Seller-provided listing description. Can be empty if photos are provided.",
    )
    photos: List[str] = Field(
        default_factory=list,
        description="Base64-encoded images (JPEG/PNG/WebP). Max 5.",
    )
    category_id: Optional[int] = Field(
        None,
        description="Category ID. Auto-detected if omitted.",
    )


class ClarifyRequest(BaseModel):
    enriched_item: Dict[str, Any] = Field(
        ...,
        description="Enriched item dict returned from Phase 1.",
    )
    answers: Dict[str, Any] = Field(
        ...,
        description="Attribute answers from the seller, e.g. {'color': 'black', 'storage_gb': 256}.",
    )
    photos: List[str] = Field(
        default_factory=list,
        description="Optional additional photos (base64).",
    )


class PriceRange(BaseModel):
    min: float = Field(..., description="Q25 price (lower bound).")
    mid: float = Field(..., description="Q50 price (median).")
    max: float = Field(..., description="Q75 price (upper bound).")


class Strategy(BaseModel):
    name: str = Field(..., description="Strategy name (e.g. 'Швидкий продаж').")
    price: float = Field(..., description="Recommended price in UAH.")
    days_estimate: float = Field(..., description="Estimated days to sell.")
    description: str = Field("", description="Explanation of this strategy.")


class ComparableListing(BaseModel):
    title: str
    price: float
    sold_price: Optional[float] = None
    status: str = "ACTIVE"
    similarity: float = 0.0
    url: Optional[str] = None
    photo_urls: List[str] = Field(default_factory=list, description="S3 photo URLs for this listing.")
    source: str = Field("monobazar", description="'monobazar' or 'olx'.")


class MonobazarContext(BaseModel):
    sold_count: int = 0
    active_count: int = 0
    sold_median: float = 0.0
    sold_min: float = 0.0
    sold_max: float = 0.0
    active_median: float = 0.0
    top_comparables: List[ComparableListing] = Field(default_factory=list)


class OLXContext(BaseModel):
    status: str = "unavailable"
    count: int = 0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    estimated_market: float = 0.0
    items: List[ComparableListing] = Field(default_factory=list)


class MarketContext(BaseModel):
    monobazar: MonobazarContext = Field(default_factory=MonobazarContext)
    olx: OLXContext = Field(default_factory=OLXContext)


class PriceFactor(BaseModel):
    factor: str
    impact: str = Field(..., description="'positive', 'negative', or 'neutral'.")
    description: str = ""


class Metadata(BaseModel):
    model_version: str = "1.0.0"
    category_id: Optional[int] = None
    category_name: str = ""
    explanation_source: str = Field("fallback", description="'llm' or 'fallback'.")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ClarificationResponse(BaseModel):
    needs_clarification: bool = True
    questions: List[str] = Field(default_factory=list)
    enriched_attrs: Dict[str, Any] = Field(default_factory=dict)
    still_missing: List[str] = Field(default_factory=list)
    enriched_description: str = ""
    category_id: Optional[int] = None
    category_name: str = ""
    confidence: str = "low"


class PricingResponse(BaseModel):
    needs_clarification: bool = False

    price_range: PriceRange
    strategies: List[Strategy] = Field(default_factory=list)

    recommendation: str = ""
    market_analysis: str = ""
    condition_impact: Optional[str] = None
    tips: List[str] = Field(default_factory=list)
    price_factors: List[PriceFactor] = Field(default_factory=list)
    market_context: MarketContext = Field(default_factory=MarketContext)

    enriched_description: str = ""
    structured_attrs: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    listing_text: str = ""

    confidence: float = Field(0.0, ge=0.0, le=1.0)
    metadata: Metadata


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    categories_loaded: int = 0
    models_loaded: int = 0
    embedder_ready: bool = False
