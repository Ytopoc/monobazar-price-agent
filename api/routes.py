# API endpoints: price estimation, clarification, photo analysis.
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas import (
    ClarificationResponse,
    ClarifyRequest,
    ComparableListing,
    HealthResponse,
    MarketContext,
    Metadata,
    MonobazarContext,
    OLXContext,
    PriceFactor,
    PriceRange,
    PricingRequest,
    PricingResponse,
    Strategy,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(request: Request) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status="ok",
        version="1.0.0",
        categories_loaded=len(getattr(state, "faiss_categories", [])),
        models_loaded=len(getattr(state, "predictors", {})),
        embedder_ready=getattr(state, "embedder_ready", False),
    )


@router.post(
    "/price",
    response_model=PricingResponse | ClarificationResponse,
    tags=["pricing"],
)
async def estimate_price(body: PricingRequest, request: Request):
    state = request.app.state
    _ensure_ready(state)

    clarification_agent = state.clarification_agent

    try:
        phase1_result = await clarification_agent.run_phase1(
            description=body.description,
            photos=body.photos,
            category_id=body.category_id,
        )
    except Exception as e:
        logger.error("Phase 1 failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phase 1 error: {e}",
        )

    if phase1_result["decision"] == "questions":
        return ClarificationResponse(
            needs_clarification=True,
            questions=phase1_result.get("questions", []),
            enriched_attrs=phase1_result.get("structured_attrs", {}),
            still_missing=phase1_result.get("missing_attrs", []),
            enriched_description=phase1_result.get("enriched_description", ""),
            category_id=phase1_result.get("category_id"),
            category_name=phase1_result.get("category_name", ""),
            confidence=phase1_result.get("confidence", "low"),
        )

    return await _run_phase2(state, phase1_result, body.photos)


@router.post(
    "/price/clarify",
    response_model=PricingResponse,
    tags=["pricing"],
)
async def clarify_and_price(body: ClarifyRequest, request: Request):
    state = request.app.state
    _ensure_ready(state)

    enriched_item = body.enriched_item

    attrs = enriched_item.get("structured_attrs", {})
    attrs.update(body.answers)
    enriched_item["structured_attrs"] = attrs

    _skip_keys = {"condition", "category", "photo_count", "is_set"}
    answer_values = [
        str(v) for k, v in body.answers.items()
        if v and k not in _skip_keys
    ]
    if answer_values:
        orig_desc = enriched_item.get("enriched_description", "")
        enriched_item["enriched_description"] = " ".join(
            [orig_desc] + answer_values
        ).strip()

    from config.category_config import REQUIRED_ATTRS
    category_id = enriched_item.get("category_id")
    if category_id is not None:
        all_required = REQUIRED_ATTRS.get(category_id, [])
        enriched_item["missing_attrs"] = [
            a for a in all_required
            if a not in attrs or attrs[a] is None
        ]

    if body.photos:
        enriched_item["photos"] = body.photos

    return await _run_phase2(state, enriched_item, body.photos)


@router.post("/analyze-photos", tags=["pricing"])
async def analyze_photos(body: PricingRequest, request: Request):
    state = request.app.state
    _ensure_ready(state)

    clarification_agent = state.clarification_agent

    try:
        result = await clarification_agent.analyze_photos_deep(
            photos=body.photos,
        )
    except Exception as e:
        logger.error("Photo analysis failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Photo analysis error: {e}",
        )

    return result


@router.post("/pre-analyze", tags=["pricing"])
async def pre_analyze(body: PricingRequest, request: Request):
    state = request.app.state
    _ensure_ready(state)

    clarification_agent = state.clarification_agent

    try:
        phase1_result = await clarification_agent.run_phase1(
            description=body.description,
            photos=body.photos,
            category_id=body.category_id,
        )
    except Exception as e:
        logger.error("Pre-analyze failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pre-analyze error: {e}",
        )

    return phase1_result


@router.post(
    "/price/quick",
    response_model=PricingResponse,
    tags=["pricing"],
)
async def quick_price(body: ClarifyRequest, request: Request):
    state = request.app.state
    _ensure_ready(state)

    enriched_item = body.enriched_item

    attrs = enriched_item.get("structured_attrs", {})
    attrs.update(body.answers)
    enriched_item["structured_attrs"] = attrs

    if body.photos:
        enriched_item["photos"] = body.photos

    return await _run_phase2_quick(state, enriched_item)


def _ensure_ready(state: Any) -> None:
    if not getattr(state, "embedder_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is starting up. Please retry in a moment.",
        )


async def _run_phase2(
    state: Any,
    enriched_item: Dict[str, Any],
    photos: List[str],
) -> PricingResponse:
    from agent.phase2 import PricingAgent

    category_id = enriched_item.get("category_id")
    predictors: Dict[int, Any] = getattr(state, "predictors", {})

    enriched_item["photos"] = photos or enriched_item.get("photos", [])

    if category_id is None or category_id not in predictors:
        logger.info(
            "No model for category %s — using AI-only pricing fallback",
            category_id,
        )
        pricing_agent = PricingAgent(
            faiss_manager=state.faiss_manager,
            predictor=None,
            embedder=state.embedder,
            olx_searcher=state.olx_searcher,
            anthropic_client=state.anthropic_client,
            category_stats={},
        )
        try:
            result = await pricing_agent.run_phase2_ai_only(enriched_item)
        except Exception as e:
            logger.error("AI-only Phase 2 failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Phase 2 error: {e}",
            )
        return _build_pricing_response(result, enriched_item, getattr(state, "photo_lookup", {}))

    pricing_agent = PricingAgent(
        faiss_manager=state.faiss_manager,
        predictor=predictors[category_id],
        embedder=state.embedder,
        olx_searcher=state.olx_searcher,
        anthropic_client=state.anthropic_client,
        category_stats=state.category_stats.get(category_id, {}),
    )

    try:
        result = await pricing_agent.run_phase2(enriched_item)
    except Exception as e:
        logger.error("Phase 2 failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phase 2 error: {e}",
        )

    return _build_pricing_response(result, enriched_item, getattr(state, "photo_lookup", {}))


async def _run_phase2_quick(
    state: Any,
    enriched_item: Dict[str, Any],
) -> PricingResponse:
    category_id = enriched_item.get("category_id")
    if category_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category could not be determined.",
        )

    predictors: Dict[int, Any] = getattr(state, "predictors", {})
    if category_id not in predictors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No trained model for category {category_id}.",
        )

    from agent.phase2 import PricingAgent

    pricing_agent = PricingAgent(
        faiss_manager=state.faiss_manager,
        predictor=predictors[category_id],
        embedder=state.embedder,
        category_stats=state.category_stats.get(category_id, {}),
    )

    try:
        result = await pricing_agent.run_phase2_quick(enriched_item)
    except Exception as e:
        logger.error("Phase 2 quick failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phase 2 quick error: {e}",
        )

    return _build_pricing_response(result, enriched_item, getattr(state, "photo_lookup", {}))


def _build_pricing_response(
    result: Dict[str, Any],
    enriched_item: Dict[str, Any],
    photo_lookup: Dict[str, list] | None = None,
) -> PricingResponse:
    prices = result.get("prices", {})
    days = result.get("days", {})

    price_range = PriceRange(
        min=prices.get("q25", 0),
        mid=prices.get("q50", 0),
        max=prices.get("q75", 0),
    )

    strategies = [
        Strategy(
            name=s.get("name", ""),
            price=s.get("price", 0),
            days_estimate=s.get("days_estimate", 0),
            description=s.get("description", ""),
        )
        for s in result.get("strategies", [])
    ]

    comps = result.get("comparables", {})
    sold_items = comps.get("sold", [])
    active_items = comps.get("active", [])

    _PHOTO_BASE = "https://resale-media.monobazar.com.ua/"
    _photo_lookup = photo_lookup or {}

    monobazar_comps = []
    for c in (sold_items + active_items)[:10]:
        ad_id = c.get("advertisement_id", "")
        s3_keys = _photo_lookup.get(ad_id, [])
        p_urls = [_PHOTO_BASE + k for k in s3_keys[:3]]
        monobazar_comps.append(ComparableListing(
            title=c.get("title", ""),
            price=c.get("original_price", 0),
            sold_price=c.get("sold_price"),
            status=c.get("status", "SOLD"),
            similarity=c.get("cosine_similarity", 0),
            photo_urls=p_urls,
            source="monobazar",
        ))
    sold_prices = [c.get("sold_price") or c.get("original_price", 0) for c in sold_items]
    active_prices = [c.get("original_price", 0) for c in active_items]

    monobazar = MonobazarContext(
        sold_count=len(sold_items),
        active_count=len(active_items),
        sold_median=_safe_median(sold_prices),
        sold_min=min(sold_prices) if sold_prices else 0,
        sold_max=max(sold_prices) if sold_prices else 0,
        active_median=_safe_median(active_prices),
        top_comparables=monobazar_comps[:5],
    )

    olx_raw = result.get("olx", {})
    olx_items = [
        ComparableListing(
            title=item.get("title", ""),
            price=item.get("price", 0),
            url=item.get("url"),
            source="olx",
        )
        for item in olx_raw.get("items", [])
    ]
    olx = OLXContext(
        status=olx_raw.get("status", "unavailable"),
        count=olx_raw.get("count", 0),
        median=olx_raw.get("median", 0),
        min=olx_raw.get("min", 0),
        max=olx_raw.get("max", 0),
        estimated_market=olx_raw.get("estimated_market", 0),
        items=olx_items,
    )

    price_factors = _extract_price_factors(result, enriched_item)

    missing = enriched_item.get("missing_attrs", [])
    assumptions = [
        f"Атрибут '{a}' не вказано -- використано значення за замовчуванням"
        for a in missing
    ] if missing else []

    _lt = result.get("listing_text", "")
    logger.info("Building response: listing_text=%s chars, source=%s",
                len(_lt), result.get("explanation_source", "?"))

    return PricingResponse(
        needs_clarification=False,
        price_range=price_range,
        strategies=strategies,
        recommendation=result.get("recommendation", ""),
        market_analysis=result.get("market_analysis", ""),
        condition_impact=result.get("condition_impact"),
        tips=result.get("tips", []),
        price_factors=price_factors,
        market_context=MarketContext(monobazar=monobazar, olx=olx),
        enriched_description=result.get("enriched_description", ""),
        structured_attrs=result.get("structured_attrs", {}),
        listing_text=_lt,
        assumptions=assumptions,
        confidence=result.get("confidence", 0),
        metadata=Metadata(
            model_version="1.0.0",
            category_id=result.get("category_id", 0),
            category_name=result.get("category_name", ""),
            explanation_source=result.get("explanation_source", "fallback"),
            confidence=result.get("confidence", 0),
        ),
    )


def _extract_price_factors(
    result: Dict[str, Any],
    enriched_item: Dict[str, Any],
) -> List[PriceFactor]:
    factors: List[PriceFactor] = []
    attrs = enriched_item.get("structured_attrs", {})

    condition = attrs.get("condition")
    if condition:
        impact = "positive" if condition in ("new", "like_new") else (
            "negative" if condition in ("fair", "needs_repair") else "neutral"
        )
        factors.append(PriceFactor(
            factor="Стан товару",
            impact=impact,
            description=f"Стан: {condition}",
        ))

    olx = result.get("olx", {})
    if olx.get("count", 0) > 0:
        factors.append(PriceFactor(
            factor="Ринкова пропозиція OLX",
            impact="neutral",
            description=f"{olx['count']} аналогів на OLX, медіана {olx.get('median', 0):.0f} грн",
        ))

    comps = result.get("comparables", {})
    if comps.get("sold", []):
        factors.append(PriceFactor(
            factor="Продані аналоги Monobazar",
            impact="neutral",
            description=f"{len(comps['sold'])} проданих аналогів у базі",
        ))

    confidence = result.get("confidence", 0)
    if confidence < 0.5:
        factors.append(PriceFactor(
            factor="Низька впевненість моделі",
            impact="negative",
            description="Недостатньо аналогів для точної оцінки",
        ))

    return factors


def _safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n % 2 == 1:
        return float(sorted_v[n // 2])
    return float((sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2)
