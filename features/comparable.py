# Comparable-features computation and ranking of FAISS results.
from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, float] = {
    "comp_sold_median": 0.0,
    "comp_sold_mean": 0.0,
    "comp_sold_std": 0.0,
    "comp_sold_min": 0.0,
    "comp_sold_max": 0.0,
    "comp_sold_count": 0,
    "comp_active_median": 0.0,
    "comp_active_count": 0,
    "comp_sold_ratio": 0.0,
    "comp_avg_days_to_sell": 0.0,
    "comp_bargain_rate": 0.0,
    "comp_avg_discount": 0.0,
    "comp_avg_similarity": 0.0,
    "comp_top1_similarity": 0.0,
}

MIN_SOLD_COUNT = 1


def compute_comparable_features(
    search_results: List[Dict[str, Any]],
    category_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if not search_results:
        return dict(category_stats) if category_stats else dict(_DEFAULTS)

    features: Dict[str, float] = {}

    sims = [r["cosine_similarity"] for r in search_results if r.get("cosine_similarity") is not None]
    features["comp_avg_similarity"] = float(np.mean(sims)) if sims else 0.0
    features["comp_top1_similarity"] = float(sims[0]) if sims else 0.0

    sold = [r for r in search_results if r.get("status") == "SOLD"]
    active = [r for r in search_results if r.get("status") == "ACTIVE"]
    total = len(search_results)

    features["comp_sold_count"] = len(sold)
    features["comp_active_count"] = len(active)
    features["comp_sold_ratio"] = len(sold) / total if total > 0 else 0.0

    sold_prices = _extract_prices(sold, key="sold_price")

    if len(sold_prices) >= MIN_SOLD_COUNT:
        features["comp_sold_median"] = float(np.median(sold_prices))
        features["comp_sold_mean"] = float(np.mean(sold_prices))
        features["comp_sold_std"] = float(np.std(sold_prices, ddof=1)) if len(sold_prices) > 1 else 0.0
        features["comp_sold_min"] = float(np.min(sold_prices))
        features["comp_sold_max"] = float(np.max(sold_prices))
    else:
        fallback = category_stats if category_stats else _DEFAULTS
        for k in ("comp_sold_median", "comp_sold_mean", "comp_sold_std",
                   "comp_sold_min", "comp_sold_max"):
            features[k] = fallback.get(k, 0.0)

    active_prices = _extract_prices(active, key="original_price")
    features["comp_active_median"] = float(np.median(active_prices)) if len(active_prices) > 0 else 0.0

    days = [r["days_to_sell"] for r in sold
            if r.get("days_to_sell") is not None and r["days_to_sell"] >= 0]
    features["comp_avg_days_to_sell"] = float(np.mean(days)) if days else 0.0

    if sold:
        bargain_count = sum(1 for r in sold if r.get("sold_via_bargain"))
        features["comp_bargain_rate"] = bargain_count / len(sold)
    else:
        features["comp_bargain_rate"] = 0.0

    discounts = []
    for r in sold:
        if r.get("sold_via_bargain"):
            orig = r.get("original_price")
            sp = r.get("sold_price")
            if orig and sp and orig > 0:
                discounts.append((orig - sp) / orig)
    features["comp_avg_discount"] = float(np.mean(discounts)) if discounts else 0.0

    return features


_STATUS_WEIGHT: Dict[str, float] = {
    "SOLD": 1.0,
    "ACTIVE": 0.7,
    "RESERVED": 0.6,
    "ORDER_PROCESSING": 0.5,
    "DELETED": 0.3,
}

_FRESHNESS_DECAY = 60.0

_W_SIM = 0.5
_W_STATUS = 0.3
_W_FRESH = 0.2

_ATTR_BOOSTS: Dict[str, float] = {
    "model": 1.3,
    "brand": 1.2,
    "storage_gb": 1.15,
    "condition": 1.1,
    "size": 1.1,
    "size_eu": 1.1,
    "set_number": 1.3,
    "book_title": 1.3,
    "series": 1.15,
    "tire_brand": 1.15,
    "size_r": 1.1,
}


def rank_comparables(
    search_results: List[Dict[str, Any]],
    query_attrs: Dict[str, Any],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    if now is None:
        now = datetime.now(timezone.utc)

    scored: List[Dict[str, Any]] = []
    for r in search_results:
        entry = dict(r)

        sim = float(r.get("cosine_similarity", 0.0))
        status = r.get("status", "DELETED")
        sw = _STATUS_WEIGHT.get(status, 0.3)
        freshness = _compute_freshness(r, now)

        base = _W_SIM * sim + _W_STATUS * sw + _W_FRESH * freshness
        boost = _compute_attr_boost(r, query_attrs)
        hybrid = base * boost

        entry["hybrid_score"] = hybrid
        entry["similarity_score"] = sim
        entry["status_score"] = sw
        entry["freshness_score"] = freshness
        entry["attr_boost"] = boost
        scored.append(entry)

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored


def _extract_prices(
    results: List[Dict[str, Any]],
    key: str = "sold_price",
) -> np.ndarray:
    prices = []
    for r in results:
        val = r.get(key)
        if val is not None:
            try:
                fval = float(val)
                if fval > 0:
                    prices.append(fval)
            except (ValueError, TypeError):
                pass
    return np.array(prices, dtype=np.float64) if prices else np.array([], dtype=np.float64)


def _compute_freshness(result: Dict[str, Any], now: datetime) -> float:
    raw = result.get("modified_at")
    if raw is None:
        return 0.5
    try:
        if isinstance(raw, str):
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00").strip())
        elif isinstance(raw, datetime):
            dt = raw
        else:
            return 0.5
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days_ago = max((now - dt).total_seconds() / 86400, 0)
        return math.exp(-days_ago / _FRESHNESS_DECAY)
    except (ValueError, TypeError, OverflowError):
        return 0.5


def _compute_attr_boost(
    result: Dict[str, Any],
    query_attrs: Dict[str, Any],
) -> float:
    result_attrs = result.get("structured_attrs", {})
    if not result_attrs or not query_attrs:
        return 1.0

    boost = 1.0
    for attr, multiplier in _ATTR_BOOSTS.items():
        q_val = query_attrs.get(attr)
        r_val = result_attrs.get(attr)
        if q_val is not None and r_val is not None:
            if str(q_val).lower().strip() == str(r_val).lower().strip():
                boost *= multiplier
    return boost
