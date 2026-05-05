# Tests for comparable-feature computation and ranking.
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.comparable import (
    compute_comparable_features,
    rank_comparables,
    MIN_SOLD_COUNT,
)

NOW = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)


def _sold(price_orig, price_sold, sim, days_to_sell=10, bargain=False, days_ago=5, attrs=None):
    return {
        "status": "SOLD",
        "original_price": price_orig,
        "sold_price": price_sold,
        "cosine_similarity": sim,
        "days_to_sell": days_to_sell,
        "sold_via_bargain": bargain,
        "modified_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "structured_attrs": attrs or {},
    }


def _active(price_orig, sim, days_ago=2, attrs=None):
    return {
        "status": "ACTIVE",
        "original_price": price_orig,
        "sold_price": None,
        "cosine_similarity": sim,
        "days_to_sell": None,
        "sold_via_bargain": False,
        "modified_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "structured_attrs": attrs or {},
    }


def _deleted(price_orig, sim, days_ago=30, attrs=None):
    return {
        "status": "DELETED",
        "original_price": price_orig,
        "sold_price": None,
        "cosine_similarity": sim,
        "days_to_sell": None,
        "sold_via_bargain": False,
        "modified_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "structured_attrs": attrs or {},
    }


def test_basic_sold_stats():
    print("\n=== test_basic_sold_stats ===")
    results = [
        _sold(25000, 24000, 0.92, days_to_sell=7, bargain=True),
        _sold(23000, 23000, 0.88, days_to_sell=14, bargain=False),
        _sold(27000, 25000, 0.85, days_to_sell=5, bargain=True),
        _sold(22000, 21000, 0.80, days_to_sell=21, bargain=True),
        _active(26000, 0.90),
        _active(24500, 0.87),
        _deleted(30000, 0.75),
    ]
    f = compute_comparable_features(results)

    import numpy as np
    sold_prices = np.array([24000, 23000, 25000, 21000], dtype=float)

    checks = {
        "comp_sold_median": float(np.median(sold_prices)),
        "comp_sold_mean": float(np.mean(sold_prices)),
        "comp_sold_min": 21000.0,
        "comp_sold_max": 25000.0,
        "comp_sold_count": 4,
        "comp_active_count": 2,
        "comp_active_median": 25250.0,
        "comp_sold_ratio": 4 / 7,
        "comp_avg_days_to_sell": (7 + 14 + 5 + 21) / 4,
        "comp_bargain_rate": 3 / 4,
    }

    ok = True
    for k, expected in checks.items():
        actual = f.get(k)
        if isinstance(expected, float):
            if abs(actual - expected) > 0.01:
                print(f"  FAIL [{k}]: expected={expected:.4f}, got={actual:.4f}")
                ok = False
        elif actual != expected:
            print(f"  FAIL [{k}]: expected={expected!r}, got={actual!r}")
            ok = False

    disc = f["comp_avg_discount"]
    expected_disc = ((25000 - 24000) / 25000 + (27000 - 25000) / 27000 + (22000 - 21000) / 22000) / 3
    if abs(disc - expected_disc) > 0.001:
        print(f"  FAIL [comp_avg_discount]: expected={expected_disc:.4f}, got={disc:.4f}")
        ok = False

    sims = [0.92, 0.88, 0.85, 0.80, 0.90, 0.87, 0.75]
    if abs(f["comp_avg_similarity"] - np.mean(sims)) > 0.001:
        print(f"  FAIL [comp_avg_similarity]")
        ok = False
    if abs(f["comp_top1_similarity"] - 0.92) > 0.001:
        print(f"  FAIL [comp_top1_similarity]")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_basic_sold_stats")
    return ok


def test_too_few_sold_fallback():
    print("\n=== test_too_few_sold_fallback ===")
    results = [
        _sold(10000, 9500, 0.90, days_to_sell=5),
        _active(11000, 0.85),
        _deleted(12000, 0.70),
    ]
    cat_stats = {
        "comp_sold_median": 9000.0,
        "comp_sold_mean": 9200.0,
        "comp_sold_std": 1500.0,
        "comp_sold_min": 5000.0,
        "comp_sold_max": 15000.0,
    }
    f = compute_comparable_features(results, category_stats=cat_stats)

    ok = True
    for k in ("comp_sold_median", "comp_sold_mean", "comp_sold_std", "comp_sold_min", "comp_sold_max"):
        if f[k] != cat_stats[k]:
            print(f"  FAIL [{k}]: expected fallback {cat_stats[k]}, got {f[k]}")
            ok = False

    if f["comp_sold_count"] != 1:
        print(f"  FAIL [comp_sold_count]: expected 1, got {f['comp_sold_count']}")
        ok = False
    if f["comp_active_count"] != 1:
        print(f"  FAIL [comp_active_count]")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_too_few_sold_fallback")
    return ok


def test_empty_results():
    print("\n=== test_empty_results ===")
    f = compute_comparable_features([])
    ok = all(v == 0.0 or v == 0 for v in f.values())
    if not ok:
        print(f"  Non-zero values in empty result: {f}")
    print(f"  {'PASS' if ok else 'FAIL'} test_empty_results")
    return ok


def test_no_sold_no_active():
    print("\n=== test_no_sold_no_active ===")
    results = [_deleted(5000, 0.80), _deleted(6000, 0.75), _deleted(7000, 0.70)]
    f = compute_comparable_features(results)

    ok = True
    if f["comp_sold_count"] != 0:
        print(f"  FAIL sold_count")
        ok = False
    if f["comp_active_count"] != 0:
        print(f"  FAIL active_count")
        ok = False
    if f["comp_sold_ratio"] != 0.0:
        print(f"  FAIL sold_ratio")
        ok = False
    if f["comp_avg_similarity"] == 0.0:
        print(f"  FAIL avg_similarity should be >0")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_no_sold_no_active")
    return ok


def test_all_sold_no_bargain():
    print("\n=== test_all_sold_no_bargain ===")
    results = [
        _sold(10000, 10000, 0.95, bargain=False),
        _sold(11000, 11000, 0.90, bargain=False),
        _sold(12000, 12000, 0.85, bargain=False),
    ]
    f = compute_comparable_features(results)

    ok = True
    if f["comp_bargain_rate"] != 0.0:
        print(f"  FAIL bargain_rate={f['comp_bargain_rate']}")
        ok = False
    if f["comp_avg_discount"] != 0.0:
        print(f"  FAIL avg_discount={f['comp_avg_discount']}")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_all_sold_no_bargain")
    return ok


def test_rank_basic_ordering():
    print("\n=== test_rank_basic_ordering ===")
    results = [
        _sold(10000, 9500, 0.70, days_ago=30),
        _active(11000, 0.90, days_ago=1),
        _deleted(12000, 0.60, days_ago=90),
        _sold(10500, 10000, 0.88, days_ago=3),
    ]
    query_attrs = {}
    ranked = rank_comparables(results, query_attrs, now=NOW)

    ok = True
    if ranked[0]["sold_price"] != 10000:
        print(f"  FAIL: top result should be sold_price=10000, got {ranked[0].get('sold_price')}")
        ok = False
    if ranked[-1]["status"] != "DELETED":
        print(f"  FAIL: last result should be DELETED, got {ranked[-1]['status']}")
        ok = False
    if not all("hybrid_score" in r for r in ranked):
        print(f"  FAIL: missing hybrid_score keys")
        ok = False
    scores = [r["hybrid_score"] for r in ranked]
    if scores != sorted(scores, reverse=True):
        print(f"  FAIL: scores not descending: {scores}")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_rank_basic_ordering")
    return ok


def test_rank_structured_boost():
    print("\n=== test_rank_structured_boost ===")
    query_attrs = {"model": "iPhone 14 Pro", "brand": "Apple", "storage_gb": 256}

    exact_match = _sold(
        25000, 24000, 0.85, days_ago=5,
        attrs={"model": "iPhone 14 Pro", "brand": "Apple", "storage_gb": 256},
    )
    partial_match = _sold(
        25000, 24000, 0.85, days_ago=5,
        attrs={"model": "iPhone 14", "brand": "Apple", "storage_gb": 128},
    )
    no_match = _sold(
        25000, 24000, 0.85, days_ago=5,
        attrs={"model": "iPhone 13", "brand": "Apple", "storage_gb": 128},
    )

    ranked = rank_comparables([no_match, exact_match, partial_match], query_attrs, now=NOW)

    ok = True
    if ranked[0]["structured_attrs"]["model"] != "iPhone 14 Pro":
        print(f"  FAIL: top should be exact match, got {ranked[0]['structured_attrs']}")
        ok = False
    if ranked[0]["attr_boost"] <= ranked[1]["attr_boost"]:
        print(f"  FAIL: exact boost {ranked[0]['attr_boost']} <= partial {ranked[1]['attr_boost']}")
        ok = False

    expected_boost = 1.3 * 1.2 * 1.15
    if abs(ranked[0]["attr_boost"] - expected_boost) > 0.01:
        print(f"  FAIL: exact boost expected={expected_boost:.3f}, got={ranked[0]['attr_boost']:.3f}")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_rank_structured_boost")
    return ok


def test_rank_freshness_decay():
    print("\n=== test_rank_freshness_decay ===")
    import math

    fresh = _sold(10000, 9500, 0.80, days_ago=1)
    stale = _sold(10000, 9500, 0.80, days_ago=120)

    ranked = rank_comparables([stale, fresh], {}, now=NOW)

    ok = True
    if ranked[0]["freshness_score"] <= ranked[1]["freshness_score"]:
        print(f"  FAIL: fresh={ranked[0]['freshness_score']:.4f} <= stale={ranked[1]['freshness_score']:.4f}")
        ok = False

    expected_fresh = math.exp(-1 / 60)
    expected_stale = math.exp(-120 / 60)
    if abs(ranked[0]["freshness_score"] - expected_fresh) > 0.01:
        print(f"  FAIL: fresh score expected={expected_fresh:.4f}, got={ranked[0]['freshness_score']:.4f}")
        ok = False
    if abs(ranked[1]["freshness_score"] - expected_stale) > 0.01:
        print(f"  FAIL: stale score expected={expected_stale:.4f}, got={ranked[1]['freshness_score']:.4f}")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_rank_freshness_decay")
    return ok


def test_rank_empty():
    print("\n=== test_rank_empty ===")
    ranked = rank_comparables([], {}, now=NOW)
    ok = ranked == []
    print(f"  {'PASS' if ok else 'FAIL'} test_rank_empty")
    return ok


def test_rank_score_components():
    print("\n=== test_rank_score_components ===")
    r = _sold(10000, 9000, 0.90, days_ago=10, attrs={"brand": "Apple"})
    query_attrs = {"brand": "Apple"}

    ranked = rank_comparables([r], query_attrs, now=NOW)
    entry = ranked[0]

    import math
    expected_sim = 0.90
    expected_status = 1.0
    expected_fresh = math.exp(-10 / 60)
    expected_boost = 1.2
    expected_score = (0.5 * expected_sim + 0.3 * expected_status + 0.2 * expected_fresh) * expected_boost

    ok = True
    if abs(entry["hybrid_score"] - expected_score) > 0.001:
        print(f"  FAIL: score expected={expected_score:.4f}, got={entry['hybrid_score']:.4f}")
        ok = False
    if abs(entry["similarity_score"] - expected_sim) > 0.001:
        print(f"  FAIL: sim component")
        ok = False
    if abs(entry["status_score"] - expected_status) > 0.001:
        print(f"  FAIL: status component")
        ok = False
    if abs(entry["attr_boost"] - expected_boost) > 0.001:
        print(f"  FAIL: boost component")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'} test_rank_score_components")
    return ok


if __name__ == "__main__":
    tests = [
        test_basic_sold_stats,
        test_too_few_sold_fallback,
        test_empty_results,
        test_no_sold_no_active,
        test_all_sold_no_bargain,
        test_rank_basic_ordering,
        test_rank_structured_boost,
        test_rank_freshness_decay,
        test_rank_empty,
        test_rank_score_components,
    ]

    passed = sum(1 for t in tests if t())
    total = len(tests)

    print(f"\n{'=' * 50}")
    print(f"  TOTAL: {passed}/{total} passed")
    print(f"{'=' * 50}")
