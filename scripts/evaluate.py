# Model quality evaluation on a held-out test set.
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.category_config import CATEGORY_DICT
from features.extractor import CategoryFeatureExtractor
from features.comparable import compute_comparable_features, rank_comparables
from search.faiss_index import FAISSIndexManager
from ml.predictor import PricingPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_data(ads_csv: str, photos_csv: str) -> pd.DataFrame:
    logger.info("Loading data from %s", ads_csv)
    ads = pd.read_csv(ads_csv, on_bad_lines="skip")

    photos = pd.read_csv(photos_csv)
    photo_agg = (
        photos.groupby("advertisement_id")
        .agg(photo_count=("s3_key", "count"))
        .reset_index()
    )

    df = ads.merge(photo_agg, on="advertisement_id", how="left")
    df["photo_count"] = df["photo_count"].fillna(0).astype(int)

    df = df[df["original_price"] >= 10]
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    df = df.reset_index(drop=True)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["modified_at"] = pd.to_datetime(df["modified_at"], errors="coerce", utc=True)
    df["days_to_sell"] = None
    sold_mask = df["status"] == "SOLD"
    df.loc[sold_mask, "days_to_sell"] = (
        (df.loc[sold_mask, "modified_at"] - df.loc[sold_mask, "created_at"]).dt.days
    )
    return df


def _evaluate_record(
    row: pd.Series,
    category_id: int,
    faiss_manager: FAISSIndexManager,
    predictor: PricingPredictor,
    extractor: CategoryFeatureExtractor,
    embedder: Any,
    category_stats: Dict[str, Any],
    top_k: int = 30,
) -> Dict[str, Any]:
    ad_id = row["advertisement_id"]
    sold_price = float(row["sold_price"])

    title = str(row.get("title", ""))
    desc = str(row.get("description", ""))[:300]
    search_text = f"{title}. {desc}"

    query_vec = embedder.embed_query(search_text)

    raw_results = faiss_manager.search(category_id, query_vec, top_k=top_k + 5)
    neighbours = [r for r in raw_results if r.get("advertisement_id") != ad_id][:top_k]

    struct_attrs = extractor.extract(
        title, str(row.get("description", "")),
        photo_count=int(row.get("photo_count", 0)),
        created_at=row.get("created_at"),
    )
    ranked = rank_comparables(neighbours, struct_attrs)

    comp_feats = compute_comparable_features(neighbours, category_stats=category_stats)

    features = {**struct_attrs, **comp_feats}
    features["original_price"] = float(row["original_price"])

    prediction = predictor.predict(features, category_id)

    pred_q25 = prediction["price_q25"]
    pred_q50 = prediction["price_q50"]
    pred_q75 = prediction["price_q75"]

    in_interval = pred_q25 <= sold_price <= pred_q75
    ape = abs(pred_q50 - sold_price) / max(sold_price, 1) * 100

    return {
        "advertisement_id": ad_id,
        "sold_price": sold_price,
        "original_price": float(row["original_price"]),
        "pred_q25": pred_q25,
        "pred_q50": pred_q50,
        "pred_q75": pred_q75,
        "pred_fast": prediction["price_fast"],
        "pred_balanced": prediction["price_balanced"],
        "pred_max": prediction["price_max"],
        "in_interval": in_interval,
        "ape": ape,
        "ae": abs(pred_q50 - sold_price),
        "confidence": prediction["confidence"],
    }


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {"mape": 0, "mae": 0, "coverage": 0, "count": 0,
                "median_ape": 0, "mae_median": 0, "avg_confidence": 0}

    apes = [r["ape"] for r in results]
    aes = [r["ae"] for r in results]
    coverages = [r["in_interval"] for r in results]

    return {
        "mape": float(np.mean(apes)),
        "mae": float(np.mean(aes)),
        "median_ape": float(np.median(apes)),
        "coverage": float(np.mean(coverages) * 100),
        "count": len(results),
        "mae_median": float(np.median(aes)),
        "avg_confidence": float(np.mean([r["confidence"] for r in results])),
    }


def _save_scatter_plot(
    all_results: Dict[int, List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    categories = sorted(all_results.keys())
    n_cats = len(categories)
    if n_cats == 0:
        return

    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, cat_id in enumerate(categories):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        results = all_results[cat_id]
        actual = [rec["sold_price"] for rec in results]
        predicted = [rec["pred_q50"] for rec in results]

        ax.scatter(actual, predicted, alpha=0.4, s=15, c="steelblue", edgecolors="none")

        all_vals = actual + predicted
        lo = min(all_vals) * 0.9 if all_vals else 0
        hi = max(all_vals) * 1.1 if all_vals else 1
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, alpha=0.7, label="Ideal")

        xs = np.linspace(lo, hi, 100)
        ax.fill_between(xs, xs * 0.8, xs * 1.2, alpha=0.08, color="green", label="+/-20%")

        cat_name = CATEGORY_DICT.get(cat_id, str(cat_id)).split("/")[-1]
        metrics = _compute_metrics(results)
        ax.set_title(
            f"[{cat_id}] {cat_name}\n"
            f"MAPE={metrics['mape']:.1f}% | MAE={metrics['mae']:,.0f} | "
            f"Cov={metrics['coverage']:.0f}% | n={metrics['count']}",
            fontsize=9,
        )
        ax.set_xlabel("Actual sold price (UAH)", fontsize=8)
        ax.set_ylabel("Predicted q50 (UAH)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")

    for idx in range(n_cats, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Monobazar Pricing Agent -- Predicted vs Actual", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    logger.info("Scatter plot saved to %s", output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pricing pipeline on test data.")
    parser.add_argument("--ads-csv", default="data/raw/hackaton_advertisements_with_id.csv")
    parser.add_argument("--photos-csv", default="data/raw/advertisement_photos.csv")
    parser.add_argument("--index-dir", default="data/indices")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--output-dir", default="data/evaluation")
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--split-date", default="2026-03-01")
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-per-category", type=int, default=None,
                        help="Max test records per category (for faster runs)")
    args = parser.parse_args()

    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)
    df = _load_data(args.ads_csv, args.photos_csv)
    logger.info("Total records: %d", len(df))

    logger.info("=" * 60)
    logger.info("STEP 2: Loading FAISS indices")
    logger.info("=" * 60)
    faiss_manager = FAISSIndexManager()
    faiss_manager.load_all(args.index_dir)
    logger.info("Loaded %d indices (%d vectors)", len(faiss_manager.categories), faiss_manager.total_size)

    stats_path = Path(args.models_dir) / "category_stats.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            all_category_stats = pickle.load(f)
        logger.info("Loaded category stats for %d categories", len(all_category_stats))
    else:
        all_category_stats = {}
        logger.warning("No category_stats.pkl found at %s", stats_path)

    logger.info("=" * 60)
    logger.info("STEP 3: Loading LightGBM models")
    logger.info("=" * 60)
    predictors: Dict[int, PricingPredictor] = {}
    for cat_id in faiss_manager.categories:
        model_path = Path(args.models_dir) / f"{cat_id}.pkl"
        if model_path.exists():
            cat_stats = all_category_stats.get(cat_id, {})
            predictors[cat_id] = PricingPredictor.from_saved(args.models_dir, cat_id, cat_stats)
            logger.info("  Loaded predictor for category %d", cat_id)
    logger.info("Loaded %d predictors", len(predictors))

    logger.info("=" * 60)
    logger.info("STEP 4: Loading embedding model")
    logger.info("=" * 60)
    from search.embedding import TextEmbedder
    embedder = TextEmbedder(model_name=args.model, batch_size=args.batch_size)
    embedder.load()

    logger.info("=" * 60)
    logger.info("STEP 5: Filtering test set (SOLD after %s)", args.split_date)
    logger.info("=" * 60)

    split_ts = pd.Timestamp(args.split_date, tz="UTC")
    test_df = df[
        (df["status"] == "SOLD") &
        (df["sold_price"] > 0) &
        (df["created_at"] >= split_ts)
    ].reset_index(drop=True)
    logger.info("Test set: %d SOLD records after %s", len(test_df), args.split_date)

    category_ids = args.categories or sorted(test_df["category_id"].unique())
    category_ids = [c for c in category_ids if c in predictors]
    logger.info("Evaluating categories: %s", category_ids)

    logger.info("=" * 60)
    logger.info("STEP 6: Running evaluation")
    logger.info("=" * 60)

    all_results: Dict[int, List[Dict[str, Any]]] = {}
    all_metrics: Dict[int, Dict[str, float]] = {}

    for cat_id in category_ids:
        cat_test = test_df[test_df["category_id"] == cat_id].reset_index(drop=True)
        if len(cat_test) == 0:
            logger.warning("No test records for category %d, skipping", cat_id)
            continue

        if args.max_per_category and len(cat_test) > args.max_per_category:
            cat_test = cat_test.sample(args.max_per_category, random_state=42).reset_index(drop=True)

        cat_name = CATEGORY_DICT.get(cat_id, "?")
        logger.info("-" * 50)
        logger.info("Category %d: %s (%d test records)", cat_id, cat_name, len(cat_test))
        logger.info("-" * 50)

        try:
            extractor = CategoryFeatureExtractor(cat_id)
        except ValueError:
            logger.error("No extractor for category %d, skipping", cat_id)
            continue

        predictor = predictors[cat_id]
        cat_stats = all_category_stats.get(cat_id, {})
        results: List[Dict[str, Any]] = []
        t_cat = time.time()

        for i, (_, row) in enumerate(cat_test.iterrows()):
            try:
                rec = _evaluate_record(
                    row, cat_id, faiss_manager, predictor,
                    extractor, embedder, cat_stats, top_k=args.top_k,
                )
                rec["category_id"] = cat_id
                results.append(rec)
            except Exception as e:
                logger.warning(
                    "  Error evaluating ad %s: %s",
                    row.get("advertisement_id", "?"), e,
                )

            if (i + 1) % 50 == 0 or i == len(cat_test) - 1:
                elapsed = time.time() - t_cat
                speed = (i + 1) / max(elapsed, 0.01)
                logger.info(
                    "  [%d/%d] %.1f rec/s | MAPE so far: %.1f%%",
                    i + 1, len(cat_test), speed,
                    _compute_metrics(results)["mape"],
                )

        all_results[cat_id] = results
        metrics = _compute_metrics(results)
        all_metrics[cat_id] = metrics

        logger.info(
            "  => MAPE=%.1f%% MAE=%.0f Coverage=%.1f%% (n=%d, %.1fs)",
            metrics["mape"], metrics["mae"], metrics["coverage"],
            metrics["count"], time.time() - t_cat,
        )

    all_recs = [r for recs in all_results.values() for r in recs]
    overall = _compute_metrics(all_recs)

    print()
    print("=" * 80)
    print("  EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"  {'Category':<45s} {'MAPE':>7s} {'MAE':>9s} {'MedAPE':>7s} "
          f"{'Cov%':>6s} {'Conf':>5s} {'N':>5s}")
    print("  " + "-" * 78)

    for cat_id in sorted(all_metrics.keys()):
        m = all_metrics[cat_id]
        cat_name = CATEGORY_DICT.get(cat_id, "?")
        display_name = cat_name if len(cat_name) <= 40 else cat_name[:37] + "..."
        print(
            f"  [{cat_id:>4d}] {display_name:<38s} "
            f"{m['mape']:>6.1f}% {m['mae']:>8,.0f} {m['median_ape']:>6.1f}% "
            f"{m['coverage']:>5.1f}% {m['avg_confidence']:>5.2f} {m['count']:>5d}"
        )

    print("  " + "-" * 78)
    print(
        f"  {'OVERALL':<45s} "
        f"{overall['mape']:>6.1f}% {overall['mae']:>8,.0f} {overall['median_ape']:>6.1f}% "
        f"{overall['coverage']:>5.1f}% {overall['avg_confidence']:>5.2f} {overall['count']:>5d}"
    )
    print("=" * 80)

    plot_path = output_dir / "predicted_vs_actual.png"
    _save_scatter_plot(all_results, plot_path)

    if all_recs:
        detail_df = pd.DataFrame(all_recs)
        detail_path = output_dir / "evaluation_details.csv"
        detail_df.to_csv(str(detail_path), index=False)
        logger.info("Detailed results saved to %s", detail_path)

    summary = {
        "split_date": args.split_date,
        "top_k": args.top_k,
        "overall": overall,
        "per_category": {
            str(k): v for k, v in all_metrics.items()
        },
    }
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s", summary_path)

    elapsed = time.time() - t_start
    logger.info("EVALUATION COMPLETE in %.1f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
