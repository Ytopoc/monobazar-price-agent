# Builds FAISS indexes and trains pricing models.
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.category_config import CATEGORY_DICT
from search.embedding import TextEmbedder
from search.faiss_index import FAISSIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_data(ads_csv: str, photos_csv: str) -> pd.DataFrame:
    logger.info("Loading advertisements from %s", ads_csv)
    ads = pd.read_csv(ads_csv, on_bad_lines="skip")
    logger.info("Loaded %d advertisements", len(ads))

    logger.info("Loading photos from %s", photos_csv)
    photos = pd.read_csv(photos_csv)
    photo_agg = (
        photos.groupby("advertisement_id")
        .agg(photo_count=("s3_key", "count"), photo_urls=("s3_key", list))
        .reset_index()
    )

    df = ads.merge(photo_agg, on="advertisement_id", how="left")
    df["photo_count"] = df["photo_count"].fillna(0).astype(int)

    before = len(df)
    df = df[df["original_price"] >= 10]
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    df = df.reset_index(drop=True)
    logger.info("Filtered %d -> %d rows", before, len(df))

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["modified_at"] = pd.to_datetime(df["modified_at"], errors="coerce", utc=True)
    df["days_to_sell"] = None
    sold_mask = df["status"] == "SOLD"
    df.loc[sold_mask, "days_to_sell"] = (
        (df.loc[sold_mask, "modified_at"] - df.loc[sold_mask, "created_at"]).dt.days
    )
    df["category_name"] = df["category_id"].map(CATEGORY_DICT)
    return df


def _compute_category_stats(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    stats = {}
    for cat_id in df["category_id"].unique():
        cat_df = df[df["category_id"] == cat_id]
        sold = cat_df[cat_df["status"] == "SOLD"]
        prices = cat_df["original_price"]
        sold_prices = sold["sold_price"].dropna()
        days = sold["days_to_sell"].dropna()

        stats[int(cat_id)] = {
            "price_min": float(prices.min()) if len(prices) > 0 else 10,
            "price_max": float(prices.max()) if len(prices) > 0 else 1_000_000,
            "price_median": float(prices.median()) if len(prices) > 0 else 0,
            "price_mean": float(prices.mean()) if len(prices) > 0 else 0,
            "sold_price_median": float(sold_prices.median()) if len(sold_prices) > 0 else 0,
            "avg_days_to_sell": float(days.mean()) if len(days) > 0 else 14.0,
            "bargain_rate": float(sold["sold_via_bargain"].mean()) if len(sold) > 0 else 0.0,
            "sold_ratio": len(sold) / len(cat_df) if len(cat_df) > 0 else 0.0,
            "n_listings": len(cat_df),
            "n_sold": len(sold),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS indexes and train pricing models."
    )
    parser.add_argument("--ads-csv", default="data/raw/hackaton_advertisements_with_id.csv")
    parser.add_argument("--photos-csv", default="data/raw/advertisement_photos.csv")
    parser.add_argument("--index-dir", default="data/indices")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--skip-faiss", action="store_true", help="Skip index building")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--categories", type=int, nargs="*", default=None,
                        help="Specific category IDs to process")
    args = parser.parse_args()

    t_start = time.time()

    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)
    df = _load_data(args.ads_csv, args.photos_csv)

    category_ids = args.categories or sorted(df["category_id"].unique())
    logger.info("Processing categories: %s", category_ids)

    cat_stats = _compute_category_stats(df)

    logger.info("=" * 60)
    logger.info("STEP 2: Loading embedding model")
    logger.info("=" * 60)
    embedder = TextEmbedder(model_name=args.model, batch_size=args.batch_size)
    embedder.load()

    manager = FAISSIndexManager()

    if not args.skip_faiss:
        logger.info("=" * 60)
        logger.info("STEP 3: Building FAISS indices")
        logger.info("=" * 60)

        df_filtered = df[df["category_id"].isin(category_ids)]
        manager.build_all(df_filtered, embedder)
        manager.save_all(args.index_dir)

        print("\n  FAISS INDEX STATISTICS")
        print("-" * 55)
        for cat_id, count in manager.stats().items():
            name = CATEGORY_DICT.get(cat_id, "?")
            print(f"  [{cat_id:>4d}] {name:<40s} {count:>7,} vectors")
        print(f"  {'TOTAL':<46s} {manager.total_size:>7,} vectors")
        print("-" * 55)
    else:
        logger.info("STEP 3: SKIPPED (--skip-faiss). Loading existing indices.")
        manager.load_all(args.index_dir)
        logger.info("Loaded %d indices, %d total vectors", len(manager.categories), manager.total_size)

    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("STEP 4: Training LightGBM models")
        logger.info("=" * 60)

        from ml.trainer import PricingModelTrainer
        import pickle

        trainer = PricingModelTrainer(faiss_manager=manager, embedder=embedder)
        all_metrics = {}

        for cat_id in category_ids:
            cat_id = int(cat_id)
            if cat_id not in manager.categories:
                logger.warning("No FAISS index for category %d, skipping training", cat_id)
                continue

            logger.info("=" * 50)
            logger.info("Training category %d: %s", cat_id, CATEGORY_DICT.get(cat_id, "?"))
            logger.info("=" * 50)

            cat_sold = df[
                (df["category_id"] == cat_id) &
                (df["status"] == "SOLD") &
                (df["sold_price"] > 0)
            ].reset_index(drop=True)

            if len(cat_sold) < 30:
                logger.warning(
                    "Category %d has only %d SOLD records, skipping training",
                    cat_id, len(cat_sold),
                )
                continue

            logger.info("  SOLD records: %d", len(cat_sold))

            X, y_price, y_days = trainer.build_training_features(
                cat_sold, category_id=cat_id, top_k=args.top_k,
            )

            models = trainer.train(
                X, y_price, y_days,
                category_id=cat_id,
                temporal_split_date="2026-03-01",
                created_at_series=cat_sold["created_at"].reset_index(drop=True),
            )

            if models:
                trainer.save_category_models(models, args.models_dir, cat_id)
                all_metrics[cat_id] = models.get("metrics", {})

        stats_path = Path(args.models_dir) / "category_stats.pkl"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(cat_stats, f)
        logger.info("Saved category stats to %s", stats_path)

        print("\n" + "=" * 60)
        print("  TRAINING SUMMARY")
        print("=" * 60)
        for cat_id, met in sorted(all_metrics.items()):
            name = CATEGORY_DICT.get(cat_id, "?")
            print(f"  [{cat_id:>4d}] {name}")
            print(f"         MAPE: {met.get('mape', 0):.1f}%  "
                  f"MAE: {met.get('mae', 0):,.0f} UAH  "
                  f"Coverage: {met.get('coverage_50', 0):.1f}%  "
                  f"Test: {met.get('test_size', 0)}")
        print("=" * 60)
    else:
        logger.info("STEP 4: SKIPPED (--skip-train)")

    elapsed = time.time() - t_start
    logger.info("PIPELINE COMPLETE in %.1f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
