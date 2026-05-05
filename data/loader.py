# Loading and preparing listings and photo metadata.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.category_config import CATEGORY_DICT

logger = logging.getLogger(__name__)


def load_advertisements(path_part1: str | Path, path_part2: str | Path) -> pd.DataFrame:
    logger.info("Loading advertisements from %s and %s", path_part1, path_part2)
    df1 = pd.read_csv(path_part1, on_bad_lines="skip")
    df2 = pd.read_csv(path_part2, on_bad_lines="skip")
    df = pd.concat([df1, df2], ignore_index=True)
    logger.info("Loaded %d + %d = %d advertisements", len(df1), len(df2), len(df))
    return df


def load_photos(path: str | Path) -> pd.DataFrame:
    logger.info("Loading photos from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d photo records", len(df))
    return df


def load_and_merge(
    ads_path1: str | Path,
    ads_path2: str | Path,
    photos_path: str | Path,
) -> pd.DataFrame:
    ads = load_advertisements(ads_path1, ads_path2)

    photos = load_photos(photos_path)
    photo_agg = (
        photos
        .groupby("advertisement_id")
        .agg(
            photo_count=("s3_key", "count"),
            photo_urls=("s3_key", list),
        )
        .reset_index()
    )
    logger.info("Aggregated photos for %d unique advertisements", len(photo_agg))

    df = ads.merge(photo_agg, on="advertisement_id", how="left")
    df["photo_count"] = df["photo_count"].fillna(0).astype(int)
    df["photo_urls"] = df["photo_urls"].apply(lambda x: x if isinstance(x, list) else [])
    logger.info("After merge: %d rows", len(df))

    before = len(df)
    df = df[df["original_price"] >= 10]
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    df = df.reset_index(drop=True)
    logger.info("Filtered %d -> %d rows (removed price<10 and empty titles)", before, len(df))

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["modified_at"] = pd.to_datetime(df["modified_at"], errors="coerce", utc=True)
    df["days_to_sell"] = None
    sold_mask = df["status"] == "SOLD"
    df.loc[sold_mask, "days_to_sell"] = (
        (df.loc[sold_mask, "modified_at"] - df.loc[sold_mask, "created_at"]).dt.days
    )

    df["category_name"] = df["category_id"].map(CATEGORY_DICT)

    _print_stats(df)

    return df


def _print_stats(df: pd.DataFrame) -> None:
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  DATASET STATISTICS")
    print(f"{sep}")
    print(f"  Total rows:          {len(df):,}")
    print(f"  Unique ads:          {df['advertisement_id'].nunique():,}")
    print(f"  Date range:          {df['created_at'].min()} -> {df['created_at'].max()}")

    print(f"\n  --- Status breakdown ---")
    status_counts = df["status"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"  {status:<22s} {count:>7,}  ({pct:5.1f}%)")

    print(f"\n  --- Category breakdown ---")
    cat_counts = df.groupby(["category_id", "category_name"]).size().reset_index(name="count")
    cat_counts = cat_counts.sort_values("count", ascending=False)
    for _, row in cat_counts.iterrows():
        pct = row["count"] / len(df) * 100
        print(f"  [{row['category_id']:>4d}] {row['category_name']:<50s} {row['count']:>7,}  ({pct:5.1f}%)")

    print(f"\n  --- Price stats (original_price) ---")
    print(f"  Mean:   {df['original_price'].mean():>10,.1f} UAH")
    print(f"  Median: {df['original_price'].median():>10,.1f} UAH")
    print(f"  Min:    {df['original_price'].min():>10,.1f} UAH")
    print(f"  Max:    {df['original_price'].max():>10,.1f} UAH")

    sold = df[df["status"] == "SOLD"]
    if len(sold) > 0:
        print(f"\n  --- Sold items ---")
        print(f"  Total SOLD:          {len(sold):,}")
        days = sold["days_to_sell"].dropna()
        if len(days) > 0:
            print(f"  Avg days to sell:    {days.mean():.1f}")
            print(f"  Median days to sell: {days.median():.1f}")
        sold_with_price = sold[sold["sold_price"] > 0]
        if len(sold_with_price) > 0:
            discount = (
                (sold_with_price["original_price"] - sold_with_price["sold_price"])
                / sold_with_price["original_price"]
                * 100
            )
            print(f"  Avg discount:        {discount.mean():.1f}%")
            print(f"  Sold via bargain:    {sold['sold_via_bargain'].sum():,} "
                  f"({sold['sold_via_bargain'].mean() * 100:.1f}%)")

    print(f"\n  --- Photos ---")
    print(f"  Ads with photos:     {(df['photo_count'] > 0).sum():,} "
          f"({(df['photo_count'] > 0).mean() * 100:.1f}%)")
    print(f"  Avg photos/ad:       {df['photo_count'].mean():.1f}")
    print(f"  Max photos/ad:       {df['photo_count'].max()}")
    print(sep)
