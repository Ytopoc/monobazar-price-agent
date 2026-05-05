# OLX listings search via the official API with a DuckDuckGo fallback.
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

_MODEL_CONTEXT_RE = re.compile(
    r"(?:iPhone|Galaxy|Pixel|Redmi|POCO|Xiaomi|MacBook|iPad)\s*\d"
    r"|\b[Rr]\s*\d{2}\b"
    r"|\b\d{2,4}\s*(?:[ГгGg][БбBb]|GB|TB|MB)\b"
    r"|\b\d{4,6}\s*(?:деталей|pcs|pieces)\b",
    re.IGNORECASE,
)

_PRICE_WITH_CURRENCY_RE = re.compile(
    r"(\d[\d\s,\.]{0,12}\d)\s*(?:₴|грн\.?|uah|UAH)"
    r"|(?:₴|грн\.?)\s*(\d[\d\s,\.]{0,12}\d)",
)

_PRICE_KEYWORD_RE = re.compile(
    r"(?:від|ціна|цена|price|вартість)[:\s]*(\d[\d\s,\.]{0,12}\d)",
    re.IGNORECASE,
)


def _clean_price_str(raw: str) -> Optional[float]:
    cleaned = raw.replace(" ", "").replace(",", "").replace("\u00a0", "")
    if cleaned.count(".") == 1 and len(cleaned.split(".")[-1]) == 3:
        cleaned = cleaned.replace(".", "")
    try:
        val = float(cleaned)
        if 10 <= val <= 1_000_000:
            return val
    except ValueError:
        pass
    return None


def extract_price(text: str) -> Optional[float]:
    for m in _PRICE_WITH_CURRENCY_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        start = max(0, m.start() - 20)
        context = text[start:m.end() + 5]
        if _MODEL_CONTEXT_RE.search(context):
            continue
        val = _clean_price_str(raw)
        if val is not None:
            return val

    for m in _PRICE_KEYWORD_RE.finditer(text):
        raw = m.group(1)
        val = _clean_price_str(raw)
        if val is not None:
            return val

    return None


_STOP_WORDS = frozenset(
    "і в на з із за до від для або та що як цей ця це ці не"
    " a the and or for with from to of in on by is".split()
)


def build_olx_query(
    structured_attrs: Dict[str, Any],
    category_id: int,
    fallback_text: str = "",
) -> str:
    a = structured_attrs
    parts: List[str] = []

    if category_id == 4:
        model = a.get("model", "")
        storage = a.get("storage_gb")
        if model:
            parts.append(model)
        if storage:
            parts.append(f"{storage}GB")

    elif category_id == 512:
        brand = a.get("brand", "")
        model_line = a.get("model_line", "")
        size = a.get("size")
        if brand:
            parts.append(brand)
        if model_line:
            parts.append(model_line)
        parts.append("кросівки")
        if size:
            parts.append(str(int(size)))

    elif category_id == 795:
        author = a.get("author", "")
        title = a.get("book_title", "")
        if author:
            parts.append(author)
        if title:
            parts.append(title)
        if not parts:
            parts.append("книга")

    elif category_id == 1677:
        series = a.get("series", "")
        franchise = a.get("franchise", "")
        character = a.get("character", "")
        if series:
            parts.append(series)
        if franchise:
            parts.append(franchise)
        if character:
            parts.append(character)
        if not parts:
            parts.append("колекційна фігурка")
        else:
            parts.append("фігурка")

    elif category_id == 743:
        brand = a.get("brand", "")
        set_number = a.get("set_number", "")
        theme = a.get("theme", "")
        if brand:
            parts.append(brand)
        if set_number:
            parts.append(set_number)
        elif theme:
            parts.append(theme)
        if not brand and not set_number:
            parts.append("конструктор")

    elif category_id == 1320:
        brand = a.get("brand", "")
        material = a.get("material", "")
        chair_type = a.get("type", "")
        parts.append("стілець")
        if chair_type:
            parts.append(chair_type)
        if brand:
            parts.append(brand)
        if material:
            parts.append(material)

    elif category_id == 1261:
        size_r = a.get("size_r")
        brand = a.get("brand", "")
        season = a.get("season", "")
        parts.append("колеса")
        if size_r:
            parts.append(f"R{size_r}")
        if brand:
            parts.append(brand)
        if season:
            parts.append(season)

    query = " ".join(str(p) for p in parts if p).strip()

    if len(query) < 5 and fallback_text:
        words = [w for w in fallback_text.split() if w.lower() not in _STOP_WORDS and len(w) > 2]
        query = " ".join(words[:6])

    return query


_EMPTY_RESULT: Dict[str, Any] = {
    "status": "no_results",
    "olx_count": 0,
    "olx_median": 0.0,
    "olx_min": 0.0,
    "olx_max": 0.0,
    "olx_estimated_market": 0.0,
    "olx_items": [],
    "query_used": "",
}

_OLX_MARKET_FACTOR = 0.85

_OLX_API_URL = "https://www.olx.ua/api/v1/offers/"
_OLX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


class OLXSearcher:
    def __init__(self, timeout: float = 5.0, max_results: int = 10) -> None:
        self.timeout = timeout
        self.max_results = max_results

    async def search_and_parse(
        self,
        structured_attrs: Dict[str, Any],
        category_id: int,
        enriched_description: str = "",
    ) -> Dict[str, Any]:
        try:
            query = build_olx_query(structured_attrs, category_id, enriched_description)
            if not query or len(query.strip()) < 3:
                return {**_EMPTY_RESULT, "status": "no_query"}

            loop = asyncio.get_event_loop()
            try:
                parsed_items = await asyncio.wait_for(
                    loop.run_in_executor(None, self._olx_api_search, query),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("OLX API timeout for: %s", query)
                return {**_EMPTY_RESULT, "status": "timeout", "query_used": query}

            if not parsed_items:
                return {**_EMPTY_RESULT, "status": "no_results", "query_used": query}

            prices = np.array([item["price"] for item in parsed_items])
            median = float(np.median(prices))

            return {
                "status": "ok",
                "olx_count": len(parsed_items),
                "olx_median": median,
                "olx_min": float(np.min(prices)),
                "olx_max": float(np.max(prices)),
                "olx_estimated_market": round(median * _OLX_MARKET_FACTOR, 2),
                "olx_items": parsed_items[:5],
                "query_used": query,
            }

        except Exception as e:
            logger.error("OLX search error: %s", e, exc_info=True)
            return {**_EMPTY_RESULT, "status": "error", "query_used": ""}

    def search_sync(
        self,
        structured_attrs: Dict[str, Any],
        category_id: int,
        enriched_description: str = "",
    ) -> Dict[str, Any]:
        return asyncio.run(
            self.search_and_parse(structured_attrs, category_id, enriched_description)
        )

    def _olx_api_search(self, query: str) -> List[Dict[str, Any]]:
        logger.info("OLX API search: %s", query)
        try:
            resp = requests.get(
                _OLX_API_URL,
                params={"query": query, "offset": 0, "limit": self.max_results},
                headers=_OLX_HEADERS,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data", [])
            if not items:
                logger.info("OLX API: no results for '%s'", query)
                return self._ddg_fallback(query)

            parsed: List[Dict[str, Any]] = []
            for item in items:
                title = item.get("title", "")
                url = item.get("url", "")

                price = None
                for p in item.get("params", []):
                    if p.get("key") == "price":
                        price_val = p.get("value", {})
                        raw = price_val.get("value")
                        if raw is not None:
                            try:
                                price = float(raw)
                            except (ValueError, TypeError):
                                pass
                        if price is None:
                            label = price_val.get("label", "")
                            price = extract_price(label)
                        break

                if price is None or price < 10 or price > 1_000_000:
                    continue

                parsed.append({
                    "title": title.strip(),
                    "price": price,
                    "url": url,
                })

            parsed.sort(key=lambda x: x["price"])
            logger.info("OLX API: found %d items with prices for '%s'", len(parsed), query)
            return parsed

        except Exception as e:
            logger.warning("OLX API failed (%s), trying DDG fallback", e)
            return self._ddg_fallback(query)

    def _ddg_fallback(self, query: str) -> List[Dict[str, Any]]:
        full_query = f"site:olx.ua {query}"
        logger.info("DDG fallback search: %s", full_query)
        try:
            from ddgs import DDGS
            raw_results = DDGS().text(full_query, max_results=5, backend="duckduckgo,google")
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                raw_results = DDGS().text(full_query, max_results=3)
            except Exception as e:
                logger.error("DDG fallback failed: %s", e)
                return []
        except Exception as e:
            logger.error("DDG fallback failed: %s", e)
            return []

        if not raw_results:
            return []

        parsed: List[Dict[str, Any]] = []
        for r in raw_results:
            url = r.get("href", "")
            title = r.get("title", "")
            body = r.get("body", "")

            if "olx.ua" not in url:
                continue

            combined_text = f"{title} {body}"
            price = extract_price(combined_text)
            if price is None:
                continue

            parsed.append({
                "title": title.strip(),
                "price": price,
                "url": url,
            })

        parsed.sort(key=lambda x: x["price"])
        return parsed
