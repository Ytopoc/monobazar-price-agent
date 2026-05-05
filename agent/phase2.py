# Phase 2: FAISS + OLX + LightGBM + LLM-generated explanation.
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from anthropic import AsyncAnthropic

from config.settings import settings
from config.category_config import CATEGORY_DICT
from features.comparable import compute_comparable_features, rank_comparables
from search.faiss_index import FAISSIndexManager
from search.embedding import TextEmbedder
from search.olx_search import OLXSearcher
from ml.predictor import PricingPredictor, _round_price
from agent.prompts import build_phase2_prompt, build_fallback

logger = logging.getLogger(__name__)

_EXPLAIN_MODEL = "claude-haiku-4-5-20251001"
_EXPLAIN_MAX_TOKENS = 800

_OLX_TIMEOUT = 4.0
_EXPLAIN_TIMEOUT = 8.0


class PricingAgent:
    def __init__(
        self,
        faiss_manager: FAISSIndexManager,
        predictor: Optional[PricingPredictor],
        embedder: TextEmbedder,
        olx_searcher: Optional[OLXSearcher] = None,
        anthropic_client: Optional[AsyncAnthropic] = None,
        category_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.faiss_manager = faiss_manager
        self.predictor = predictor
        self.embedder = embedder
        self.olx_searcher = olx_searcher or OLXSearcher()
        self.client = anthropic_client or AsyncAnthropic(
            api_key=settings.anthropic_api_key,
        )
        self.category_stats = category_stats or {}

    async def local_pipeline(self, enriched_item: Dict[str, Any]) -> Dict[str, Any]:
        category_id = enriched_item["category_id"]
        structured_attrs = enriched_item.get("structured_attrs", {})
        description = enriched_item.get("enriched_description", "")
        original_price = enriched_item.get("original_price", 0)

        search_text = description or " ".join(
            str(v) for v in structured_attrs.values() if v is not None
        )
        if not search_text.strip():
            search_text = CATEGORY_DICT.get(category_id, "item")

        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None, self.embedder.embed_query, search_text,
        )

        faiss_results = self.faiss_manager.search(category_id, query_vector, top_k=80)
        if not faiss_results:
            logger.warning("FAISS returned 0 results for category %d", category_id)

        faiss_results = [
            r for r in faiss_results
            if r.get("status") in ("SOLD", "ACTIVE", "RESERVED", "ORDER_PROCESSING")
        ]

        ranked = rank_comparables(faiss_results, structured_attrs)
        ranked = self._filter_bundles(ranked, category_id)

        sold = [r for r in ranked if r.get("status") == "SOLD"]
        active = [r for r in ranked if r.get("status") == "ACTIVE"]

        comp_features = compute_comparable_features(
            ranked[:10],
            category_stats=self.category_stats,
        )

        features = {**structured_attrs, **comp_features}

        if original_price and original_price > 0:
            features["original_price"] = original_price
        else:
            proxy_price = comp_features.get("comp_sold_median", 0)
            if proxy_price <= 0:
                proxy_price = self.category_stats.get("price_median", 0)
            features["original_price"] = proxy_price

        prediction = self.predictor.predict(features, category_id)

        return {
            "prediction": prediction,
            "sold_comparables": sold[:10],
            "active_comparables": active[:10],
            "comparable_features": comp_features,
            "ranked_results": ranked[:15],
        }

    async def olx_pipeline(self, enriched_item: Dict[str, Any]) -> Dict[str, Any]:
        category_id = enriched_item.get("category_id") or 0
        structured_attrs = enriched_item.get("structured_attrs", {})
        description = enriched_item.get("enriched_description", "")

        try:
            result = await self.olx_searcher.search_and_parse(
                structured_attrs=structured_attrs,
                category_id=category_id,
                enriched_description=description,
            )
            return result
        except Exception as e:
            logger.error("OLX pipeline error: %s", e, exc_info=True)
            return {
                "status": "unavailable",
                "olx_count": 0,
                "olx_median": 0.0,
                "olx_min": 0.0,
                "olx_max": 0.0,
                "olx_estimated_market": 0.0,
                "olx_items": [],
                "query_used": "",
            }

    async def generate_explanation(
        self,
        local_result: Dict[str, Any],
        olx_result: Dict[str, Any],
        enriched_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        prediction = local_result["prediction"]
        ranked = local_result.get("ranked_results", [])
        structured_attrs = enriched_item.get("structured_attrs", {})
        description = enriched_item.get("enriched_description", "")

        system_prompt = build_phase2_prompt(
            enriched_description=description,
            structured_attrs=structured_attrs,
            prediction=prediction,
            comparables=ranked[:3],
            olx_data=olx_result,
            category_stats=self.category_stats,
        )

        user_content: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                "На основі наданих ринкових даних, сформуй рекомендації "
                "щодо ціноутворення для цього товару. "
                "Відповідай ТІЛЬКИ у форматі JSON."
            ),
        }]

        if not settings.anthropic_api_key:
            logger.info("No ANTHROPIC_API_KEY — using fallback")
            return self._build_fallback_explanation(prediction, local_result, olx_result)

        try:
            logger.info("Calling LLM (%s) with timeout %.0fs...", _EXPLAIN_MODEL, _EXPLAIN_TIMEOUT)
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=_EXPLAIN_MODEL,
                    max_tokens=_EXPLAIN_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                ),
                timeout=_EXPLAIN_TIMEOUT,
            )

            raw_text = response.content[0].text
            parsed = self._parse_json_response(raw_text)

            validated = self._validate_explanation(parsed, prediction)
            validated["source"] = "llm"
            return validated

        except asyncio.TimeoutError:
            logger.warning("LLM explanation timed out (%.1fs)", _EXPLAIN_TIMEOUT)
            return self._build_fallback_explanation(prediction, local_result, olx_result)

        except Exception as e:
            logger.error("LLM explanation failed: %s", e, exc_info=True)
            return self._build_fallback_explanation(prediction, local_result, olx_result)

    async def run_phase2_quick(self, enriched_item: Dict[str, Any]) -> Dict[str, Any]:
        category_id = enriched_item["category_id"]
        category_name = enriched_item.get(
            "category_name", CATEGORY_DICT.get(category_id, "unknown"),
        )

        local_result = await self.local_pipeline(enriched_item)
        prediction = local_result["prediction"]
        fallback = self._build_fallback_explanation(
            prediction, local_result, self._empty_olx_result(),
        )

        return {
            "category_id": category_id,
            "category_name": category_name,
            "structured_attrs": enriched_item.get("structured_attrs", {}),
            "enriched_description": enriched_item.get("enriched_description", ""),
            "prices": {
                "fast": prediction["price_fast"],
                "balanced": prediction["price_balanced"],
                "max": prediction["price_max"],
                "q25": prediction["price_q25"],
                "q50": prediction["price_q50"],
                "q75": prediction["price_q75"],
            },
            "days": {
                "fast": prediction["days_fast"],
                "balanced": prediction["days_balanced"],
                "max": prediction["days_max"],
            },
            "confidence": prediction["confidence"],
            "strategies": fallback.get("strategies", []),
            "recommendation": fallback.get("recommendation", ""),
            "market_analysis": "",
            "condition_impact": None,
            "tips": [],
            "comparables": {
                "sold": local_result.get("sold_comparables", [])[:5],
                "active": local_result.get("active_comparables", [])[:5],
                "count": len(local_result.get("ranked_results", [])),
            },
            "olx": {
                "status": "pending",
                "count": 0, "median": 0, "min": 0, "max": 0,
                "estimated_market": 0, "items": [],
            },
            "explanation_source": "fallback",
        }

    async def run_phase2(self, enriched_item: Dict[str, Any]) -> Dict[str, Any]:
        category_id = enriched_item["category_id"]
        category_name = enriched_item.get(
            "category_name", CATEGORY_DICT.get(category_id, "unknown"),
        )

        olx_task = asyncio.create_task(self._olx_with_timeout(enriched_item))

        try:
            local_result = await self.local_pipeline(enriched_item)
        except Exception as e:
            logger.error("Local pipeline failed: %s", e, exc_info=True)
            local_result = self._empty_local_result()

        prediction = local_result["prediction"]

        llm_task = asyncio.create_task(
            self._generate_ai_texts(prediction, enriched_item)
        )

        olx_result, ai_texts = await asyncio.gather(
            olx_task, llm_task, return_exceptions=True,
        )

        if isinstance(olx_result, Exception):
            logger.error("OLX pipeline failed: %s", olx_result)
            olx_result = self._empty_olx_result()
        if isinstance(ai_texts, Exception):
            logger.warning("LLM texts failed: %s", ai_texts)
            ai_texts = None

        explanation = self._build_smart_explanation(
            prediction, local_result, olx_result, enriched_item,
        )
        if ai_texts:
            strategies = ai_texts.get("strategies")
            if strategies:
                for i, desc in enumerate(strategies[:3]):
                    if desc and i < len(explanation["strategies"]):
                        explanation["strategies"][i]["description"] = desc
            if ai_texts.get("listing_text"):
                explanation["listing_text"] = ai_texts["listing_text"]
            explanation["source"] = "template+ai"

        return {
            "category_id": category_id,
            "category_name": category_name,
            "structured_attrs": enriched_item.get("structured_attrs", {}),
            "enriched_description": enriched_item.get("enriched_description", ""),
            "prices": {
                "fast": prediction["price_fast"],
                "balanced": prediction["price_balanced"],
                "max": prediction["price_max"],
                "q25": prediction["price_q25"],
                "q50": prediction["price_q50"],
                "q75": prediction["price_q75"],
            },
            "days": {
                "fast": prediction["days_fast"],
                "balanced": prediction["days_balanced"],
                "max": prediction["days_max"],
            },
            "confidence": prediction["confidence"],
            "strategies": explanation.get("strategies", []),
            "recommendation": explanation.get("recommendation", ""),
            "market_analysis": explanation.get("market_analysis", ""),
            "condition_impact": explanation.get("condition_impact"),
            "tips": explanation.get("tips", []),
            "comparables": {
                "sold": local_result.get("sold_comparables", [])[:5],
                "active": local_result.get("active_comparables", [])[:5],
                "count": len(local_result.get("ranked_results", [])),
            },
            "olx": {
                "status": olx_result.get("status", "unavailable"),
                "count": olx_result.get("olx_count", 0),
                "median": olx_result.get("olx_median", 0),
                "min": olx_result.get("olx_min", 0),
                "max": olx_result.get("olx_max", 0),
                "estimated_market": olx_result.get("olx_estimated_market", 0),
                "items": olx_result.get("olx_items", [])[:5],
            },
            "listing_text": explanation.get("listing_text", ""),
            "explanation_source": explanation.get("source", "template"),
        }

    async def run_phase2_ai_only(self, enriched_item: Dict[str, Any]) -> Dict[str, Any]:
        description = enriched_item.get("enriched_description", "")
        structured_attrs = enriched_item.get("structured_attrs", {})
        condition = structured_attrs.get("condition", "good")

        olx_result = await self._olx_with_timeout(enriched_item)
        if isinstance(olx_result, Exception):
            olx_result = self._empty_olx_result()

        olx_median = olx_result.get("olx_median", 0)
        olx_min = olx_result.get("olx_min", 0)
        olx_max = olx_result.get("olx_max", 0)
        olx_count = olx_result.get("olx_count", 0)

        prices = {"fast": 0, "balanced": 0, "max": 0}
        recommendation = ""
        market_analysis = ""

        if olx_count > 0 and olx_median > 0:
            _COND_MULT = {
                "new": 1.0, "like_new": 1.0, "very_good": 1.0,
                "good": 1.0, "fair": 0.85, "needs_repair": 0.55,
            }
            cond_mult = _COND_MULT.get(condition, 1.0)

            bat_mult = 1.0
            battery_pct = enriched_item.get("structured_attrs", {}).get("battery_pct")
            if battery_pct is not None:
                try:
                    bp = float(battery_pct)
                    if bp < 70:
                        bat_mult = 0.85
                    elif bp < 85:
                        bat_mult = 0.93
                    elif bp < 93:
                        bat_mult = 0.97
                except (ValueError, TypeError):
                    pass

            balanced = olx_median * cond_mult * bat_mult

            if olx_min > 0 and olx_max > 0 and olx_count >= 2:
                prices["fast"] = olx_min * cond_mult * bat_mult
                prices["balanced"] = balanced
                prices["max"] = olx_max * cond_mult * bat_mult * 0.95
            else:
                prices["fast"] = balanced * 0.9
                prices["balanced"] = balanced
                prices["max"] = balanced * 1.15

            logger.info(
                "AI-only pricing (OLX-anchored): median=%.0f, "
                "cond=%s(x%.2f), fast=%.0f, balanced=%.0f, max=%.0f",
                olx_median, condition, cond_mult,
                prices["fast"], prices["balanced"], prices["max"],
            )

            olx_items_str = "; ".join(
                f"{it.get('title', '')} — {it.get('price', 0)}₴"
                for it in olx_result.get("olx_items", [])[:5]
            )
            try:
                explain_prompt = (
                    f"Товар: {description}\nСтан: {condition}\n"
                    f"Аналоги на OLX: {olx_count} шт, медіана {olx_median:.0f}₴, "
                    f"мін {olx_min:.0f}₴, макс {olx_max:.0f}₴.\n"
                    f"Приклади: {olx_items_str}\n\n"
                    f"Рекомендована ціна продажу: {balanced:.0f}₴.\n"
                    f"Напиши коротке пояснення (2-3 речення) чому така ціна, "
                    f"враховуючи стан і ринок.\n\n"
                    f"Відповідай JSON: "
                    f'{{"recommendation":"...","market_analysis":"..."}}'
                )
                resp = await asyncio.wait_for(
                    self.client.messages.create(
                        model=_EXPLAIN_MODEL, max_tokens=300,
                        messages=[{"role": "user", "content": explain_prompt}],
                    ),
                    timeout=_EXPLAIN_TIMEOUT,
                )
                parsed = self._parse_json_response(resp.content[0].text)
                recommendation = parsed.get("recommendation", "")
                market_analysis = parsed.get("market_analysis", "")
            except Exception as e:
                logger.warning("AI-only explanation failed: %s", e)
                recommendation = f"Ціна базується на {olx_count} аналогах з OLX."

        else:
            attrs_str = ", ".join(
                f"{k}: {v}" for k, v in structured_attrs.items() if v
            ) if structured_attrs else "невідомо"

            prompt = (
                f"Ти експерт з оцінки б/у товарів на вторинному ринку України.\n"
                f"Курс: 1 USD ≈ 41 UAH, 1 EUR ≈ 45 UAH.\n\n"
                f"Опис товару: {description}\n"
                f"Атрибути: {attrs_str}\n"
                f"Стан: {condition}\n\n"
                f"ІНСТРУКЦІЯ для оцінки:\n"
                f"1. Визнач приблизну роздрібну ціну НОВОГО товару в Україні (грн)\n"
                f"2. Б/у товар у хорошому стані коштує 60-75% від нового\n"
                f"3. fair (помітні дефекти) = 45-55% від нового\n"
                f"4. needs_repair (розбитий, не працює) = 25-40% від нового\n"
                f"5. Дай 3 ціни продажу на маркетплейсі:\n"
                f"   - price_fast: нижня межа (швидкий продаж)\n"
                f"   - price_balanced: ринкова ціна\n"
                f"   - price_max: верхня межа (якщо чекати)\n\n"
                f"Відповідай ТІЛЬКИ JSON:\n"
                f'{{"new_retail_price":число,"price_fast":число,"price_balanced":число,"price_max":число,'
                f'"recommendation":"пояснення","market_analysis":"аналіз ринку"}}'
            )

            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=_EXPLAIN_MODEL, max_tokens=500,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=_EXPLAIN_TIMEOUT,
                )
                parsed = self._parse_json_response(response.content[0].text)
                prices["fast"] = float(parsed.get("price_fast", 0))
                prices["balanced"] = float(parsed.get("price_balanced", 0))
                prices["max"] = float(parsed.get("price_max", 0))
                recommendation = parsed.get("recommendation", "")
                market_analysis = parsed.get("market_analysis", "")
            except Exception as e:
                logger.error("AI-only pricing failed: %s", e, exc_info=True)

        if prices["fast"] > prices["balanced"]:
            prices["fast"], prices["balanced"] = prices["balanced"], prices["fast"]
        if prices["balanced"] > prices["max"]:
            prices["balanced"], prices["max"] = prices["max"], prices["balanced"]

        from ml.predictor import _round_price

        return {
            "category_id": enriched_item.get("category_id", 0),
            "category_name": enriched_item.get("category_name", "Інше"),
            "structured_attrs": structured_attrs,
            "enriched_description": description,
            "prices": {
                "fast": _round_price(prices["fast"]),
                "balanced": _round_price(prices["balanced"]),
                "max": _round_price(prices["max"]),
                "q25": _round_price(prices["fast"]),
                "q50": _round_price(prices["balanced"]),
                "q75": _round_price(prices["max"]),
            },
            "days": {"fast": 3, "balanced": 7, "max": 14},
            "confidence": 0.5 if olx_count > 0 else 0.2,
            "strategies": [
                {
                    "name": "Швидкий продаж",
                    "price": _round_price(prices["fast"]),
                    "days_estimate": 3,
                    "description": "Ціна для швидкого продажу.",
                },
                {
                    "name": "Збалансована ціна",
                    "price": _round_price(prices["balanced"]),
                    "days_estimate": 7,
                    "description": "Оптимальна ринкова ціна.",
                },
                {
                    "name": "Максимальна ціна",
                    "price": _round_price(prices["max"]),
                    "days_estimate": 14,
                    "description": "Максимум, якщо готові чекати.",
                },
            ],
            "recommendation": recommendation,
            "market_analysis": market_analysis,
            "condition_impact": None,
            "tips": [],
            "comparables": {"sold": [], "active": [], "count": 0},
            "olx": {
                "status": olx_result.get("status", "unavailable"),
                "count": olx_result.get("olx_count", 0),
                "median": olx_result.get("olx_median", 0),
                "min": olx_result.get("olx_min", 0),
                "max": olx_result.get("olx_max", 0),
                "estimated_market": olx_result.get("olx_estimated_market", 0),
                "items": olx_result.get("olx_items", [])[:5],
            },
            "explanation_source": "ai_only",
        }

    async def _generate_ai_texts(
        self,
        prediction: Dict[str, Any],
        enriched_item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not settings.anthropic_api_key:
            return None

        description = enriched_item.get("enriched_description", "")
        structured_attrs = enriched_item.get("structured_attrs", {})
        condition = structured_attrs.get("condition", "good")

        skip_keys = {"condition", "photo_count", "category_id"}
        attrs_str = ", ".join(
            f"{k}: {v}" for k, v in structured_attrs.items()
            if v and k not in skip_keys
        ) or "невідомо"

        condition_details = enriched_item.get("condition_details", "")

        prompt = (
            f"Ти копірайтер маркетплейсу Монобазарі. Відповідай ТІЛЬКИ українською.\n\n"
            f"Товар: {description}\n"
            f"Атрибути: {attrs_str}\n"
            f"Стан: {condition}\n"
            + (f"Деталі стану (з фото): {condition_details}\n" if condition_details else "")
            + f"Ціни: швидкий {prediction['price_fast']}₴ (~{prediction['days_fast']}дн), "
            f"збалансована {prediction['price_balanced']}₴ (~{prediction['days_balanced']}дн), "
            f"макс {prediction['price_max']}₴ (~{prediction['days_max']}дн).\n\n"
            f"Зроби 2 речі:\n"
            f"1. strategies — 3 короткі описи цінових стратегій (по 1 речення).\n"
            f"2. listing_text — опис оголошення (3-5 рядків, через \\n).\n\n"
            f"СУВОРІ ПРАВИЛА для listing_text:\n"
            f"- НЕ повторюй характеристики (назву, модель, колір, пам'ять, батарею, ціну, стан, комплектність) — вони вже є окремо.\n"
            f"- Пиши ТІЛЬКИ ФАКТИ з опису та атрибутів. НЕ ВИГАДУЙ нічого.\n"
            f"- Якщо є 'Деталі стану (з фото)' — ОБОВ'ЯЗКОВО опиши ці пошкодження в тексті.\n"
            f"- НЕ вигадуй дефекти яких НЕМАЄ в описі чи деталях стану.\n"
            f"- НЕ пиши про чохол якщо в атрибутах немає has_case.\n"
            f"- НЕ пиши про торг, відправку, доставку — ти цього не знаєш.\n"
            f"- НЕ вигадуй імена людей, посилання, сервіси, назви магазинів.\n"
            f"- Опиши: що це за товар, його переваги, для кого підійде.\n"
            f"- Якщо стан needs_repair і в описі вказано що саме зламано — напиши це.\n"
            f"- Пиши коротко, по суті, живою українською мовою.\n\n"
            f"Відповідай ТІЛЬКИ JSON:\n"
            f'{{"strategies":["опис1","опис2","опис3"],"listing_text":"текст"}}'
        )

        try:
            logger.info("Calling LLM for listing texts (timeout 6s)...")
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=_EXPLAIN_MODEL,
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=6.0,
            )
            raw = response.content[0].text.strip()
            parsed = self._parse_json_response(raw)
            result = {}
            strategies = parsed.get("strategies", [])
            if isinstance(strategies, list) and len(strategies) >= 3:
                result["strategies"] = [str(d) for d in strategies[:3]]
            listing = parsed.get("listing_text", "")
            if listing and len(listing) > 20:
                result["listing_text"] = str(listing)
            if result:
                logger.info("LLM listing texts OK (strategies=%s, listing=%s)",
                            "strategies" in result, "listing_text" in result)
                return result
        except asyncio.TimeoutError:
            logger.warning("LLM listing texts timed out (4s)")
        except Exception as e:
            logger.warning("LLM listing texts failed: %s", e)
        return None

    async def _olx_with_timeout(
        self, enriched_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self.olx_pipeline(enriched_item),
                timeout=_OLX_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("OLX pipeline timed out (%.1fs)", _OLX_TIMEOUT)
            return self._empty_olx_result()
        except Exception as e:
            logger.error("OLX pipeline error: %s", e)
            return self._empty_olx_result()

    @staticmethod
    def _filter_bundles(
        ranked: List[Dict[str, Any]],
        category_id: int,
    ) -> List[Dict[str, Any]]:
        import re

        if category_id not in (795, 743, 1677):
            return ranked

        bundle_re = re.compile(
            r"\b(\d+)\s*(?:книг|книж|шт|штук|комплект|набір|серія|том)"
            r"|\bкомплект\b|\bнабір з\b|\bсерія\b|\bвсі\s+\d+"
            r"|\b(?:1-\d|повна\s+(?:серія|колекція))",
            re.IGNORECASE,
        )

        filtered = []
        for r in ranked:
            title = r.get("title", "")
            m = bundle_re.search(title)
            if m:
                num = m.group(1) if m.group(1) else None
                if num and int(num) > 1:
                    continue
                if not num:
                    continue
            filtered.append(r)

        return filtered if len(filtered) >= 3 else ranked

    def _validate_explanation(
        self,
        parsed: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        ml_prices = {
            "fast": prediction["price_fast"],
            "balanced": prediction["price_balanced"],
            "max": prediction["price_max"],
        }
        ml_days = {
            "fast": prediction["days_fast"],
            "balanced": prediction["days_balanced"],
            "max": prediction["days_max"],
        }

        strategies = parsed.get("strategies", [])
        strategy_map = {0: "fast", 1: "balanced", 2: "max"}

        for i, strategy in enumerate(strategies[:3]):
            key = strategy_map.get(i)
            if key:
                strategy["price"] = ml_prices[key]
                strategy["days_estimate"] = ml_days[key]

        while len(strategies) < 3:
            idx = len(strategies)
            key = strategy_map.get(idx, "balanced")
            strategies.append({
                "name": {
                    "fast": "Швидкий продаж",
                    "balanced": "Збалансована ціна",
                    "max": "Максимальна ціна",
                }.get(key, "Стратегія"),
                "price": ml_prices.get(key, 0),
                "days_estimate": ml_days.get(key, 0),
                "description": "",
            })

        prices = [s["price"] for s in strategies[:3]]
        prices.sort()
        for i, s in enumerate(strategies[:3]):
            s["price"] = _round_price(prices[i])

        return {
            "strategies": strategies[:3],
            "recommendation": parsed.get("recommendation", ""),
            "market_analysis": parsed.get("market_analysis", ""),
            "condition_impact": parsed.get("condition_impact"),
            "tips": parsed.get("tips", []),
        }

    def _build_smart_explanation(
        self,
        prediction: Dict[str, Any],
        local_result: Dict[str, Any],
        olx_result: Dict[str, Any],
        enriched_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        comp_count = len(local_result.get("ranked_results", []))
        sold_count = len(local_result.get("sold_comparables", []))
        structured_attrs = enriched_item.get("structured_attrs", {})
        condition = structured_attrs.get("condition", "good")

        olx_count = olx_result.get("olx_count", 0)
        olx_median = olx_result.get("olx_median", 0)
        olx_min = olx_result.get("olx_min", 0)
        olx_max = olx_result.get("olx_max", 0)

        p_fast = prediction["price_fast"]
        p_balanced = prediction["price_balanced"]
        p_max = prediction["price_max"]
        d_balanced = prediction["days_balanced"]

        cat_median = self.category_stats.get("price_median", 0)
        cat_days = self.category_stats.get("avg_days_to_sell", 14)
        cat_bargain = self.category_stats.get("bargain_rate", 0)

        rec_parts = []
        rec_parts.append(
            f"Рекомендуємо збалансовану ціну {p_balanced:.0f} грн — "
            f"оптимальне співвідношення ціни та швидкості продажу "
            f"(орієнтовно {d_balanced} днів)."
        )
        if comp_count > 0:
            rec_parts.append(
                f"Ціни розраховані на основі {comp_count} аналогічних товарів у базі"
                + (f" та {olx_count} оголошень на OLX." if olx_count > 0 else ".")
            )
        elif olx_count > 0:
            rec_parts.append(
                f"Ціни розраховані на основі {olx_count} оголошень на OLX."
            )
        recommendation = " ".join(rec_parts)

        market_parts = []
        if olx_count > 0 and olx_median > 0:
            market_parts.append(
                f"На OLX знайдено {olx_count} аналогів: "
                f"медіана {olx_median:.0f} грн"
            )
            if olx_count >= 2 and olx_min > 0 and olx_max > 0:
                market_parts[-1] += f" (діапазон {olx_min:.0f}–{olx_max:.0f} грн)"
            market_parts[-1] += "."

        if sold_count > 0:
            sold_prices = [
                r.get("sold_price") or r.get("original_price", 0)
                for r in local_result.get("sold_comparables", [])[:5]
                if (r.get("sold_price") or r.get("original_price", 0)) > 0
            ]
            if sold_prices:
                avg_sold = np.mean(sold_prices)
                market_parts.append(
                    f"Середня ціна проданих аналогів у базі: {avg_sold:.0f} грн."
                )

        if cat_median > 0:
            market_parts.append(
                f"Медіана по категорії: {cat_median:.0f} грн, "
                f"середній час продажу ~{cat_days:.0f} днів."
            )

        if cat_bargain > 0.05:
            market_parts.append(
                f"Торг у категорії: ~{cat_bargain * 100:.0f}% покупців торгуються."
            )

        market_analysis = " ".join(market_parts) if market_parts else ""

        _COND_LABELS = {
            "new": "Новий товар — ціна максимальна.",
            "like_new": "Стан «як новий» — ціна близька до нового.",
            "very_good": "Стан «дуже добрий» — мінімальна знижка за стан (~3%).",
            "good": "Стан «добрий» — помірна знижка за стан (~7%).",
            "fair": "Стан «задовільний» — помітні сліди використання, знижка ~15%.",
            "needs_repair": "Потребує ремонту — суттєва знижка (~40%).",
        }
        condition_impact = _COND_LABELS.get(condition)

        battery_pct = structured_attrs.get("battery_pct")
        if battery_pct is not None:
            try:
                bp = float(battery_pct)
                if bp < 70:
                    condition_impact = (condition_impact or "") + f" Батарея {bp:.0f}% — знижка ~15%."
                elif bp < 85:
                    condition_impact = (condition_impact or "") + f" Батарея {bp:.0f}% — знижка ~7%."
                elif bp < 93:
                    condition_impact = (condition_impact or "") + f" Батарея {bp:.0f}% — невелика знижка ~3%."
            except (ValueError, TypeError):
                pass

        tips = []
        tips = tips[:3]

        fast_desc = f"Швидкий продаж за {prediction['days_fast']} днів. Нижче за медіану — привабить покупців одразу."
        balanced_desc = f"Оптимальна ринкова ціна. Продаж орієнтовно за {d_balanced} днів."
        max_desc = f"Максимум, якщо готові чекати ~{prediction['days_max']} днів. Для терплячих продавців."

        return {
            "strategies": [
                {
                    "name": "Швидкий продаж",
                    "price": p_fast,
                    "days_estimate": prediction["days_fast"],
                    "description": fast_desc,
                },
                {
                    "name": "Збалансована ціна",
                    "price": p_balanced,
                    "days_estimate": d_balanced,
                    "description": balanced_desc,
                },
                {
                    "name": "Максимальна ціна",
                    "price": p_max,
                    "days_estimate": prediction["days_max"],
                    "description": max_desc,
                },
            ],
            "recommendation": recommendation,
            "market_analysis": market_analysis,
            "condition_impact": condition_impact,
            "tips": tips,
            "source": "template",
        }

    def _build_fallback_explanation(
        self,
        prediction: Dict[str, Any],
        local_result: Dict[str, Any],
        olx_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._build_smart_explanation(
            prediction, local_result, olx_result,
            {"structured_attrs": {}, "enriched_description": ""},
        )

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n")
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Phase2 JSON: %.200s", cleaned)
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def _empty_local_result() -> Dict[str, Any]:
        return {
            "prediction": {
                "price_q25": 0, "price_q50": 0, "price_q75": 0,
                "price_fast": 0, "price_balanced": 0, "price_max": 0,
                "days_fast": 0, "days_balanced": 0, "days_max": 0,
                "confidence": 0,
            },
            "sold_comparables": [],
            "active_comparables": [],
            "comparable_features": {},
            "ranked_results": [],
        }

    @staticmethod
    def _empty_olx_result() -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "olx_count": 0,
            "olx_median": 0.0,
            "olx_min": 0.0,
            "olx_max": 0.0,
            "olx_estimated_market": 0.0,
            "olx_items": [],
            "query_used": "",
        }
