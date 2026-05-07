# System prompts and templates for the LLM (Phase 1 and Phase 2).
from __future__ import annotations

from typing import Any, Dict, List


PHASE1_SYSTEM_PROMPT = """\
Ти -- AI-асистент Монобазарі. Твоя задача -- проаналізувати фотографії товару \
та витягти максимум структурованої інформації для точної оцінки ціни.

Категорія товару: {category_name}

Вже відомі атрибути (з тексту оголошення):
{found_attrs}

Ще потрібно визначити:
{missing_attrs}

ІНСТРУКЦІЇ:
1. Уважно розглянь ВСІ фотографії товару.
2. Визнач кожен з відсутніх атрибутів, який можеш побачити на фото.
3. Оціни загальний стан товару (condition): new, like_new, very_good, good, fair, needs_repair.
4. Напиши збагачений опис товару (enriched_description) -- 2-3 речення \
   українською, що описують товар на основі фото + наявної інформації. \
   Цей опис буде використовуватись для пошуку аналогів.
5. Для кожного знайденого атрибута вкажи рівень впевненості: high, medium, low.
6. Якщо не можеш визначити атрибут з фото -- не вигадуй, залиш його в still_missing.
7. Якщо потрібна додаткова інформація від продавця -- сформулюй 1-3 коротких \
   питання українською (questions).

ФОРМАТ ВІДПОВІДІ (тільки JSON, без markdown):
{{
  "extracted_attrs": {{
    "attribute_name": "value",
    ...
  }},
  "confidence": {{
    "attribute_name": "high|medium|low",
    ...
  }},
  "still_missing": ["attr1", "attr2"],
  "enriched_description": "Збагачений опис товару...",
  "questions": ["Питання 1?", "Питання 2?"],
  "overall_confidence": "high|medium|low"
}}"""


PHASE2_SYSTEM_PROMPT = """\
Ти експерт ціноутворення Монобазарі. Відповідай українською. Будь ЛАКОНІЧНИМ - description макс 1 речення.

ТОВАР: {enriched_description}
Атрибути: {structured_attrs}

ML-ЦІНИ: швидкий {price_fast}₴ (~{days_fast}дн), збалансована {price_balanced}₴ (~{days_balanced}дн), макс {price_max}₴ (~{days_max}дн). Впевненість: {confidence}

Схожі товари з бази ({comp_count}шт, УВАГА: можуть бути ІНШІ моделі/версії - не плутай з оцінюваним товаром!): {comparables}

OLX: {olx_status}, {olx_count}шт, медіана {olx_median}₴, діапазон {olx_min}-{olx_max}₴
Категорія: медіана {cat_price_median}₴, продаж ~{cat_avg_days}дн, торг {cat_bargain_rate}%

ВАЖЛИВО: Називай товар ТОЧНО як в описі/атрибутах. Не підміняй модель на модель з аналогів.

Відповідай ТІЛЬКИ JSON (без markdown):
{{"strategies":[{{"name":"Швидкий продаж","price":0,"days_estimate":0,"description":"1 речення"}},{{"name":"Збалансована ціна","price":0,"days_estimate":0,"description":"1 речення"}},{{"name":"Максимальна ціна","price":0,"days_estimate":0,"description":"1 речення"}}],"recommendation":"1-2 речення яку стратегію обрати","market_analysis":"1-2 речення аналіз ринку","condition_impact":"вплив стану або null","tips":["порада1","порада2"]}}"""


TEMPLATE_FALLBACK = """\
Рекомендуємо збалансовану ціну {price_balanced} грн - оптимальне \
співвідношення ціни та швидкості продажу (орієнтовно {days_balanced} днів). \
Ціни розраховані на основі {comp_count} аналогічних товарів у базі."""


def format_found_attrs(attrs: Dict[str, Any]) -> str:
    if not attrs:
        return "(нічого не знайдено)"
    lines = [f"- {k}: {v}" for k, v in attrs.items()
             if v is not None and not k.startswith(("title_", "desc_", "has_", "month", "day_"))]
    return "\n".join(lines) if lines else "(нічого не знайдено)"


def format_missing_attrs(missing: List[str]) -> str:
    if not missing:
        return "(всі атрибути знайдено)"
    return "\n".join(f"- {a}" for a in missing)


def format_comparables(items: List[Dict[str, Any]], max_items: int = 3) -> str:
    if not items:
        return "немає"
    parts = []
    for item in items[:max_items]:
        status = item.get("status", "?")
        price = item.get("sold_price") or item.get("original_price", 0)
        title = item.get("title", "")[:50]
        parts.append(f"{title} [{status}] {price:.0f}₴")
    return "; ".join(parts)


def build_phase1_prompt(
    category_name: str,
    found_attrs: Dict[str, Any],
    missing_attrs: List[str],
) -> str:
    return PHASE1_SYSTEM_PROMPT.format(
        category_name=category_name,
        found_attrs=format_found_attrs(found_attrs),
        missing_attrs=format_missing_attrs(missing_attrs),
    )


def build_phase2_prompt(
    enriched_description: str,
    structured_attrs: Dict[str, Any],
    prediction: Dict[str, Any],
    comparables: List[Dict[str, Any]],
    olx_data: Dict[str, Any],
    category_stats: Dict[str, Any],
) -> str:
    return PHASE2_SYSTEM_PROMPT.format(
        enriched_description=enriched_description,
        structured_attrs=format_found_attrs(structured_attrs),
        price_fast=prediction.get("price_fast", 0),
        days_fast=prediction.get("days_fast", 0),
        price_balanced=prediction.get("price_balanced", 0),
        days_balanced=prediction.get("days_balanced", 0),
        price_max=prediction.get("price_max", 0),
        days_max=prediction.get("days_max", 0),
        confidence=prediction.get("confidence", 0),
        comp_count=len(comparables),
        comparables=format_comparables(comparables),
        olx_status=olx_data.get("status", "n/a"),
        olx_count=olx_data.get("olx_count", 0),
        olx_median=olx_data.get("olx_median", 0),
        olx_min=olx_data.get("olx_min", 0),
        olx_max=olx_data.get("olx_max", 0),
        cat_price_median=category_stats.get("price_median", 0),
        cat_avg_days=category_stats.get("avg_days_to_sell", 14),
        cat_bargain_rate=round(category_stats.get("bargain_rate", 0) * 100, 1),
    )


def build_fallback(
    prediction: Dict[str, Any],
    comp_count: int,
    olx_count: int,
) -> str:
    return TEMPLATE_FALLBACK.format(
        price_balanced=prediction.get("price_balanced", 0),
        days_balanced=prediction.get("days_balanced", 0),
        comp_count=comp_count,
    )
