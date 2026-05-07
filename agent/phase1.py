# Phase 1: photo analysis + attribute extraction via Vision API.
from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

from config.settings import settings
from config.category_config import (
    CATEGORY_DICT,
    CATEGORY_KEYWORDS,
    REQUIRED_ATTRS,
)
from features.extractor import CategoryFeatureExtractor
from agent.prompts import build_phase1_prompt

logger = logging.getLogger(__name__)

_VISION_MODEL = "claude-sonnet-4-20250514"
_VISION_MAX_TOKENS = 2000


class ClarificationAgent:
    def __init__(
        self,
        anthropic_client: Optional[AsyncAnthropic] = None,
        extractor_factory: Optional[Callable[[int], CategoryFeatureExtractor]] = None,
    ) -> None:
        self.client = anthropic_client or AsyncAnthropic(
            api_key=settings.anthropic_api_key,
        )
        self._extractor_factory = extractor_factory or CategoryFeatureExtractor

    async def analyze_photos_deep(
        self,
        photos: List[str],
    ) -> Dict[str, Any]:
        if not photos:
            return {
                "photo_analysis": "",
                "detected_category": None,
                "detected_attrs": {},
                "condition": "good",
                "condition_details": "",
            }

        content: List[Dict[str, Any]] = []
        for photo_b64 in photos[:5]:
            media_type = self._detect_media_type(photo_b64)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": photo_b64,
                },
            })

        content.append({
            "type": "text",
            "text": (
                "Ти експерт-оцінювач товарів для маркетплейсу Монобазарі.\n\n"
                "Уважно розглянь ВСІ фотографії та надай детальний аналіз:\n"
                "1. Що це за товар? (тип, бренд, модель якщо видно)\n"
                "2. Фізичний стан - чи є тріщини, подряпини, сколи, потертості, "
                "пошкодження екрану/корпусу/підошви? Опиши КОЖЕН дефект.\n"
                "3. Комплектність - чи є коробка, аксесуари, документи?\n"
                "4. Загальна оцінка стану СТРОГО один з:\n"
                "   - new: запечатаний, в плівці\n"
                "   - like_new: без жодних подряпин\n"
                "   - very_good: мінімальні сліди використання\n"
                "   - good: невеликі подряпини, робочий\n"
                "   - fair: помітні подряпини, потертості\n"
                "   - needs_repair: тріщини екрану, розбитий корпус, НЕ працює, пошкодження\n\n"
                "ВАЖЛИВО: якщо бачиш БУДЬ-ЯКІ тріщини на екрані або корпусі → needs_repair\n\n"
                "Відповідай ТІЛЬКИ JSON:\n"
                '{"photo_analysis":"Детальний опис стану 3-5 речень",'
                '"detected_category":"тип товару або null",'
                '"detected_attrs":{"brand":"","model":"","color":"","title":"назва книги","author":"автор","publisher":"видавництво"},'
                '"condition":"needs_repair",'
                '"condition_details":"Короткий опис стану 1 речення"}'
            ),
        })

        try:
            response = await self.client.messages.create(
                model=_VISION_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": content}],
            )

            raw_text = response.content[0].text
            parsed = self._parse_json_response(raw_text)

            return {
                "photo_analysis": parsed.get("photo_analysis", ""),
                "detected_category": parsed.get("detected_category"),
                "detected_attrs": parsed.get("detected_attrs", {}),
                "condition": parsed.get("condition", "good"),
                "condition_details": parsed.get("condition_details", ""),
            }

        except Exception as e:
            logger.error("Deep photo analysis failed: %s", e, exc_info=True)
            return {
                "photo_analysis": "",
                "detected_category": None,
                "detected_attrs": {},
                "condition": "good",
                "condition_details": "",
            }

    async def analyze_photos(
        self,
        description: str,
        photos: List[str],
        category_id: int,
        found_attrs: Dict[str, Any],
        missing_attrs: List[str],
    ) -> Dict[str, Any]:
        if not photos:
            return {
                "extracted_attrs": {},
                "still_missing": missing_attrs,
                "questions": [],
                "enriched_description": description,
                "confidence": "low",
            }

        category_name = CATEGORY_DICT.get(category_id, f"category_{category_id}")
        system_prompt = build_phase1_prompt(category_name, found_attrs, missing_attrs)

        content: List[Dict[str, Any]] = []

        detail_levels = self._photo_detail_strategy(len(photos))
        for i, photo_b64 in enumerate(photos):
            media_type = self._detect_media_type(photo_b64)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": photo_b64,
                },
            })

        text_parts = []
        if description:
            text_parts.append(f"Опис продавця: {description}")
        text_parts.append(
            "Проаналізуй фотографії та витягни атрибути товару. "
            "Відповідай ТІЛЬКИ у форматі JSON."
        )
        content.append({"type": "text", "text": "\n\n".join(text_parts)})

        try:
            response = await self.client.messages.create(
                model=_VISION_MODEL,
                max_tokens=_VISION_MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )

            raw_text = response.content[0].text
            parsed = self._parse_json_response(raw_text)

            return {
                "extracted_attrs": parsed.get("extracted_attrs", {}),
                "still_missing": parsed.get("still_missing", missing_attrs),
                "questions": parsed.get("questions", []),
                "enriched_description": parsed.get("enriched_description", description),
                "confidence": parsed.get("overall_confidence", "medium"),
            }

        except Exception as e:
            logger.error("Vision API call failed: %s", e, exc_info=True)
            return {
                "extracted_attrs": {},
                "still_missing": missing_attrs,
                "questions": [],
                "enriched_description": description,
                "confidence": "low",
            }

    async def run_phase1(
        self,
        description: str,
        photos: List[str],
        category_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if category_id is None:
            category_id = self._detect_category(description)
            if category_id is None:
                condition = self._infer_condition(description) if description else "good"
                return {
                    "category_id": None,
                    "category_name": "Інше",
                    "structured_attrs": {"condition": condition},
                    "missing_attrs": [],
                    "enriched_description": description,
                    "questions": [],
                    "confidence": "low",
                    "decision": "proceed",
                    "source": {},
                }

        category_name = CATEGORY_DICT.get(category_id, f"category_{category_id}")

        try:
            extractor = self._extractor_factory(category_id)
            regex_attrs = extractor.extract(description, "", 0)
            missing = extractor.get_missing(regex_attrs)
        except ValueError:
            regex_attrs = {}
            missing = list(REQUIRED_ATTRS.get(category_id, []))

        structured_attrs = {
            k: v for k, v in regex_attrs.items()
            if v is not None and not k.startswith(("title_", "desc_", "has_", "month", "day_"))
        }

        if "condition" not in structured_attrs or structured_attrs.get("condition") is None:
            structured_attrs["condition"] = self._infer_condition(description)

        source: Dict[str, str] = {k: "regex" for k in structured_attrs}

        vision_result = {
            "extracted_attrs": {},
            "still_missing": missing,
            "questions": [],
            "enriched_description": description,
            "confidence": "medium" if not missing else "low",
        }

        if photos and missing:
            vision_result = await self.analyze_photos(
                description=description,
                photos=photos,
                category_id=category_id,
                found_attrs=structured_attrs,
                missing_attrs=missing,
            )

        vision_attrs = vision_result.get("extracted_attrs", {})
        for key, val in vision_attrs.items():
            if val is not None and key not in structured_attrs:
                structured_attrs[key] = val
                source[key] = "vision"

        all_required = REQUIRED_ATTRS.get(category_id, [])
        still_missing = [
            a for a in all_required
            if a not in structured_attrs or structured_attrs[a] is None
        ]

        enriched_description = vision_result.get("enriched_description", description)
        questions = vision_result.get("questions", [])
        confidence = vision_result.get("confidence", "low")

        _NON_CRITICAL = {"color", "battery_pct", "gender", "cover_type", "language"}
        critical_missing = [a for a in still_missing if a not in _NON_CRITICAL]
        decision = "proceed" if len(critical_missing) == 0 else "questions"

        if decision == "questions" and not questions:
            questions = self._generate_default_questions(still_missing, category_name)

        return {
            "category_id": category_id,
            "category_name": category_name,
            "structured_attrs": structured_attrs,
            "missing_attrs": still_missing,
            "enriched_description": enriched_description,
            "questions": questions,
            "confidence": confidence,
            "decision": decision,
            "source": source,
        }

    @staticmethod
    def _infer_condition(text: str) -> str:
        t = text.lower()

        if any(w in t for w in (
            "тріщина", "розбит", "не працює", "зламан", "дефект камер",
            "не вмикається", "не включається", "битий", "розбитий",
        )):
            return "needs_repair"

        if any(w in t for w in (
            "потертості", "подряпини на екрані", "потертий корпус",
            "сліди використання", "видні подряпини",
        )):
            return "fair"

        if any(w in t for w in (
            "б/у", "вживан", "є подряпин", "невелик подряпин",
            "нормальний стан", "робочий стан",
        )):
            return "good"

        if any(w in t for w in (
            "як новий", "стан нового", "ідеальний", "відмінний",
            "майже новий", "без подряпин", "без дефектів",
        )):
            return "like_new"

        if any(w in t for w in (
            "новий", "запечатаний", "запакований", "в плівці", "new",
        )):
            return "new"

        return "good"

    def _detect_category(self, text: str) -> Optional[int]:
        if not text:
            return None

        text_lower = text.lower()
        best_id: Optional[int] = None
        best_score = 0

        for cat_id, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_id = cat_id

        return best_id if best_score > 0 else None

    @staticmethod
    def _photo_detail_strategy(count: int) -> List[str]:
        if count <= 2:
            return ["high"] * count
        if count == 3:
            return ["medium"] * 3
        return ["high"] + ["low"] * min(count - 1, 4)

    @staticmethod
    def _detect_media_type(b64_data: str) -> str:
        try:
            header = base64.b64decode(b64_data[:32])
            if header[:3] == b"\xff\xd8\xff":
                return "image/jpeg"
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                return "image/webp"
            if header[:3] == b"GIF":
                return "image/gif"
        except Exception:
            pass
        return "image/jpeg"

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response: %.200s", cleaned)
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def _generate_default_questions(
        missing: List[str],
        category_name: str,
    ) -> List[str]:
        attr_questions = {
            "model": "Яка точна модель вашого товару?",
            "brand": "Який бренд/виробник?",
            "storage_gb": "Який об'єм пам'яті (GB)?",
            "condition": "В якому стані товар? (новий, б/у, як новий)",
            "color": "Якого кольору товар?",
            "battery_pct": "Який відсоток здоров'я батареї?",
            "size_eu": "Який розмір (EU)?",
            "size": "Який розмір?",
            "size_r": "Який радіус (R)?",
            "gender": "Для кого: чоловічі, жіночі чи унісекс?",
            "set_number": "Який номер набору?",
            "piece_count": "Скільки деталей у наборі?",
            "theme": "Яка тема/серія?",
            "completeness": "Набір повний чи є відсутні деталі?",
            "author": "Хто автор?",
            "book_title": "Яка назва книги?",
            "title": "Яка назва книги?",
            "language": "Якою мовою?",
            "cover_type": "Тверда чи м'яка обкладинка?",
            "character": "Який персонаж зображений?",
            "series": "З якої серії/франшизи?",
            "franchise": "З якої франшизи?",
            "size_cm": "Який розмір (см)?",
            "tire_brand": "Який бренд шин?",
            "season": "Який сезон: літні, зимові, всесезонні?",
            "tread_mm": "Який залишок протектора (мм)?",
            "quantity": "Скільки штук у комплекті?",
            "type": "Який тип?",
            "material": "З якого матеріалу?",
        }

        questions = []
        for attr in missing[:3]:
            q = attr_questions.get(attr, f"Вкажіть, будь ласка, {attr}.")
            questions.append(q)
        return questions
