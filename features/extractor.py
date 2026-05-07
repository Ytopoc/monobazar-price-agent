# Per-category regex-based attribute extraction.
from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.category_config import REQUIRED_ATTRS

logger = logging.getLogger(__name__)

CONDITION_PRIORITY: Dict[str, int] = {
    "needs_repair": 1,
    "fair": 2,
    "good": 3,
    "very_good": 4,
    "like_new": 5,
    "new": 6,
}

_CONDITION_PATTERNS: List[tuple[str, str]] = [
    (r"потребує\s*ремонту", "needs_repair"),
    (r"на\s*запчастин", "needs_repair"),
    (r"задовільн", "fair"),
    (r"нормальн", "fair"),
    (r"гарн(?:ий|ому|а|е)", "good"),
    (r"хорош(?:ий|ому|а|е)", "good"),
    (r"(?:стан\s*)?(?:9\.?5|10)\s*/\s*10", "like_new"),
    (r"(?:стан\s*)?9\s*/\s*10", "very_good"),
    (r"(?:стан\s*)?[78]\s*/\s*10", "good"),
    (r"(?:стан\s*)?[56]\s*/\s*10", "fair"),
    (r"(?:стан\s*)?[1-4]\s*/\s*10", "needs_repair"),
    (r"ідеальн", "like_new"),
    (r"відмінн", "like_new"),
    (r"як\s*нов[аеий]", "like_new"),
    (r"майже\s*нов[аеий]", "like_new"),
    (r"чудов(?:ий|ому|а|е)\s*стан", "like_new"),
    (r"\bнов(?:ий|а|е|і|ої|ого|ому)\b", "new"),
    (r"\bстан\s*нов", "new"),
    (r"\bnew\b", "new"),
    (r"б\s*/?\s*[ув]", "good"),
    (r"\bused\b", "good"),
    (r"вживан", "good"),
    (r"refurb", "very_good"),
]

_CONDITION_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _CONDITION_PATTERNS]


def _detect_condition(text: str) -> Optional[str]:
    best_label: Optional[str] = None
    best_priority = -1
    for regex, label in _CONDITION_RE:
        if regex.search(text):
            prio = CONDITION_PRIORITY.get(label, 0)
            if prio > best_priority:
                best_priority = prio
                best_label = label
    return best_label


_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
)


_PHONE_MODEL_RE = re.compile(
    r"\b(?:Apple\s+)?"
    r"(iPhone\s*"
    r"(?:SE\s*(?:3|2|1)?(?:\s*(?:20\d{2}))?"
    r"|1[0-7](?:\s*(?:Pro\s*Max|Pro|Plus|Mini))?"
    r"|[5-9](?:\s*[SCsc])?"
    r"))"
    r"\b",
    re.IGNORECASE,
)

_SAMSUNG_MODEL_RE = re.compile(
    r"\b(Samsung\s*Galaxy\s*"
    r"(?:S\s*2[0-5](?:\s*(?:Ultra|FE|\+|Plus))?"
    r"|A\s*\d{2}(?:s)?"
    r"|Z\s*(?:Fold|Flip)\s*\d?"
    r"|Note\s*\d{1,2}"
    r"|M\s*\d{2}"
    r"))"
    r"\b",
    re.IGNORECASE,
)

_XIAOMI_MODEL_RE = re.compile(
    r"\b((?:Xiaomi|Redmi|POCO)\s*"
    r"(?:Note\s*)?\d{1,2}(?:\s*(?:Pro|Ultra|Lite|T|S|C|A))?"
    r")\b",
    re.IGNORECASE,
)

_GOOGLE_PIXEL_RE = re.compile(
    r"\b(Google\s*Pixel\s*\d[a-zA-Z]?(?:\s*(?:Pro|XL|a))?)\b",
    re.IGNORECASE,
)

_STORAGE_RE = re.compile(
    r"\b(\d{2,4})\s*(?:[ГгGg][БбBb]|GB)\b",
    re.IGNORECASE,
)

_BATTERY_RE = re.compile(
    r"(?:акумулятор|акб|батаре[яї]|battery|ємніст|health)[^\d]{0,50}(\d{2,3})\s*%"
    r"|(\d{2,3})\s*%[^\d]{0,50}(?:акумулятор|акб|батаре[яї]|battery|ємніст)",
    re.IGNORECASE,
)

_NEVERLOCK_RE = re.compile(
    r"\b(?:neverlock|неверлок|невер\s*лок|rsim|r-sim)\b",
    re.IGNORECASE,
)

_PHONE_COLOR_MAP: Dict[str, re.Pattern] = {
    "black": re.compile(r"\b(?:чорн\w*|black|midnight|space\s*(?:black|gray|grey))\b", re.IGNORECASE),
    "white": re.compile(r"\b(?:білий|біл\w*|white|starlight|silver|сріблястий)\b", re.IGNORECASE),
    "gold": re.compile(r"\b(?:gold|золот\w*)\b", re.IGNORECASE),
    "blue": re.compile(r"\b(?:синій|синьо\w*|блакитн\w*|blue|sierra\s*blue|deep\s*blue|pacific\s*blue)\b", re.IGNORECASE),
    "purple": re.compile(r"\b(?:фіолетов\w*|purple|deep\s*purple)\b", re.IGNORECASE),
    "green": re.compile(r"\b(?:зелен\w*|green|alpine\s*green)\b", re.IGNORECASE),
    "red": re.compile(r"\b(?:червон\w*|red|product\s*red)\b", re.IGNORECASE),
    "pink": re.compile(r"\b(?:рожев\w*|pink)\b", re.IGNORECASE),
    "graphite": re.compile(r"\b(?:graphite|графіт\w*)\b", re.IGNORECASE),
    "titanium": re.compile(r"\b(?:titanium|титанов\w*|natural\s*titanium|desert\s*titanium|blue\s*titanium|black\s*titanium)\b", re.IGNORECASE),
}

_HAS_BOX_RE = re.compile(
    r"\b(?:коробк[аиу]|повн(?:ий|а)\s*комплект|комплектаці[яї]|box|full\s*set|в\s*коробці)\b",
    re.IGNORECASE,
)


def _extract_cat4(text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    for regex in [_PHONE_MODEL_RE, _SAMSUNG_MODEL_RE, _XIAOMI_MODEL_RE, _GOOGLE_PIXEL_RE]:
        m = regex.search(text)
        if m:
            raw = m.group(1).strip()
            attrs["model"] = re.sub(r"\s+", " ", raw)
            break

    model = attrs.get("model", "")
    if "iphone" in model.lower() or re.search(r"\bapple\b", text, re.IGNORECASE):
        attrs["brand"] = "Apple"
    elif "samsung" in model.lower() or "galaxy" in model.lower():
        attrs["brand"] = "Samsung"
    elif any(k in model.lower() for k in ("xiaomi", "redmi", "poco")):
        attrs["brand"] = "Xiaomi"
    elif "pixel" in model.lower():
        attrs["brand"] = "Google"

    storage_matches = _STORAGE_RE.findall(text)
    valid_storages = [int(s) for s in storage_matches if int(s) in (16, 32, 64, 128, 256, 512, 1024)]
    if valid_storages:
        attrs["storage_gb"] = max(valid_storages)

    m = _BATTERY_RE.search(text)
    if m:
        val = int(m.group(1) or m.group(2))
        if 50 <= val <= 100:
            attrs["battery_pct"] = val

    cond = _detect_condition(text)
    if cond:
        attrs["condition"] = cond

    for color_name, color_re in _PHONE_COLOR_MAP.items():
        if color_re.search(text):
            attrs["color"] = color_name
            break

    attrs["neverlock"] = bool(_NEVERLOCK_RE.search(text))
    attrs["has_box"] = bool(_HAS_BOX_RE.search(text))

    return attrs


_SNEAKER_BRANDS = [
    ("Nike", re.compile(r"\bNike\b", re.IGNORECASE)),
    ("Jordan", re.compile(r"\bJordan\b", re.IGNORECASE)),
    ("Adidas", re.compile(r"\bAdidas\b", re.IGNORECASE)),
    ("New Balance", re.compile(r"\bNew\s*Balance\b", re.IGNORECASE)),
    ("Puma", re.compile(r"\bPuma\b", re.IGNORECASE)),
    ("Converse", re.compile(r"\bConverse\b", re.IGNORECASE)),
    ("Reebok", re.compile(r"\bReebok\b", re.IGNORECASE)),
    ("Salomon", re.compile(r"\bSalomon\b", re.IGNORECASE)),
    ("Asics", re.compile(r"\bAsics\b", re.IGNORECASE)),
    ("Vans", re.compile(r"\bVans\b", re.IGNORECASE)),
    ("Under Armour", re.compile(r"\bUnder\s*Armou?r\b", re.IGNORECASE)),
    ("Fila", re.compile(r"\bFila\b", re.IGNORECASE)),
    ("Skechers", re.compile(r"\bSkechers\b", re.IGNORECASE)),
]

_SNEAKER_MODEL_RE = re.compile(
    r"\b("
    r"Air\s*(?:Max|Force|Jordan)\s*[\w\d]*"
    r"|Yeezy\s*[\w\d]*"
    r"|Dunk\s*(?:Low|High|Mid)?\s*\w*"
    r"|NB\s*\d+"
    r"|(?:\d{3,4})\b"
    r"|Gel[\s-]\w+"
    r"|Ultra\s*Boost\w*"
    r"|Stan\s*Smith"
    r"|Superstar"
    r"|Forum\s*\w*"
    r"|Chuck\s*Taylor(?:\s*All\s*Star)?"
    r"|Old\s*Skool"
    r"|Giannis\s*\w*"
    r"|Fresh\s*Foam\w*"
    r"|XT[\s-]?\d+"
    r"|Tiempo\w*"
    r"|Cortez\w*"
    r"|Blazer\w*"
    r")",
    re.IGNORECASE,
)

_SIZE_EU_RE = re.compile(
    r"(?:розмір|size|р\.?)\s*(3[2-9]|4[0-9]|50)(?:\.5)?\b"
    r"|\b(3[5-9]|4[0-9]|50)(?:\.5)?\s*(?:розмір|size|EUR)\b",
    re.IGNORECASE,
)

_IS_ORIGINAL_RE = re.compile(
    r"\b(?:оригінал\w*|original|100\s*%\s*оригінал)\b",
    re.IGNORECASE,
)

_GENDER_RE = re.compile(
    r"\b(чоловіч\w*|жіноч\w*|унісекс|men(?:'s)?|women(?:'s)?|unisex)\b",
    re.IGNORECASE,
)


def _extract_cat512(text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    for brand_name, brand_re in _SNEAKER_BRANDS:
        if brand_re.search(text):
            attrs["brand"] = brand_name
            break

    m = _SNEAKER_MODEL_RE.search(text)
    if m:
        attrs["model_line"] = re.sub(r"\s+", " ", m.group(1).strip())

    m = _SIZE_EU_RE.search(text)
    if m:
        val = m.group(1) or m.group(2)
        attrs["size"] = float(val)

    cond = _detect_condition(text)
    if cond:
        attrs["condition"] = cond

    attrs["is_original"] = bool(_IS_ORIGINAL_RE.search(text))

    m = _GENDER_RE.search(text)
    if m:
        raw = m.group(1).lower()
        if any(k in raw for k in ("чоловіч", "men")):
            attrs["gender"] = "male"
        elif any(k in raw for k in ("жіноч", "women")):
            attrs["gender"] = "female"
        else:
            attrs["gender"] = "unisex"

    return attrs


_BOOK_TITLE_RE = re.compile(
    r'["\u201c\u201e\u00ab]([^"\u201d\u201f\u00bb]{2,80})["\u201d\u201f\u00bb]',
)

_AUTHOR_RE = re.compile(
    r"(?:^|\b)([А-ЯІЇЄҐA-Z][а-яіїєґa-z']+(?:\s+[А-ЯІЇЄҐA-Z]\.?)?"
    r"\s+[А-ЯІЇЄҐA-Z][а-яіїєґa-z']{2,})"
    r"(?:\s*[\"«\u201c\u201e]|\s+[-–-]|\s+серія|\s+книга)",
)

_COVER_TYPE_RE = re.compile(
    r"\b(тверд\w*(?:\s*(?:обкладинк|палітурк)\w*)?|м'як\w*(?:\s*(?:обкладинк|палітурк)\w*)?"
    r"|hardcover|paperback)\b",
    re.IGNORECASE,
)

_IS_SET_RE = re.compile(
    r"\b(?:серія|набір|комплект|том[иів]|[1-9]\s*(?:том|книг)|серію)\b",
    re.IGNORECASE,
)

_LANGUAGE_RE = re.compile(
    r"\b(українськ\w*|англійськ\w*|росі[йї]ськ\w*|english|ukrainian|russian|укр\b|англ\b)\b",
    re.IGNORECASE,
)


def _extract_cat795(text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    m = _BOOK_TITLE_RE.search(text)
    if m:
        attrs["book_title"] = m.group(1).strip()

    m = _AUTHOR_RE.search(text)
    if m:
        attrs["author"] = m.group(1).strip()

    cond = _detect_condition(text)
    if cond:
        attrs["condition"] = cond

    m = _COVER_TYPE_RE.search(text)
    if m:
        raw = m.group(1).lower()
        if any(k in raw for k in ("тверд", "hardcover")):
            attrs["cover_type"] = "hard"
        else:
            attrs["cover_type"] = "soft"

    attrs["is_set"] = bool(_IS_SET_RE.search(text))

    m = _LANGUAGE_RE.search(text)
    if m:
        raw = m.group(1).lower()
        if any(k in raw for k in ("українськ", "ukrainian", "укр")):
            attrs["language"] = "uk"
        elif any(k in raw for k in ("англійськ", "english", "англ")):
            attrs["language"] = "en"
        elif any(k in raw for k in ("росі", "russian")):
            attrs["language"] = "ru"

    return attrs


_FIGURE_SERIES_RE = re.compile(
    r"\b(Funko\s*Pop|Kinder\s*Joy|Hot\s*Toys|McFarlane|NECA|Bandai|"
    r"Figma|Figuarts|Good\s*Smile|Nendoroid|McDonald'?s|"
    r"Warhammer|Schleich|Hasbro)\b",
    re.IGNORECASE,
)

_FRANCHISE_RE = re.compile(
    r"\b(Stranger\s*Things|Дивні\s*[Дд]ива|Marvel|DC|Star\s*Wars|"
    r"Зоряні\s*[Вв]ійни|Harry\s*Potter|Гаррі\s*Поттер|"
    r"Dragon\s*Ball|Naruto|Наруто|One\s*Piece|Pokemon|Покемон|"
    r"Spider[\s-]?Man|Batman|Людина[\s-]?Павук|"
    r"Transformers|Трансформер\w*|"
    r"Friends|Simpsons|Minecraft|Disney|"
    r"Lord\s*of\s*(?:the\s*)?Rings|Володар\s*Перстнів)\b",
    re.IGNORECASE,
)

_CHARACTER_RE = re.compile(
    r"(?:фігурка|фигурка|figure|модель)\s+(\w[\w\s]{1,30}?)(?:\s+(?:з|із|from|колекційна|оригінальна|у|в)\b|$)",
    re.IGNORECASE,
)

_COMPLETENESS_RE = re.compile(
    r"\b(повн(?:ий|а)\s*(?:комплект|набір)|без\s*коробки|в\s*коробці|"
    r"з\s*(?:коробкою|вкладишем|підставкою)|розпакован|sealed|"
    r"не\s*(?:відкрива|розпаков))\b",
    re.IGNORECASE,
)

_IS_RARE_RE = re.compile(
    r"\b(?:рідкісн\w*|rare|лімітован\w*|limited|ексклюзив\w*|exclusive|"
    r"колекційн\w*\s*рідкіс|chase)\b",
    re.IGNORECASE,
)


def _extract_cat1677(text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    m = _FIGURE_SERIES_RE.search(text)
    if m:
        attrs["series"] = re.sub(r"\s+", " ", m.group(1).strip())

    m = _FRANCHISE_RE.search(text)
    if m:
        attrs["franchise"] = re.sub(r"\s+", " ", m.group(1).strip())

    m = _CHARACTER_RE.search(text)
    if m:
        attrs["character"] = m.group(1).strip()

    cond = _detect_condition(text)
    if cond:
        attrs["condition"] = cond

    m = _COMPLETENESS_RE.search(text)
    if m:
        raw = m.group(1).lower()
        if any(k in raw for k in ("повн", "sealed", "не відкрива", "не розпаков", "в коробці", "з коробкою")):
            attrs["completeness"] = "complete"
        elif "без коробки" in raw:
            attrs["completeness"] = "no_box"
        else:
            attrs["completeness"] = "opened"

    attrs["is_rare"] = bool(_IS_RARE_RE.search(text))

    return attrs


_CONSTRUCTOR_BRAND_RE = re.compile(
    r"\b(LEGO|Cobi|Mega\s*Bloks|Playmobil|BanBao|Конструктор)\b",
    re.IGNORECASE,
)

_SET_NUMBER_RE = re.compile(
    r"\b(?:набір|set|артикул|арт\.?|#)?\s*(\d{4,6})\b",
)

_PIECE_COUNT_RE = re.compile(
    r"\b(\d{2,5})\s*(?:деталей|деталі|шт|pcs|pieces|елементів|блоків|blocks)\b",
    re.IGNORECASE,
)

_LEGO_THEME_RE = re.compile(
    r"\b(Technic|City|Star\s*Wars|Creator|Friends|Ninjago|"
    r"Marvel|Harry\s*Potter|Speed\s*Champions|Icons|Architecture|"
    r"Minecraft|DUPLO|Classic|Ideas|Botanical|Disney)\b",
    re.IGNORECASE,
)

_CONSTRUCTOR_COMPLETENESS_RE = re.compile(
    r"\b(повн(?:ий|а)\s*(?:комплект|набір)|неповн\w*|без\s*(?:коробки|інструкції)|"
    r"з\s*інструкцією|всі\s*деталі|не\s*вистачає|запаков|розпаков|sealed|"
    r"пломб\w*|нові\s*пакет|не\s*відкрива\w*|не\s*розпаков\w*)\b",
    re.IGNORECASE,
)


def _extract_cat743(text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    m = _CONSTRUCTOR_BRAND_RE.search(text)
    if m:
        raw = m.group(1)
        attrs["brand"] = "LEGO" if raw.upper() == "LEGO" else raw.strip()

    m = _SET_NUMBER_RE.search(text)
    if m:
        attrs["set_number"] = m.group(1)

    m = _PIECE_COUNT_RE.search(text)
    if m:
        attrs["piece_count"] = int(m.group(1))

    m = _LEGO_THEME_RE.search(text)
    if m:
        attrs["theme"] = m.group(1).strip()

    cond = _detect_condition(text)
    if cond:
        attrs["condition"] = cond

    m = _CONSTRUCTOR_COMPLETENESS_RE.search(text)
    if m:
        raw = m.group(1).lower()
        if any(k in raw for k in ("повн", "всі деталі", "sealed", "запаков", "пломб", "нові пакет", "не відкрива", "не розпаков")):
            attrs["completeness"] = "complete"
        elif any(k in raw for k in ("неповн", "не вистачає")):
            attrs["completeness"] = "incomplete"
        elif "без" in raw:
            attrs["completeness"] = "no_box"
        else:
            attrs["completeness"] = "opened"

    attrs["has_box"] = bool(re.search(r"\b(?:коробк[аиуоює]|коробці|box|в\s*коробці)\b", text, re.IGNORECASE))

    return attrs


_CHAIR_MATERIAL_RE = re.compile(
    r"\b(екошкір\w*|натуральн\w*\s*шкір\w*|шкір\w*|тканин\w*|велюр\w*|"
    r"сітк\w*|mesh|пластик\w*|метал\w*|дерев'?ян\w*|дерево)\b",
    re.IGNORECASE,
)

_CHAIR_BRAND_RE = re.compile(
    r"\b(Herman\s*Miller|IKEA|Nowy\s*Styl|BONRO|DEXTER|Hexter|GT\s*Racer|"
    r"Barsky|DXRacer|Hator|AeroCool|Xiaomi)\b",
    re.IGNORECASE,
)

_CHAIR_TYPE_RE = re.compile(
    r"\b(офісн\w*|барн\w*|дитяч\w*|кухонн\w*|складн\w*|геймерськ\w*|ігров\w*|"
    r"комп'ютерн\w*|обідн\w*|ортопедичн\w*|ергономічн\w*)\b",
    re.IGNORECASE,
)


def _extract_cat1320(text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    _UPHOLSTERY_PRIORITY = [
        (re.compile(r"\bекошкір\w*\b", re.IGNORECASE), "eco_leather"),
        (re.compile(r"\bнатуральн\w*\s*шкір\w*\b", re.IGNORECASE), "leather"),
        (re.compile(r"\b(?:оббивк\w*|сидінн\w*|спинк\w*)[^\n]{0,30}(?:тканин\w*|велюр\w*)\b", re.IGNORECASE), "fabric"),
        (re.compile(r"\b(?:тканин\w*|велюр\w*)\s*(?:оббивк\w*)?\b", re.IGNORECASE), "fabric"),
        (re.compile(r"\bсітк\w*|mesh\b", re.IGNORECASE), "mesh"),
        (re.compile(r"\bшкір\w*\b", re.IGNORECASE), "leather"),
        (re.compile(r"\bпластик\w*\b", re.IGNORECASE), "plastic"),
        (re.compile(r"\bметал\w*\b", re.IGNORECASE), "metal"),
        (re.compile(r"\bдерев'?ян\w*|дерево\b", re.IGNORECASE), "wood"),
    ]
    for mat_re, mat_label in _UPHOLSTERY_PRIORITY:
        if mat_re.search(text):
            attrs["material"] = mat_label
            break

    m = _CHAIR_BRAND_RE.search(text)
    if m:
        attrs["brand"] = re.sub(r"\s+", " ", m.group(1).strip())

    m = _CHAIR_TYPE_RE.search(text)
    if m:
        raw = m.group(1).lower()
        if any(k in raw for k in ("геймерськ", "ігров")):
            attrs["type"] = "gaming"
        elif any(k in raw for k in ("офісн", "комп'ютерн")):
            attrs["type"] = "office"
        elif "ортопедичн" in raw or "ергономічн" in raw:
            attrs["type"] = "ergonomic"
        elif "барн" in raw:
            attrs["type"] = "bar"
        elif "дитяч" in raw:
            attrs["type"] = "kids"
        elif "кухонн" in raw or "обідн" in raw:
            attrs["type"] = "dining"
        elif "складн" in raw:
            attrs["type"] = "folding"

    cond = _detect_condition(text)
    if cond:
        attrs["condition"] = cond

    return attrs


_WHEEL_SIZE_R_RE = re.compile(r"\b[Rr]\s*(\d{2})\b")

_WHEEL_WIDTH_PROFILE_RE = re.compile(
    r"\b(\d{3})\s*/\s*(\d{2,3})\s*[Rr]\s*(\d{2})\b",
)

_WHEEL_SEASON_RE = re.compile(
    r"\b(літн[іи]\w*|зимов[іи]\w*|всесезонн\w*|summer|winter|all[\s-]?season|M\+S)\b",
    re.IGNORECASE,
)

_TIRE_BRAND_RE = re.compile(
    r"\b(Michelin|Continental|Bridgestone|Goodyear|Pirelli|Hankook|"
    r"Dunlop|Nokian|Yokohama|Toyo|BFGoodrich|Firestone|Kumho|"
    r"Maxxis|Falken|Cooper|General|Nexen|Sailun|Triangle)\b",
    re.IGNORECASE,
)

_DISK_BRAND_RE = re.compile(
    r"\b(Volkswagen|Toyota|BMW|Mercedes|Audi|Skoda|Honda|Hyundai|Kia|"
    r"Ford|Renault|Opel|Mazda|Nissan|Peugeot|OZ\s*Racing|BBS|Borbet|MAK)\b",
    re.IGNORECASE,
)

_TREAD_RE = re.compile(
    r"(?:протектор|tread)[^\d]{0,20}(\d{1,2}(?:[.,]\d)?)\s*(?:мм|mm)",
    re.IGNORECASE,
)

_TREAD_PCT_RE = re.compile(
    r"(?:протектор|залишок|tread)[^\d]{0,20}(\d{2,3})\s*%",
    re.IGNORECASE,
)


def _extract_cat1261(text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    m = _WHEEL_SIZE_R_RE.search(text)
    if m:
        attrs["size_r"] = int(m.group(1))

    m = _WHEEL_WIDTH_PROFILE_RE.search(text)
    if m:
        attrs["width"] = int(m.group(1))
        attrs["profile"] = int(m.group(2))
        attrs["width_profile"] = f"{m.group(1)}/{m.group(2)}"
        attrs["size_r"] = int(m.group(3))

    m = _WHEEL_SEASON_RE.search(text)
    if m:
        raw = m.group(1).lower()
        if any(k in raw for k in ("літн", "summer")):
            attrs["season"] = "summer"
        elif any(k in raw for k in ("зимов", "winter")):
            attrs["season"] = "winter"
        else:
            attrs["season"] = "all_season"

    m = _TIRE_BRAND_RE.search(text)
    if m:
        attrs["brand"] = m.group(1)
    else:
        m = _DISK_BRAND_RE.search(text)
        if m:
            attrs["brand"] = m.group(1)

    m = _TREAD_PCT_RE.search(text)
    if m:
        attrs["tread_pct"] = int(m.group(1))
    else:
        m = _TREAD_RE.search(text)
        if m:
            val = float(m.group(1).replace(",", "."))
            if val <= 12:
                attrs["tread_pct"] = int(val / 8 * 100)

    cond = _detect_condition(text)
    if cond:
        attrs["condition"] = cond

    return attrs


def _extract_meta(
    title: str,
    description: str,
    photo_count: int = 0,
    created_at: Optional[str | datetime] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    meta["title_length"] = len(title)
    meta["desc_length"] = len(description)
    meta["photo_count"] = photo_count
    meta["has_emoji"] = bool(_EMOJI_RE.search(title + description))

    if created_at is not None:
        if isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                dt = None
        else:
            dt = created_at
        if dt is not None:
            meta["month"] = dt.month
            meta["day_of_week"] = dt.weekday()

    return meta


_CATEGORY_EXTRACTORS: Dict[int, Any] = {
    4: _extract_cat4,
    512: _extract_cat512,
    795: _extract_cat795,
    1677: _extract_cat1677,
    743: _extract_cat743,
    1320: _extract_cat1320,
    1261: _extract_cat1261,
}


class CategoryFeatureExtractor:
    def __init__(self, category_id: int) -> None:
        if category_id not in _CATEGORY_EXTRACTORS:
            raise ValueError(
                f"Unknown category_id {category_id}. "
                f"Available: {list(_CATEGORY_EXTRACTORS.keys())}"
            )
        self.category_id = category_id
        self._extract_fn = _CATEGORY_EXTRACTORS[category_id]
        self._required = REQUIRED_ATTRS.get(category_id, [])

    def extract(
        self,
        title: str,
        description: str,
        photo_count: int = 0,
        created_at: Optional[str | datetime] = None,
    ) -> Dict[str, Any]:
        text = f"{title}\n{description}"
        attrs = self._extract_fn(text)
        meta = _extract_meta(title, description, photo_count, created_at)
        attrs.update(meta)
        return attrs

    def get_missing(self, found_attrs: Dict[str, Any]) -> List[str]:
        return [a for a in self._required if a not in found_attrs or found_attrs[a] is None]
