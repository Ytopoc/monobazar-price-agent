# Per-category configuration: attributes, templates, regex patterns, keywords.
import re
from typing import Any, Callable, Dict, List, Pattern

CATEGORY_DICT: Dict[int, str] = {
    4:    "Електроніка/Телефони/Смартфони/Apple",
    512:  "Стиль і краса/Взуття/Кросівки",
    743:  "Дитячий світ/Конструктори",
    795:  "Хобі, спорт і відпочинок/Книги та журнали",
    1677: "Колекції та антикваріат/Колекційні фігурки",
    1261: "Запчастини для транспорту/Шини, диски і колеса/Колеса в зборі",
    1320: "Дім і сад/Меблі/Стільці",
}

CATEGORY_ALIAS: Dict[int, str] = {
    4:    "smartphones_apple",
    512:  "sneakers",
    743:  "constructors",
    795:  "books",
    1677: "collectible_figures",
    1261: "wheels_tires",
    1320: "chairs",
}

REQUIRED_ATTRS: Dict[int, List[str]] = {
    4: [
        "brand",
        "model",
        "storage_gb",
        "color",
        "condition",
        "battery_pct",
    ],
    512: [
        "brand",
        "model",
        "size_eu",
        "condition",
        "gender",
        "color",
    ],
    743: [
        "brand",
        "set_number",
        "piece_count",
        "theme",
        "condition",
        "completeness",
    ],
    795: [
        "author",
        "title",
        "language",
        "condition",
        "cover_type",
    ],
    1677: [
        "brand",
        "character",
        "series",
        "condition",
        "size_cm",
    ],
    1261: [
        "tire_brand",
        "size",
        "season",
        "condition",
        "tread_mm",
        "quantity",
    ],
    1320: [
        "type",
        "material",
        "condition",
        "color",
        "brand",
    ],
}

OLX_QUERY_TEMPLATES: Dict[int, Callable[[Dict[str, Any]], str]] = {
    4: lambda a: (
        f"site:olx.ua {a.get('brand', 'Apple')} {a.get('model', 'iPhone')} "
        f"{a.get('storage_gb', '')}GB {a.get('condition', '')} купити"
    ).strip(),

    512: lambda a: (
        f"site:olx.ua {a.get('brand', '')} {a.get('model', '')} кросівки "
        f"{a.get('size_eu', '')} {a.get('gender', '')}"
    ).strip(),

    743: lambda a: (
        f"site:olx.ua {a.get('brand', 'LEGO')} {a.get('set_number', '')} "
        f"{a.get('theme', '')} конструктор"
    ).strip(),

    795: lambda a: (
        f"site:olx.ua {a.get('author', '')} {a.get('title', '')} книга "
        f"{a.get('language', '')} купити"
    ).strip(),

    1677: lambda a: (
        f"site:olx.ua {a.get('brand', '')} {a.get('character', '')} "
        f"фігурка {a.get('series', '')} колекційна"
    ).strip(),

    1261: lambda a: (
        f"site:olx.ua шини {a.get('tire_brand', '')} {a.get('size', '')} "
        f"{a.get('season', '')} {a.get('condition', '')}"
    ).strip(),

    1320: lambda a: (
        f"site:olx.ua стілець {a.get('type', '')} {a.get('material', '')} "
        f"{a.get('brand', '')} {a.get('condition', '')}"
    ).strip(),
}

REGEX_PATTERNS: Dict[int, Dict[str, "re.Pattern[str]"]] = {
    4: {
        "model": re.compile(
            r"\b(iPhone\s*(?:SE|1[0-6]|[5-9])(?:\s*(?:Pro\s*Max|Pro|Plus|Mini))?)\b",
            re.IGNORECASE,
        ),
        "storage_gb": re.compile(r"\b(\d{2,4})\s*(?:gb|гб)\b", re.IGNORECASE),
        "battery_pct": re.compile(r"\b(?:акб|батаре[яї]|battery|акум)[^\d]{0,10}(\d{2,3})\s*%", re.IGNORECASE),
        "condition": re.compile(
            r"\b(нов(?:ий|а|е)|б/?у|used|ідеальн|відмінн|refurb|як нов|вживан)",
            re.IGNORECASE,
        ),
        "color": re.compile(
            r"\b(чорн\w*|білий|сірий|silver|gold|space\s*gr[ae]y|midnight|starlight|"
            r"блакитн\w*|фіолетов\w*|синій|зелен\w*|червон\w*|рожев\w*|titanium)\b",
            re.IGNORECASE,
        ),
    },
    512: {
        "brand": re.compile(
            r"\b(Nike|Adidas|New\s*Balance|Puma|Reebok|Asics|Converse|Vans|"
            r"Jordan|Salomon|Under\s*Armour|Skechers|Fila)\b",
            re.IGNORECASE,
        ),
        "model": re.compile(
            r"\b(Air\s*(?:Max|Force|Jordan)\s*\w*|Yeezy\s*\w*|"
            r"NB\s*\d+|Gel-\w+|Ultra\s*Boost\w*|Dunk\s*\w*|"
            r"Stan\s*Smith|Superstar|Forum\s*\w*|Tiempo\s*\w*)\b",
            re.IGNORECASE,
        ),
        "size_eu": re.compile(r"\b(?:розмір|size|р\.?)?\s*(3[5-9]|4[0-9]|50)(?:\.5)?\b", re.IGNORECASE),
        "condition": re.compile(
            r"\b(нов[іиіеа]|б/?у|used|ідеальн|відмінн|як нов|вживан|ds|vnds)\b",
            re.IGNORECASE,
        ),
        "gender": re.compile(r"\b(чоловіч\w*|жіноч\w*|унісекс|men|women|unisex)\b", re.IGNORECASE),
    },
    743: {
        "brand": re.compile(r"\b(LEGO|Cobi|Mega\s*Bloks|Playmobil|BanBao)\b", re.IGNORECASE),
        "set_number": re.compile(r"\b(\d{4,6})\b"),
        "piece_count": re.compile(r"\b(\d{2,5})\s*(?:деталей|шт|pcs|pieces|елементів)\b", re.IGNORECASE),
        "theme": re.compile(
            r"\b(Technic|City|Star\s*Wars|Creator|Friends|Ninjago|"
            r"Marvel|Harry\s*Potter|Speed\s*Champions|Icons|Architecture)\b",
            re.IGNORECASE,
        ),
        "condition": re.compile(
            r"\b(нов(?:ий|а|е)|б/?у|запаков|розпаков|без коробки|sealed|відкрит)\b",
            re.IGNORECASE,
        ),
    },
    795: {
        "language": re.compile(
            r"\b(українськ\w*|англійськ\w*|russian|english|ukrainian|росі[йї]ськ\w*|укр|англ)\b",
            re.IGNORECASE,
        ),
        "cover_type": re.compile(
            r"\b(тверд\w*(?:\s*обкладинк\w*)?|м'як\w*(?:\s*обкладинк\w*)?|hardcover|paperback)\b",
            re.IGNORECASE,
        ),
        "condition": re.compile(
            r"\b(нов(?:ий|а|е)|б/?у|ідеальн|як нов|відмінн|вживан)\b",
            re.IGNORECASE,
        ),
    },
    1677: {
        "brand": re.compile(
            r"\b(Funko\s*Pop|Hot\s*Toys|McFarlane|NECA|Bandai|Hasbro|"
            r"Figma|S\.H\.\s*Figuarts|Good\s*Smile)\b",
            re.IGNORECASE,
        ),
        "series": re.compile(
            r"\b(Marvel|DC|Star\s*Wars|Anime|Dragon\s*Ball|Naruto|"
            r"One\s*Piece|Pokemon|Гаррі\s*Поттер|Harry\s*Potter)\b",
            re.IGNORECASE,
        ),
        "condition": re.compile(
            r"\b(нов(?:ий|а|е)|б/?у|в коробці|без коробки|розпаков|mint|opened)\b",
            re.IGNORECASE,
        ),
        "size_cm": re.compile(r"\b(\d{1,3})\s*(?:см|cm)\b", re.IGNORECASE),
    },
    1261: {
        "tire_brand": re.compile(
            r"\b(Michelin|Continental|Bridgestone|Goodyear|Pirelli|Hankook|"
            r"Dunlop|Nokian|Yokohama|Toyo|BFGoodrich|Firestone|Kumho)\b",
            re.IGNORECASE,
        ),
        "size": re.compile(
            r"\b(\d{3})\s*/\s*(\d{2,3})\s*[Rr]\s*(\d{2})\b",
        ),
        "season": re.compile(
            r"\b(літн[іи]\w*|зимов[іи]\w*|всесезонн\w*|summer|winter|all[- ]?season|M\+S)\b",
            re.IGNORECASE,
        ),
        "tread_mm": re.compile(r"\b(\d(?:[.,]\d)?)\s*(?:мм|mm)\b", re.IGNORECASE),
        "condition": re.compile(
            r"\b(нов[іиеа]|б/?у|used|вживан)\b",
            re.IGNORECASE,
        ),
        "quantity": re.compile(r"\b(\d)\s*(?:шт|штуки?|к-т|комплект)\b", re.IGNORECASE),
    },
    1320: {
        "type": re.compile(
            r"\b(офісн\w*|барн\w*|дитяч\w*|кухонн\w*|складн\w*|геймерськ\w*|"
            r"комп'ютерн\w*|обідн\w*)\b",
            re.IGNORECASE,
        ),
        "material": re.compile(
            r"\b(дерев'?ян\w*|дерево|метал\w*|пластик\w*|тканин\w*|шкір\w*|"
            r"велюр\w*|сітк\w*|екошкір\w*)\b",
            re.IGNORECASE,
        ),
        "condition": re.compile(
            r"\b(нов(?:ий|а|е)|б/?у|ідеальн|як нов|вживан)\b",
            re.IGNORECASE,
        ),
        "color": re.compile(
            r"\b(чорн\w*|білий|сірий|коричнев\w*|бежев\w*|червон\w*|синій|зелен\w*)\b",
            re.IGNORECASE,
        ),
    },
}

CATEGORY_KEYWORDS: Dict[int, List[str]] = {
    4:    [
        "iphone", "apple", "айфон", "смартфон", "телефон", "phone",
        "samsung", "самсунг", "galaxy", "xiaomi", "pixel", "redmi", "poco",
        "huawei", "хуавей", "oneplus", "realme", "honor", "хонор",
        "fold", "flip", "ultra",
    ],
    512:  [
        "кросівки", "кроссовки", "sneakers", "nike", "adidas", "взуття",
        "salomon", "саломон", "new balance", "нью баланс", "puma", "пума",
        "reebok", "рібок", "asics", "асікс", "jordan", "джордан",
        "converse", "конверс", "vans", "ванс", "skechers", "скечерс",
        "fila", "філа", "under armour", "hoka", "хока",
        "кеди", "кросовки", "кроси", "обувь", "туфли",
        "xt-6", "xt-4", "air max", "air force", "yeezy",
    ],
    743:  ["конструктор", "lego", "лего", "набір деталей", "cobi"],
    795:  ["книга", "книжка", "роман", "автор", "видавництво", "book"],
    1677: ["фігурка", "колекційна", "funko", "figurine", "статуетка", "hot toys"],
    1261: [
        "шини", "диски", "колеса", "покришки", "шины", "колёса", "резина",
        "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21", "r22",
        "michelin", "continental", "bridgestone", "goodyear", "pirelli",
        "hankook", "nokian", "yokohama",
    ],
    1320: [
        "стілець", "крісло офісне", "табурет", "chair", "крісло",
        "стул", "кресло", "dxracer", "ikea", "ікеа", "геймерське",
        "офісне крісло", "компьютерное кресло", "игровое кресло",
    ],
}

CATEGORY_STATS_DEFAULTS: Dict[int, Dict[str, Any]] = {
    cat_id: {
        "price_median": None,
        "price_mean": None,
        "price_std": None,
        "price_p10": None,
        "price_p90": None,
        "avg_days_to_sell": None,
        "sold_ratio": None,
        "n_listings": None,
    }
    for cat_id in CATEGORY_DICT
}
