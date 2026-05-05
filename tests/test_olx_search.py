# OLX tests: extract_price, build_olx_query, search_and_parse.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.olx_search import extract_price, build_olx_query, OLXSearcher


class TestExtractPrice:
    def test_price_with_hryvnia_symbol(self):
        assert extract_price("Продам телефон за 16 500 ₴") == 16500.0

    def test_price_with_grn_suffix(self):
        assert extract_price("Ціна 8500грн торг") == 8500.0

    def test_price_with_grn_dot_suffix(self):
        assert extract_price("Кросівки Nike 3 200 грн.") == 3200.0

    def test_price_with_comma_separator(self):
        assert extract_price("Продаю за 16,500 ₴") == 16500.0

    def test_price_with_dot_thousands(self):
        assert extract_price("LEGO Star Wars 4.500 грн") == 4500.0

    def test_price_keyword_vid(self):
        assert extract_price("від 15000 можливий торг") == 15000.0

    def test_price_keyword_tsina(self):
        assert extract_price("ціна 5500 за комплект") == 5500.0

    def test_price_keyword_price(self):
        assert extract_price("price: 12000 negotiable") == 12000.0

    def test_no_price_returns_none(self):
        assert extract_price("Продам iPhone 13 Pro Max") is None

    def test_ignores_model_number_iphone(self):
        assert extract_price("iPhone 13 Pro Max 256GB") is None

    def test_ignores_storage_gb(self):
        assert extract_price("Телефон 256GB пам'яті") is None

    def test_ignores_too_small_price(self):
        assert extract_price("всього 5 ₴") is None

    def test_ignores_too_large_price(self):
        assert extract_price("2000000 грн") is None

    def test_price_with_nbsp(self):
        assert extract_price("Стілець 2\u00a0500 ₴") == 2500.0

    def test_price_prefix_symbol(self):
        assert extract_price("₴ 7500 чудовий стан") == 7500.0

    def test_combined_title_body(self):
        text = "LEGO Technic 42130. Стан ідеальний. 12 500 грн"
        assert extract_price(text) == 12500.0

    def test_price_keyword_vartist(self):
        assert extract_price("вартість 9800 грн") == 9800.0


class TestBuildOlxQuery:
    def test_cat4_smartphones(self):
        attrs = {"model": "iPhone 13 Pro", "storage_gb": 256}
        q = build_olx_query(attrs, category_id=4)
        assert "iPhone 13 Pro" in q
        assert "256GB" in q

    def test_cat4_smartphones_no_storage(self):
        attrs = {"model": "Samsung Galaxy S22"}
        q = build_olx_query(attrs, category_id=4)
        assert "Samsung Galaxy S22" in q
        assert "GB" not in q

    def test_cat512_sneakers(self):
        attrs = {"brand": "Nike", "model_line": "Air Max 90", "size": 42}
        q = build_olx_query(attrs, category_id=512)
        assert "Nike" in q
        assert "Air Max 90" in q
        assert "кросівки" in q
        assert "42" in q

    def test_cat512_sneakers_no_size(self):
        attrs = {"brand": "Adidas", "model_line": "Ultraboost"}
        q = build_olx_query(attrs, category_id=512)
        assert "Adidas" in q
        assert "кросівки" in q

    def test_cat795_books(self):
        attrs = {"author": "Стівен Кінг", "book_title": "Воно"}
        q = build_olx_query(attrs, category_id=795)
        assert "Стівен Кінг" in q
        assert "Воно" in q

    def test_cat795_books_fallback(self):
        attrs = {}
        q = build_olx_query(attrs, category_id=795)
        assert "книга" in q

    def test_cat1677_figures_with_character(self):
        attrs = {"series": "Marvel", "franchise": "Avengers", "character": "Iron Man"}
        q = build_olx_query(attrs, category_id=1677)
        assert "Marvel" in q
        assert "Iron Man" in q
        assert "фігурка" in q

    def test_cat1677_figures_no_attrs(self):
        attrs = {}
        q = build_olx_query(attrs, category_id=1677)
        assert "колекційна фігурка" in q

    def test_cat743_constructors_with_set_number(self):
        attrs = {"brand": "LEGO", "set_number": "42130"}
        q = build_olx_query(attrs, category_id=743)
        assert "LEGO" in q
        assert "42130" in q
        assert "конструктор" not in q

    def test_cat743_constructors_no_brand(self):
        attrs = {"theme": "Star Wars"}
        q = build_olx_query(attrs, category_id=743)
        assert "Star Wars" in q
        assert "конструктор" in q

    def test_cat1320_chairs(self):
        attrs = {"brand": "IKEA", "material": "дерево", "type": "барний"}
        q = build_olx_query(attrs, category_id=1320)
        assert "стілець" in q
        assert "IKEA" in q
        assert "барний" in q
        assert "дерево" in q

    def test_cat1261_wheels(self):
        attrs = {"size_r": 16, "brand": "Michelin", "season": "зимові"}
        q = build_olx_query(attrs, category_id=1261)
        assert "колеса" in q
        assert "R16" in q
        assert "Michelin" in q
        assert "зимові" in q

    def test_fallback_text_used_when_query_short(self):
        attrs = {}
        q = build_olx_query(attrs, category_id=9999, fallback_text="Продам червоний велосипед")
        assert len(q) > 3
        assert "Продам" in q or "червоний" in q or "велосипед" in q

    def test_empty_attrs_unknown_category(self):
        attrs = {}
        q = build_olx_query(attrs, category_id=9999, fallback_text="")
        assert q == ""


class TestSearchAndParse:
    @pytest.mark.asyncio
    async def test_search_iphone(self):
        searcher = OLXSearcher(timeout=15.0, max_results=10)
        result = await searcher.search_and_parse(
            structured_attrs={"model": "iPhone 13 Pro", "storage_gb": 256},
            category_id=4,
        )
        assert "status" in result
        assert "olx_count" in result
        assert "olx_median" in result
        assert "query_used" in result

        if result["status"] == "ok":
            assert result["olx_count"] > 0
            assert result["olx_median"] > 0
            assert result["olx_min"] <= result["olx_median"] <= result["olx_max"]
            assert result["olx_estimated_market"] > 0
            assert len(result["olx_items"]) <= 5
            for item in result["olx_items"]:
                assert "title" in item
                assert "price" in item
                assert "url" in item
                assert "olx.ua" in item["url"]
            print(f"\n  Query: {result['query_used']}")
            print(f"  Found: {result['olx_count']} listings")
            print(f"  Median: {result['olx_median']:.0f} UAH")
            print(f"  Range: {result['olx_min']:.0f} - {result['olx_max']:.0f} UAH")
            print(f"  Market estimate: {result['olx_estimated_market']:.0f} UAH")
        else:
            print(f"\n  Search status: {result['status']} (may be network issue)")
            assert result["status"] in ("no_results", "timeout", "error")

    @pytest.mark.asyncio
    async def test_search_no_query(self):
        searcher = OLXSearcher(timeout=5.0)
        result = await searcher.search_and_parse(
            structured_attrs={},
            category_id=9999,
            enriched_description="",
        )
        assert result["status"] == "no_query"
        assert result["olx_count"] == 0
