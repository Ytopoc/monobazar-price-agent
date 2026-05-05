# Tests for the per-category regex attribute extractor.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.extractor import CategoryFeatureExtractor


def _run(cat_id: int, title: str, desc: str, expected: dict, label: str = ""):
    ext = CategoryFeatureExtractor(cat_id)
    result = ext.extract(title, desc, photo_count=3)
    missing = ext.get_missing(result)
    ok = True
    for k, v in expected.items():
        actual = result.get(k)
        if actual != v:
            print(f"  FAIL {label} [{k}]: expected={v!r}, got={actual!r}")
            ok = False
    if ok:
        print(f"  PASS {label}")
    else:
        print(f"  DETAILS: {result}")
    return ok


def test_cat4_smartphones():
    print("\n=== Category 4: Smartphones ===")
    total = 0
    passed = 0

    cases = [
        (
            "iPhone 14 Pro 256 ГБ Deep Purple з новим акумулятором",
            "Продається iPhone 14 Pro на 256 ГБ у кольорі Deep Purple. Телефон повністю справний.",
            {"model": "iPhone 14 Pro", "brand": "Apple", "storage_gb": 256, "color": "purple"},
            "iPhone 14 Pro 256GB",
        ),
        (
            "iPhone 11 Pro Max 256GB Gold Neverlock від першого власника",
            "Оригінальний iPhone 11 Pro Max у золотому кольорі від єдиного власника. Neverlock.",
            {"model": "iPhone 11 Pro Max", "storage_gb": 256, "color": "gold", "neverlock": True},
            "iPhone 11 Pro Max Neverlock",
        ),
        (
            "iPhone 14 Pro 512GB фіолетовий Neverlock стан 9/10",
            "Модель: iPhone 14 Pro. Пам'ять: 512GB. Стан акумулятора: 81%",
            {"model": "iPhone 14 Pro", "storage_gb": 512, "battery_pct": 81, "neverlock": True, "condition": "very_good"},
            "iPhone 14 Pro 512GB batt 81%",
        ),
        (
            "iPhone 15 Pro Max 256GB синій - вживаний, акб 83%",
            "Apple iPhone 15 Pro Max у синьому кольорі з пам'яттю 256 ГБ. Акумулятор: 83% (рідний)",
            {"model": "iPhone 15 Pro Max", "storage_gb": 256, "color": "blue", "battery_pct": 83},
            "iPhone 15 Pro Max blue 83%",
        ),
        (
            "iPhone 13 mini 128 ГБ білий неверлок, акумулятор 88%, повна комплектація",
            "Apple iPhone 13 mini 128 ГБ в білому кольорі. Неверлок. Акумулятор: 88% ємності. Оригінальна коробка.",
            {"model": "iPhone 13 mini", "storage_gb": 128, "color": "white", "neverlock": True, "battery_pct": 88, "has_box": True},
            "iPhone 13 mini white neverlock box",
        ),
    ]

    for title, desc, expected, label in cases:
        total += 1
        if _run(4, title, desc, expected, label):
            passed += 1

    print(f"  Result: {passed}/{total}")
    return passed, total


def test_cat512_sneakers():
    print("\n=== Category 512: Sneakers ===")
    total = 0
    passed = 0

    cases = [
        (
            "Сороконожки Nike Tiempo оригінал, стан 9/10",
            "Оригінальні сороконожки Nike Tiempo в чорному кольорі. Стан: 9/10",
            {"brand": "Nike", "model_line": "Tiempo", "is_original": True, "condition": "very_good"},
            "Nike Tiempo original",
        ),
        (
            "Кросівки жіночі шкіра+замша з зіркою, платформа, розмір 38",
            "Стильні жіночі кросівки. Розмір: 38",
            {"size": 38.0, "gender": "female"},
            "Women shoes size 38",
        ),
        (
            "Salomon XT-6 Gore-Tex білі кросівки 42 розмір б/в",
            "Трекінгові кросівки Salomon XT-6 Gore-Tex у білому кольорі, 42 розмір. Стан: б/в.",
            {"brand": "Salomon", "size": 42.0, "condition": "good"},
            "Salomon XT-6 42 used",
        ),
        (
            "Кросівки New Balance Fresh Foam розмір 40 (US 8.5)",
            "Спортивні кросівки New Balance з технологією Fresh Foam. Розмір: 40 EUR",
            {"brand": "New Balance", "model_line": "Fresh Foam", "size": 40.0},
            "NB Fresh Foam 40",
        ),
        (
            "Converse Chuck Taylor All Star зимові кеди сині 44 розмір",
            "Легендарні кеди Converse Chuck Taylor All Star",
            {"brand": "Converse", "model_line": "Chuck Taylor All Star", "size": 44.0},
            "Converse Chuck Taylor 44",
        ),
    ]

    for title, desc, expected, label in cases:
        total += 1
        if _run(512, title, desc, expected, label):
            passed += 1

    print(f"  Result: {passed}/{total}")
    return passed, total


def test_cat795_books():
    print("\n=== Category 795: Books ===")
    total = 0
    passed = 0

    cases = [
        (
            'Томас Майн Рід "Вершник без голови" книга',
            "Легендарний роман Томаса Майна Ріда. Ілюстрації до розділів.",
            {"book_title": "Вершник без голови"},
            "Vershnik bez golovy",
        ),
        (
            'Шарлотта Бронте "Джейн Ейр" - подарункове видання',
            "Класичний роман. Тверда палітурка з художньою обкладинкою. Стан нової книги.",
            {"book_title": "Джейн Ейр", "cover_type": "hard", "condition": "new"},
            "Jane Eyre hardcover new",
        ),
        (
            "Elle Kennedy серія Briar U (1-4 томи) англійською мовою",
            "Повна серія романів Briar U. Стан: нові книги. Мова: англійська.",
            {"is_set": True, "language": "en", "condition": "new"},
            "Elle Kennedy set english",
        ),
        (
            'Кайлі Скотт "Останні дні Лайли Гудлак" - роман у чудовому стані',
            "Тверда палітурка. Прочитана лише один раз. Чудовий стан без пошкоджень.",
            {"book_title": "Останні дні Лайли Гудлак", "cover_type": "hard", "condition": "like_new"},
            "Kyli Scott like_new hard",
        ),
        (
            "Американський психопат - Брет Істон Елліс (українське видання)",
            "Культовий роман Брета Істона Елліса в українському перекладі. Книжка у чудовому стані.",
            {"language": "uk", "condition": "like_new"},
            "American Psycho ukrainian",
        ),
    ]

    for title, desc, expected, label in cases:
        total += 1
        if _run(795, title, desc, expected, label):
            passed += 1

    print(f"  Result: {passed}/{total}")
    return passed, total


def test_cat1677_figures():
    print("\n=== Category 1677: Collectible Figures ===")
    total = 0
    passed = 0

    cases = [
        (
            "Золотий Гаррі Поттер Kinder Joy колекція Квідич фігурка",
            "Колекційна фігурка Гаррі Поттера з серії Kinder Joy",
            {"series": "Kinder Joy", "franchise": "Гаррі Поттер"},
            "Kinder Joy Harry Potter",
        ),
        (
            "Фігурка Робін з Виворіту Stranger Things Kinder Joy колекційна",
            "Колекційна фігурка Робін з серіалу Stranger Things із серії Kinder Joy. Частина лімітованої колекції.",
            {"series": "Kinder Joy", "franchise": "Stranger Things", "is_rare": True},
            "Stranger Things Robin limited",
        ),
        (
            "Фігурка Орочімару Наруто - колекційна модель аніме персонажа",
            "Колекційна фігурка легендарного Орочімару з аніме Наруто.",
            {"franchise": "Наруто"},
            "Orochimaru Naruto",
        ),
        (
            "Kinder Joy Stranger Things Демогорган колекційна іграшка",
            "Колекційна іграшка Демогорган з серії Kinder Joy Stranger Things.",
            {"series": "Kinder Joy", "franchise": "Stranger Things"},
            "Demogorgon Kinder Joy",
        ),
        (
            "Колекційні іграшки Friends McDonald's",
            "Оригінальні колекційні іграшки з серіалу Friends від McDonald's. У наборі: фігурка Фібі.",
            {"series": "McDonald's", "franchise": "Friends"},
            "Friends McDonalds",
        ),
    ]

    for title, desc, expected, label in cases:
        total += 1
        if _run(1677, title, desc, expected, label):
            passed += 1

    print(f"  Result: {passed}/{total}")
    return passed, total


def test_cat743_constructors():
    print("\n=== Category 743: Constructors ===")
    total = 0
    passed = 0

    cases = [
        (
            "LEGO Minecraft 21189 - новий набір з пломбами",
            "Оригінальний конструктор LEGO Minecraft 21189 у новому стані з неушкодженими заводськими пломбами.",
            {"brand": "LEGO", "set_number": "21189", "theme": "Minecraft", "condition": "new", "completeness": "complete"},
            "LEGO Minecraft 21189 sealed",
        ),
        (
            "LEGO City 60197 Пасажирський поїзд з інструкцією",
            "Повний набір LEGO City 60197. Повна інструкція з усіма етапами збирання.",
            {"brand": "LEGO", "set_number": "60197", "theme": "City"},
            "LEGO City 60197 train",
        ),
        (
            "LEGO Icons Transformers Bumblebee 10338 (950 деталей)",
            "Колекційний конструктор LEGO Icons. 950 деталей для складання. Артикул: 10338. Новий, не відкривався.",
            {"brand": "LEGO", "set_number": "10338", "piece_count": 950, "theme": "Icons", "condition": "new", "completeness": "complete"},
            "LEGO Icons 10338 950pcs",
        ),
        (
            "LEGO деталі оригінальні, поліцейська тематика + додаткові кубики",
            "Оригінальні деталі LEGO поліцейської тематики. Більшість деталей у хорошому стані.",
            {"brand": "LEGO", "condition": "good"},
            "LEGO police parts used",
        ),
        (
            "LEGO DUPLO Town Поліцейська дільниця + мотоцикл (10902+10900)",
            "Два набори LEGO DUPLO в одній коробці для малюків від 2 років.",
            {"brand": "LEGO", "theme": "DUPLO", "has_box": True},
            "LEGO DUPLO police box",
        ),
    ]

    for title, desc, expected, label in cases:
        total += 1
        if _run(743, title, desc, expected, label):
            passed += 1

    print(f"  Result: {passed}/{total}")
    return passed, total


def test_cat1320_chairs():
    print("\n=== Category 1320: Chairs ===")
    total = 0
    passed = 0

    cases = [
        (
            "Офісне крісло рожеве з екошкіри, регульоване, до 120 кг",
            "Стильне офісне крісло. М'яке велике сидіння з екошкіри. Регульована висота.",
            {"type": "office", "material": "eco_leather"},
            "Office eco_leather",
        ),
        (
            "Геймерське крісло Nowy Styl Hexter чорно-червоне з подушками",
            "Ігрове крісло Nowy Styl Hexter. Матеріал: якісна екошкіра.",
            {"type": "gaming", "brand": "Nowy Styl", "material": "eco_leather"},
            "Gaming Nowy Styl Hexter",
        ),
        (
            "Офісне крісло BONRO з екошкіри, коричневе, б/в",
            "Офісне крісло бренду BONRO з екошкіри. Стан: вживане.",
            {"type": "office", "brand": "BONRO", "material": "eco_leather", "condition": "good"},
            "BONRO office used",
        ),
        (
            "Ортопедичне крісло Herman Miller SAYL Найкраще на ринку",
            "Ортопедичне крісло Herman Miller SAYL у ідеальному стані. Сітка на спинці.",
            {"type": "ergonomic", "brand": "Herman Miller", "material": "mesh", "condition": "like_new"},
            "Herman Miller SAYL mesh ideal",
        ),
        (
            "Стілець мінімалістичний темно-синій з металевими ніжками",
            "Сучасний стілець. М'яке сидіння та спинка з якісною тканинною оббивкою. Міцні металеві ніжки.",
            {"material": "fabric"},
            "Minimalist fabric chair",
        ),
    ]

    for title, desc, expected, label in cases:
        total += 1
        if _run(1320, title, desc, expected, label):
            passed += 1

    print(f"  Result: {passed}/{total}")
    return passed, total


def test_cat1261_wheels():
    print("\n=== Category 1261: Wheels/Tires ===")
    total = 0
    passed = 0

    cases = [
        (
            "Оригінальні диски Volkswagen R16 5/112, чудовий стан",
            "Комплект з 4 оригінальних дисків Volkswagen R16.",
            {"brand": "Volkswagen", "size_r": 16, "condition": "like_new"},
            "VW R16 disks",
        ),
        (
            "Оригінальний диск Toyota R17 5x114.3 - новий, не використовувався",
            "Оригінальний литий диск Toyota діаметром 17 дюймів.",
            {"brand": "Toyota", "size_r": 17, "condition": "new"},
            "Toyota R17 new",
        ),
        (
            "Зимові шини Michelin 205/55 R16 комплект",
            "Комплект зимових шин Michelin 205/55 R16. Протектор 5мм.",
            {"brand": "Michelin", "size_r": 16, "width": 205, "profile": 55, "width_profile": "205/55", "season": "winter"},
            "Michelin 205/55 R16 winter",
        ),
        (
            "Літні шини Continental 225/45 R18 б/у",
            "Continental ContiSportContact 225/45 R18. Стан: б/у.",
            {"brand": "Continental", "size_r": 18, "width_profile": "225/45", "season": "summer", "condition": "good"},
            "Continental summer 225/45 R18",
        ),
        (
            "Диски BMW R19 стиль 313 оригінал",
            "Оригінальні диски BMW R19. Стан диска хороший.",
            {"brand": "BMW", "size_r": 19, "condition": "good"},
            "BMW R19 original",
        ),
    ]

    for title, desc, expected, label in cases:
        total += 1
        if _run(1261, title, desc, expected, label):
            passed += 1

    print(f"  Result: {passed}/{total}")
    return passed, total


def test_meta_features():
    print("\n=== Meta features ===")
    ext = CategoryFeatureExtractor(4)
    title = "iPhone 15 Pro Max 256GB"
    desc = "Опис товару з emoji: \U0001f525\U0001f4aa"
    result = ext.extract(
        title,
        desc,
        photo_count=5,
        created_at="2026-01-15 10:30:00+00:00",
    )
    checks = {
        "title_length": len(title),
        "desc_length": len(desc),
        "photo_count": 5,
        "has_emoji": True,
        "month": 1,
        "day_of_week": 3,  # Thursday
    }
    ok = True
    for k, v in checks.items():
        if result.get(k) != v:
            print(f"  FAIL meta [{k}]: expected={v!r}, got={result.get(k)!r}")
            ok = False
    if ok:
        print("  PASS meta features")
    print(f"  Result: {'1/1' if ok else '0/1'}")
    return (1, 1) if ok else (0, 1)


def test_get_missing():
    print("\n=== get_missing() ===")
    ext = CategoryFeatureExtractor(4)
    result = ext.extract("iPhone 15 256GB", "Тест", photo_count=1)
    missing = ext.get_missing(result)
    # condition, color, battery_pct should be missing
    has_condition_missing = "condition" in missing
    has_color_missing = "color" in missing
    ok = has_condition_missing and has_color_missing
    if ok:
        print(f"  PASS get_missing (missing={missing})")
    else:
        print(f"  FAIL get_missing (missing={missing}, found={result})")
    print(f"  Result: {'1/1' if ok else '0/1'}")
    return (1, 1) if ok else (0, 1)


if __name__ == "__main__":
    grand_total = 0
    grand_passed = 0

    for test_fn in [
        test_cat4_smartphones,
        test_cat512_sneakers,
        test_cat795_books,
        test_cat1677_figures,
        test_cat743_constructors,
        test_cat1320_chairs,
        test_cat1261_wheels,
        test_meta_features,
        test_get_missing,
    ]:
        p, t = test_fn()
        grand_passed += p
        grand_total += t

    print(f"\n{'='*50}")
    print(f"  TOTAL: {grand_passed}/{grand_total} passed")
    print(f"{'='*50}")
