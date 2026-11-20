"""Заменить найденные ФИО и реквизиты вымышленными значениями."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import secrets
import sys

from faker.providers.person.ru_RU import Provider as RussianNames
import phonenumbers
from pdn_lab.detector import Detector
from pdn_lab.model import NameModel, default_model_directory
from pdn_lab.validators import digits, valid_card, valid_inn, valid_phone, valid_snils


# Первый подходящий псевдоним выбирается в порядке появления людей в тексте.
NAME_POOLS = {
    ("NAME", "female"): ("Елена", "Ирина", "Вера", *RussianNames.first_names_female,
                         "Маша", "Даша", "Катя", "Оля", "Аня", "Настя", "Саша", "Женя"),
    ("NAME", "male"): ("Антон", "Роман", "Михаил", *RussianNames.first_names_male,
                       "Ваня", "Петя", "Серёжа", "Коля", "Миша", "Дима", "Саша", "Женя"),
    ("SURNAME", "female"): ("Белова", "Миронова", "Крылова", *RussianNames.last_names_female),
    ("SURNAME", "male"): ("Белов", "Миронов", "Крылов", *RussianNames.last_names_male),
    ("PATRONYMIC", "female"): ("Павловна", "Олеговна", "Романовна", *RussianNames.middle_names_female),
    ("PATRONYMIC", "male"): ("Павлович", "Олегович", "Романович", *RussianNames.middle_names_male),
}
PERSON_LABELS = {"NAME", "SURNAME", "PATRONYMIC"}
WORD = re.compile(r"[^\W\d_]+")
DIGIT = re.compile(r"\d")
NAME_GAP = re.compile(r"[^\S\r\n]*,?[^\S\r\n]*")
PRECEDING_WORD = re.compile(r"([^\W\d_]+)[^\S\r\n]+$")
PREPOSITIONS = {
    "от": 1, "для": 1, "у": 1, "без": 1, "к": 2, "ко": 2,
    "с": 4, "со": 4, "перед": 4, "между": 4, "о": 5, "об": 5, "при": 5,
}


def normalized(word):
    return word.casefold().replace("ё", "е")


@lru_cache(maxsize=12000)
def declined(name, label, gender):
    """Распространённые формы в порядке И, Р, Д, В, Т, П."""
    low = normalized(name)
    if not name.isalpha():
        return (name,) * 6
    if label == "SURNAME" and low.endswith(("ова", "ева", "ина", "ына")):
        stem, endings = name[:-1], ("ой", "ой", "у", "ой", "ой")
    elif label == "SURNAME" and low.endswith(("ов", "ев", "ин", "ын")):
        stem, endings = name, ("а", "у", "а", "ым", "е")
    elif label == "SURNAME" and low.endswith(("ий", "ый", "ой", "ая", "яя")):
        stem = name[:-2]
        if gender == "female":
            endings = ("ей", "ей", "юю", "ей", "ей") if low.endswith("яя") else ("ой", "ой", "ую", "ой", "ой")
        else:
            endings = ("ого", "ому", "ого", "им" if low.endswith("ий") else "ым", "ом")
    elif label == "SURNAME" and (low.endswith(("ко", "их", "ых")) or (gender == "female" and not low.endswith(("а", "я")))):
        return (name,) * 6
    elif low.endswith("ия"):
        stem, endings = name[:-1], ("и", "и", "ю", "ей", "и")
    elif low.endswith("а"):
        stem = name[:-1]
        genitive = "и" if stem.lower().endswith(tuple("гкхжчшщ")) else "ы"
        instrumental = "ей" if stem.lower().endswith(tuple("жчшщц")) else "ой"
        endings = (genitive, "е", "у", instrumental, "е")
    elif low.endswith("я"):
        stem, endings = name[:-1], ("и", "е", "ю", "ей", "е")
    elif low.endswith("ь") and gender == "female":
        stem, endings = name[:-1], ("и", "и", "ь", "ью", "и")
    elif low.endswith(("й", "ь")):
        stem = name[:-1]
        endings = ("я", "ю", "я", "ем", "и" if low.endswith("ий") else "е")
    elif label == "PATRONYMIC" and low.endswith("ич"):
        stem = name
        endings = ("а", "у", "а", "ом" if low in {"ильич", "кузьмич", "фомич"} else "ем", "е")
    elif low.endswith(tuple("еёиоуыэю")):
        return (name,) * 6
    else:
        stem = {"лев": "Льв", "павел": "Павл", "петр": "Петр"}.get(low, name)
        endings = ("а", "у", "а", "ом", "е")
    return (name, *(stem + ending for ending in endings))


@lru_cache(maxsize=1)
def name_index():
    index = {}
    for (label, gender), pool in NAME_POOLS.items():
        for name in dict.fromkeys(pool):
            for case, surface in enumerate(declined(name, label, gender)):
                index.setdefault((label, normalized(surface)), []).append((name, gender, case))
    return index


def name_readings(label, word):
    surface = normalized(word)
    known = name_index().get((label, surface))
    if known:
        return known
    # Регулярные фамилии и отчества, отсутствующие в словаре Faker.
    candidates = []
    if label == "PATRONYMIC":
        male = re.fullmatch(r"(.+ич)(?:а|у|ем|ом|е)?", surface)
        female = re.fullmatch(r"(.+(?:вн|чн|шн))(?:а|ы|е|у|ой)", surface)
        if male:
            candidates.append((male[1], "male"))
        elif female:
            candidates.append((female[1] + "а", "female"))
    elif label == "SURNAME":
        family = re.fullmatch(r"(.+(?:ов|ев|ин|ын))(?:а|ой|у|ым|е)?", surface)
        if family:
            candidates = [(family[1], "male"), (family[1] + "а", "female")]
    return [(base, gender, case) for base, gender in candidates
            for case, form in enumerate(declined(base, label, gender)) if normalized(form) == surface]


def matching_cases(forms):
    return {(i, j) for i, current in enumerate(forms) for j in range(i) if forms[j] == current}


class NameReplacements:
    def __init__(self, original_words):
        self.originals = {normalized(word) for word in original_words}
        self.aliases = {}
        self.used = set()

    def alias_for(self, label, gender, name):
        key = (label, gender, normalized(name))
        if key not in self.aliases:
            ambiguity = matching_cases(declined(name, label, gender))
            first_available = None
            for candidate in NAME_POOLS[label, gender]:
                if (label, candidate) in self.used:
                    continue
                forms = declined(candidate, label, gender)
                if any(normalized(form) in self.originals for form in forms):
                    continue
                if len(name) == 1 and normalized(candidate[0]) == normalized(name):
                    continue
                first_available = first_available or candidate
                # У «Марии» совпадают три падежа. «Виктории» подходит во всех трёх.
                if ambiguity <= matching_cases(forms):
                    break
            else:
                candidate = first_available or f"Персона-{len(self.aliases) + 1}"
            self.aliases[key] = candidate
            self.used.add((label, candidate))
        return self.aliases[key]

    def replace(self, word, label, gender_hint, case_hint):
        readings = name_readings(label, word)
        if readings:
            name, gender, case = min(readings, key=lambda r: (r[1] != gender_hint, r[2] != case_hint, r[2]))
        else:
            female_ending = normalized(word).endswith(("а", "я"))
            gender = gender_hint or ("female" if female_ending else "male")
            name, case = word, case_hint or 0
        alias = self.alias_for(label, gender, name)
        replacement = declined(alias, label, gender)[case]
        if len(word) == 1:
            replacement = replacement[0]
        if word.isupper():
            return replacement.upper()
        return replacement.lower() if word.islower() else replacement


def mask_names(text, hits):
    names = [(label, start, end) for label, start, end in hits if label in PERSON_LABELS]
    aliases = NameReplacements(word for _, start, end in names for word in WORD.findall(text[start:end]))
    masked = {}
    first = 0
    while first < len(names):
        last = first + 1
        labels = {names[first][0]}
        while (last < len(names) and names[last][0] not in labels
               and NAME_GAP.fullmatch(text, names[last - 1][2], names[last][1])):
            labels.add(names[last][0])
            last += 1
        group = names[first:last]
        start = group[0][1]
        before = PRECEDING_WORD.search(text, max(0, start - 32), start)
        case_hint = PREPOSITIONS.get(before[1].casefold()) if before else None
        gender_hint = None
        # Отчество позволяет различить, например, «Иванова» и «от Иванова».
        for preferred in ("PATRONYMIC", "NAME"):
            for label, start, end in group:
                if label != preferred:
                    continue
                genders = {r[1] for r in name_readings(label, text[start:end])}
                if len(genders) == 1:
                    gender_hint = genders.pop()
                    break
            if gender_hint:
                break
        for label, start, end in group:
            def replace_word(match):
                return aliases.replace(match[0], label, gender_hint, case_hint)
            masked[start] = WORD.sub(replace_word, text[start:end]) if WORD.search(text[start:end]) else f"[{label}]"
        first = last
    return masked


CHECKED_NUMBERS = {"CARD": (4, valid_card), "INN": (2, valid_inn), "SNILS": (2, valid_snils)}


def format_digits(template, number):
    chars = list(template)
    for match, digit in zip(DIGIT.finditer(template), number, strict=True):
        # У каждой группы десятичных цифр Unicode значения 0–9 идут подряд.
        chars[match.start()] = chr(ord(match[0]) - int(match[0]) + int(digit))
    return "".join(chars)


def random_bodies(width):
    """Перебор без повторов со случайного места и со случайным шагом."""
    size = 10 ** width
    position = secrets.randbelow(size)
    # Взаимно простой с 10 шаг обходит все комбинации, в том числе с ведущими нулями.
    step = 10 * secrets.randbelow(size // 10) + secrets.choice((1, 3, 7, 9))
    for _ in range(min(size, 16_384)):
        yield str(position).zfill(width)
        position = (position + step) % size


@dataclass(frozen=True)
class PhoneLayout:
    country: int
    dial_prefix: str
    national: str
    extension: str
    area_width: int
    check_plan: bool
    international: bool
    canonical_national: bool

    @classmethod
    def parse(cls, value):
        try:
            phone = phonenumbers.parse(value, "RU")
        except phonenumbers.NumberParseException as error:
            raise ValueError("Не удалось разобрать распознанный телефон.") from error
        raw = phonenumbers.normalize_digits_only(value)
        extension = phone.extension or ""
        main = raw.removesuffix(extension) if extension else raw
        national = phonenumbers.national_significant_number(phone)
        canonical_national = main.endswith(national)
        if canonical_national:
            prefix = main.removesuffix(national)
        else:
            # В некоторых международных записях оператора указывают внутри номера.
            prefix = str(phone.country_code) if value.startswith("+") else ""
            national = main[len(prefix):]
        area = re.search(r"\(\s*(\d{1,5})\s*\)", value)
        area_width = 0
        if area:
            before = len(DIGIT.findall(value, 0, area.start()))
            if before <= len(prefix) <= before + len(area[1]):
                area_width = before + len(area[1]) - len(prefix)
        return cls(phone.country_code, prefix, national, extension, area_width,
                   phonenumbers.is_valid_number(phone), value.startswith("+"), canonical_national)

    def accepts(self, national):
        if self.country == 7 and not valid_phone("7" + national):
            return False
        if self.country == 7 and not self.check_plan:
            return True  # Проект принимает также российские номера по формату.
        try:
            dialed = ("+" if self.international else "") + self.dial_prefix + national
            phone = phonenumbers.parse(dialed, "RU")
            if self.canonical_national and phonenumbers.national_significant_number(phone) != national:
                return False
            return phone.country_code == self.country and phonenumbers.is_valid_number(phone)
        except phonenumbers.NumberParseException:
            return False


class StructuredReplacements:
    def __init__(self, text, hits):
        self.occupied = defaultdict(set)
        self.aliases = {}
        self.phones = {}
        self.phone_formats = defaultdict(set)
        self.extension_shifts = {}
        # Учитываем также исходные значения, которые встретятся дальше по тексту.
        for label, start, end in hits:
            value = text[start:end]
            if label in CHECKED_NUMBERS:
                self.occupied[label].add(digits(value))
            elif label == "EMAIL":
                local, domain = value.rsplit("@", 1)
                self.occupied[("EMAIL", domain.casefold())].add(local)
            elif label == "PHONE":
                phone = self.phones[start] = PhoneLayout.parse(value)
                key = (phone.country, phone.national)
                self.phone_formats[key].add(phone)
                self.occupied[("PHONE", phone.country)].add(phone.national)

    def assign(self, scope, original, width, *, prefix="", suffix="", validate=None):
        key = (scope, original)
        if key in self.aliases:
            return self.aliases[key]
        if width <= 0:
            raise ValueError("В номере нет свободных цифр для замены с заданным окончанием.")
        prefixes = [prefix]
        if isinstance(scope, tuple) and scope[0] == "PHONE" and width > 3:
            # Запасной поиск внутри исходного диапазона для редких планов нумерации.
            prefixes.append(original[:len(prefix) + width - 3])
        for head in prefixes:
            free = len(prefix) + width - len(head)
            for body in random_bodies(free):
                candidate = head + body + suffix
                if candidate in self.occupied[scope] or (validate and not validate(candidate)):
                    continue
                self.occupied[scope].add(candidate)
                self.aliases[key] = candidate
                return candidate
        raise ValueError("Не удалось подобрать отличающуюся замену с заданными открытыми цифрами.")

    def phone_value(self, value, phone):
        key = (phone.country, phone.national)
        formats = self.phone_formats[key]
        area_width = max(layout.area_width for layout in formats)
        national = self.assign(("PHONE", phone.country), phone.national,
                               len(phone.national) - area_width - 2,
                               prefix=phone.national[:area_width], suffix=phone.national[-2:],
                               validate=lambda number: all(layout.accepts(number) for layout in formats))
        extension = phone.extension
        if extension:
            # Разные добавочные одного телефона остаются различимыми даже в диапазоне 0–9.
            width = len(extension)
            ext_key = (*key, width)
            size = 10 ** width
            if ext_key not in self.extension_shifts:
                self.extension_shifts[ext_key] = 1 + secrets.randbelow(size - 1)
            extension = str((int(extension) + self.extension_shifts[ext_key]) % size).zfill(width)
        return format_digits(value, phone.dial_prefix + national + extension)

    def replace(self, label, value, start):
        if label in CHECKED_NUMBERS:
            tail, validate = CHECKED_NUMBERS[label]
            original = digits(value)
            number = self.assign(label, original, len(original) - tail,
                                 suffix=original[-tail:], validate=validate)
            return format_digits(value, number)
        if label == "PHONE":
            return self.phone_value(value, self.phones[start])
        if label == "EMAIL":
            local, domain = value.rsplit("@", 1)
            mailbox = self.assign(("EMAIL", domain.casefold()), local, 8)
            return mailbox + "@" + domain
        return f"[{label}]"


def mask_text(text, hits):
    names = mask_names(text, hits)
    fields = StructuredReplacements(text, hits)
    fragments = []
    cursor = 0
    for label, start, end in hits:
        value = text[start:end]
        if label in PERSON_LABELS:
            replacement = names[start]
        else:
            replacement = fields.replace(label, value, start)
        fragments.extend((text[cursor:start], replacement))
        cursor = end
    fragments.append(text[cursor:])
    return "".join(fragments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Путь к файлу в кодировке UTF-8")
    args = parser.parse_args()

    try:
        text = args.file.read_bytes().decode("utf-8-sig")
        detector = Detector(NameModel.load(default_model_directory()))
        hits = detector.detect(text)
        masked = mask_text(text, hits)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")

    sys.stdout.reconfigure(encoding="utf-8", newline="")
    print(f"Найдено фрагментов: {len(hits)}")
    print(json.dumps(hits, ensure_ascii=False))
    print()
    sys.stdout.write(masked)
    if not text.endswith(("\r", "\n")):
        print()


if __name__ == "__main__":
    main()
