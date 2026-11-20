"""Проверки замены ПДн с сохранением открытой части номера."""

import json
from pathlib import Path
import random
import re
import runpy
import subprocess
import sys

import phonenumbers
import pytest

from pdn_lab.model import default_model_directory
from pdn_lab.rules import find_structured
from pdn_lab.validators import digits, make_card, make_inn, snils_check, valid_card, valid_inn, valid_phone, valid_snils

PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts" / "mask_file.py"
FIXTURES = Path(__file__).parent / "fixtures" / "masking"
mask_text = runpy.run_path(str(RUNNER))["mask_text"]


def sample(marked, expected):
    source = []
    hits = []
    cursor = position = 0
    for hit in re.finditer(r'\[\[(\w+)\|([^\]]+)\]\]', marked):
        gap = marked[cursor:hit.start()]
        tag, value = hit.groups()
        position += len(gap)
        hits.append((tag, position, position + len(value)))
        source.extend((gap, value))
        cursor = hit.end()
        position += len(value)
    source.append(marked[cursor:])
    raw = ''.join(source)
    result = mask_text(raw, hits)
    assert result == expected, (raw, result, expected)
    assert mask_text(raw, hits) == result


def test_name_forms_and_repeated_mentions():
    sample('Ничего для маскирования.\r\n', 'Ничего для маскирования.\r\n')
    sample('[[NAME|Марина]] ответила [[NAME|Марине]]. Письмо от [[NAME|Марины]].',
           'Елена ответила Елене. Письмо от Елены.')
    sample('С [[NAME|Мариной]] [[PATRONYMIC|Игоревной]] [[SURNAME|Соколовой-Воронцовой]] встретилась [[NAME|Марина]] [[PATRONYMIC|Игоревна]] [[SURNAME|Соколова]].',
           'С Еленой Павловной Беловой-Мироновой встретилась Елена Павловна Белова.')
    sample('К [[SURNAME|Иванову]] [[NAME|Ивану]] [[PATRONYMIC|Ивановичу]], от [[SURNAME|Иванова]] [[NAME|Ивана]] [[PATRONYMIC|Ивановича]].',
           'К Белову Антону Павловичу, от Белова Антона Павловича.')
    sample('С [[NAME|Иваном]] [[PATRONYMIC|Ивановичем]].',
           'С Антоном Павловичем.')
    sample('[[NAME|Павел]], к [[NAME|Павлу]], нет [[NAME|Павла]]. [[NAME|Лев]] и [[NAME|Льву]].',
           'Антон, к Антону, нет Антона. Роман и Роману.')
    sample('[[NAME|Елена]], [[NAME|Антон]], [[NAME|ЕЛЕНА]] и [[NAME|антон]].',
           'Ирина, Роман, ИРИНА и роман.')
    sample('[[SURNAME|СОКОЛОВА‑ВОРОНЦОВА]], [[SURNAME|Соколова]].',
           'БЕЛОВА‑МИРОНОВА, Белова.')


aliases = {
    'Марина':'Елена', 'Игоревна':'Павловна', 'Соколова-Воронцова':'Белова-Миронова', 'Соколова':'Белова',
    'Лебедев':'Белов','Артём':'Антон','Сергеевич':'Павлович','Соколова-Морозова':'Белова-Миронова',
    'Анна':'Елена','Викторовна':'Павловна','Лев':'Роман','Андреевич':'Олегович','Орлов':'Миронов',
}
validators = {'CARD': valid_card, 'INN': valid_inn, 'SNILS': valid_snils}


def assert_checksum(tag, number):
    n = list(map(int, number))
    if tag == 'CARD':
        doubled = [2 * d for d in n[-2::-2]]
        assert (sum(n[::-2]) + sum(t // 10 + t % 10 for t in doubled)) % 10 == 0
    elif tag == 'SNILS':
        checksum = sum(d * w for d, w in zip(n[:9], range(9, 0, -1))) % 101
        assert int(number[-2:]) == (checksum if checksum < 100 else 0)
    else:
        def control(weights):
            return sum(d * w for d, w in zip(n, weights)) % 11 % 10
        if len(n) == 10:
            assert n[-1] == control([2,4,10,3,5,9,4,6,8])
        else:
            assert len(n) == 12 and n[-2] == control([7,2,4,10,3,5,9,4,6,8])
            assert n[-1] == control([3,7,2,4,10,3,5,9,4,6,8])


def verify_fields(raw, hits, rendered):
    expression = []
    position = 0
    for tag, start, end in hits:
        value = raw[start:end]
        expression.append(re.escape(raw[position:start]))
        if tag in {'NAME', 'SURNAME', 'PATRONYMIC'}:
            pattern = re.escape(aliases[value])
        elif tag == 'EMAIL':
            pattern = r'[0-9]{8}@' + re.escape(value.rsplit('@', 1)[1])
        else:
            pattern = ''.join(r'\d' if char.isdecimal() else re.escape(char) for char in value)
        expression.extend(('(', pattern, ')'))
        position = end
    expression.append(re.escape(raw[position:]))
    match = re.fullmatch(''.join(expression), rendered)
    assert match, 'Text, name, separators, domain or number length changed unexpectedly'
    assigned = {}
    original_keys = set()
    for (tag, start, end), output in zip(hits, match.groups(), strict=True):
        value = raw[start:end]
        if tag in validators:
            src, dst = digits(value), digits(output)
            assert src != dst and validators[tag](dst)
            tail = 4 if tag == 'CARD' else 2
            assert src[-tail:] == dst[-tail:]
            assert_checksum(tag, dst)
            old_key, new_key = (tag, src), (tag, dst)
        elif tag == 'PHONE':
            old = phonenumbers.parse(value, 'RU')
            new = phonenumbers.parse(output, 'RU')
            src = phonenumbers.national_significant_number(old)
            dst = phonenumbers.national_significant_number(new)
            assert old.country_code == new.country_code and src != dst and src[-2:] == dst[-2:]
            assert not phonenumbers.is_valid_number(old) or phonenumbers.is_valid_number(new), (value, output, old, new)
            if old.country_code == 7:
                assert valid_phone('7' + dst)
            explicit_area = re.match(r'(?:\+\d{1,3}|[78])?[\s‑–-]*\(\s*\d{1,5}\s*\)', value)
            if explicit_area:
                assert output.startswith(explicit_area[0])
            elif value.startswith('+'):
                assert output.startswith('+' + str(old.country_code))
            elif len(digits(value)) == 11:
                assert output[0] == value[0]
            if old.extension:
                assert len(new.extension) == len(old.extension) and new.extension != old.extension
            old_key = (tag, old.country_code, src)
            new_key = (tag, new.country_code, dst)
        elif tag == 'EMAIL':
            local, domain = value.rsplit('@', 1)
            alias, new_domain = output.rsplit('@', 1)
            assert domain == new_domain and alias != local
            old_key, new_key = (tag, domain.casefold(), local), (tag, domain.casefold(), alias)
        else:
            continue
        original_keys.add(old_key)
        assert assigned.setdefault(old_key, new_key) == new_key
    assert len(set(assigned.values())) == len(assigned)
    assert original_keys.isdisjoint(assigned.values())
    return match.groups()


def make_fixture(fields):
    parts = ['🙂\r\n']
    hits = []
    offset = len(parts[0])
    for tag, value in fields:
        hits.append((tag, offset, offset + len(value)))
        parts.extend((value, ' | '))
        offset += len(value) + 3
    return ''.join(parts), hits


def test_repeated_values_and_phone_formats():
    probes = [
        ('EMAIL', '"a@b"@пример.example.org'), ('EMAIL', '87654321@example.org'),
        ('EMAIL', '"a@b"@пример.example.org'),
        ('PHONE', '+79000000123'), ('PHONE', '8 (900) 000-01-23'), ('PHONE', '(900) 000-01-23'),
        ('PHONE', '+44 20 7946 0958'), ('PHONE', '+420 212 345 678'), ('PHONE', '+44 (20) 7946 0958'),
        ('PHONE', '+1 (202) 555-0147 ext. 349'), ('PHONE', '+1 (202) 555-0147 ext. 349'),
        ('PHONE', '+1 (202) 555-0147 ext. 350'), ('PHONE', '+1 202 555 (0147)'),
        ('PHONE', '+49 (0)30 12345678'), ('PHONE', '+39 (02) 3661 8300'),
        ('CARD', '0000‑1234‑5678‑9012‑37'), ('CARD', '000012345678901237'),
        ('SNILS', '009‑876‑543 95'), ('SNILS', '00987654395'),
        ('INN', '772953641745'), ('INN', '772953641745'), ('INN', '7704872160'),
        ('CARD', '4111 1111 1111 1111'), ('CARD', '4012 8888 8888 1881'),
    ]
    raw, hits = make_fixture(probes)
    runs = set()
    for _ in range(25):
        result = mask_text(raw, hits)
        groups = verify_fields(raw, hits, result)
        assert groups[9] == groups[10] and groups[10] != groups[11]
        runs.add(result)
    assert len(runs) > 1


def test_checksum_preserving_replacements():
    rng = random.Random(29)

    def draw(width):
        return ''.join(str(rng.randrange(10)) for _ in range(width))

    stress = []
    for width in range(13, 20):
        for _ in range(24):
            stress.append(('CARD', make_card(draw(width - 1))))
    for width in (9, 10):
        for _ in range(64):
            stress.append(('INN', make_inn(draw(width))))
    # Проверяем все 100 возможных окончаний СНИЛС, включая 00.
    snils = {}
    while len(snils) < 100:
        base = str(rng.randrange(1_002_000, 1_000_000_000)).zfill(9)
        check = snils_check(base)
        snils[check] = base + str(check).zfill(2)
    stress.extend(('SNILS', number) for number in snils.values())
    raw, hits = make_fixture(stress)
    verify_fields(raw, hits, mask_text(raw, hits))


def load_example(relative):
    path = PROJECT / relative
    reference = json.loads((FIXTURES / f"{path.stem}_fields.json").read_text(encoding="utf-8"))
    text = path.read_bytes().decode("utf-8-sig")
    spans = []
    cursor = 0
    for label, value in reference:
        start = text.index(value, cursor)
        cursor = start + len(value)
        spans.append([label, start, cursor])
    return text, spans


def verify_example(text, spans, body):
    values = verify_fields(text, spans, body)
    structured = [(tag, value) for (tag, _, _), value in zip(spans, values, strict=True)
                  if tag in validators or tag in {"PHONE", "EMAIL"}]
    found = [(e.label, body[e.start:e.end]) for e in find_structured(body)]
    assert found == structured


def test_published_letter():
    text, spans = load_example("examples/pismo.txt")
    saved = (PROJECT / "examples" / "pismo_masked.txt").read_bytes().decode("utf-8")
    verify_example(text, spans, saved)


@pytest.mark.parametrize("relative", ["examples/pismo.txt", "pdn_all_types.txt"])
def test_mask_file_cli(relative):
    if not (default_model_directory() / "model.safetensors").is_file():
        pytest.skip("Для сквозной проверки нужны готовые веса модели")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    text, expected = load_example(relative)
    result = subprocess.run([sys.executable, "-B", str(RUNNER), str(PROJECT / relative)],
                            cwd=PROJECT, capture_output=True, check=True)
    count, spans, blank, body = result.stdout.decode("utf-8").split("\n", 3)
    assert json.loads(spans) == expected
    assert count == f"Найдено фрагментов: {len(expected)}" and blank == ""
    if not text.endswith(("\n", "\r")):
        text += "\n"
    verify_example(text, expected, body)
