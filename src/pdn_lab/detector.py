from __future__ import annotations

from .rules import find_structured, overlaps
from .schema import validate_entities
from .name_context import refine_name_fields


class Detector:
    def __init__(self, name_model=None, rules_only: bool = False):
        if name_model is None and not rules_only:
            raise ValueError("Для полного распознавания требуется обученная модель")
        self.name_model = name_model
        self.rules_only = rules_only

    def detect_many(self, texts: list[str], batch_size: int = 16):
        names = self.name_model.predict_many(texts, batch_size) if self.name_model else [[] for _ in texts]
        results = []
        for text, model_entities in zip(texts, names):
            model_entities = refine_name_fields(text, model_entities)
            structured = find_structured(text)
            merged = structured + [entity for entity in model_entities if not any(overlaps(entity, rule) for rule in structured)]
            merged = sorted(set(merged), key=lambda entity: (entity.start, entity.end))
            validate_entities(text, merged)
            results.append(merged)
        return results

    def detect(self, text: str) -> list[list]:
        return [entity.as_list() for entity in self.detect_many([text])[0]]
