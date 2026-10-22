from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .data import prepare, sha256
from .detector import Detector
from .metrics import evaluate_predictions, markdown_report
from .model import NameModel, default_model_directory, default_data_directory
from .schema import ATTRIBUTES, PROJECT_ROOT, read_jsonl, write_json


def _detector(args) -> Detector:
    if args.rules_only:
        return Detector(rules_only=True)
    return Detector(NameModel.load(args.model_dir, args.device))


def run_evaluation(args) -> dict:
    data_path = args.data_dir / "processed" / f"{args.split}.jsonl"
    manifest = json.loads((args.data_dir / "processed" / "manifest.json").read_text(encoding="utf-8"))
    if sha256(data_path) != manifest["files"][args.split]:
        raise ValueError("Проверяемая выборка изменилась после подготовки")
    records = read_jsonl(data_path)
    detector = _detector(args)
    predictions = detector.detect_many([record["text"] for record in records], args.batch_size)
    scores = evaluate_predictions(records, predictions)
    by_source = {}
    for source in sorted({record["source"] for record in records}):
        indices = [i for i, record in enumerate(records) if record["source"] == source]
        by_source[source] = evaluate_predictions([records[i] for i in indices], [predictions[i] for i in indices])
    model_manifest = None
    compatible_training = False
    if not args.rules_only:
        model_manifest = json.loads((args.model_dir / "training_manifest.json").read_text(encoding="utf-8"))
        compatible_training = all(model_manifest.get(f"{split}_sha256") == manifest["files"][split] for split in ("train", "validation"))
    requirements = {
        "все_8_атрибутов_представлены": all(scores["dataset"]["texts_per_attribute"][label] > 0 for label in ATTRIBUTES),
        "больше_50_текстов_на_каждый_атрибут": all(scores["dataset"]["texts_per_attribute"][label] > 50 for label in ATTRIBUTES),
        "weighted_f1_строго_выше_порога": scores["token_weighted_f1_without_o"] > args.threshold,
        "полный_гибридный_режим": not args.rules_only,
        "обучение_завершено_на_этих_данных": bool(compatible_training and model_manifest.get("training_completed", False)) if model_manifest else False,
        "итоговая_тестовая_выборка": args.split == "test",
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "rules_only" if args.rules_only else "hybrid", "split": args.split,
        "threshold_strictly_greater_than": args.threshold, "evaluation_sha256": sha256(data_path), "data_manifest": manifest,
        "model_manifest": model_manifest, "scores": scores, "by_source": by_source, "requirements": requirements,
        "lab_requirements_met_on_this_dataset": all(requirements.values()),
        "metric_definition": "weighted F1 по B/I-меткам восьми атрибутов, веса по эталонной частоте, O исключён только из усреднения; все токены участвуют в подсчёте ошибок",
    }
    if args.output.exists() and not args.overwrite_report:
        raise FileExistsError("Отчёт уже существует. Укажите другой --output или --overwrite-report.")
    write_json(args.output, report)
    args.output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    if args.save_predictions:
        write_json(args.output.with_name(args.output.stem + "_predictions.json"), {"records": [{"id": record["id"], "entities": [e.as_list() for e in entities]} for record, entities in zip(records, predictions)]})
    return report


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ЛР 1: распознавание восьми атрибутов ПДн")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Подготовить factRuEval и синтетические train/validation/test")
    preparation.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    preparation.add_argument("--seed", type=int, default=20261022)
    preparation.add_argument("--train-count", type=int, default=1800)
    preparation.add_argument("--validation-count", type=int, default=320)
    preparation.add_argument("--test-count", type=int, default=480)
    preparation.add_argument("--synthetic-only", action="store_true", help="Диагностика генератора без скачивания корпуса")
    training = commands.add_parser("train", help="Дообучить RuBERT, выбирать модель только по validation")
    training.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    training.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "rubert-pdn")
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--accumulation", type=int, default=2)
    training.add_argument("--learning-rate", type=float, default=3e-5)
    training.add_argument("--max-length", type=int, default=256)
    training.add_argument("--seed", type=int, default=20261022)
    training.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    training.add_argument("--continue-from", type=Path, help="Продолжить настройку весов в новой папке; оптимизатор создаётся заново")
    evaluation = commands.add_parser("evaluate", help="Оценить всю программу, включая правила и валидаторы")
    evaluation.add_argument("--data-dir", type=Path, default=default_data_directory())
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--threshold", type=float, default=0.85)
    evaluation.add_argument("--batch-size", type=int, default=16)
    evaluation.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "test.json")
    evaluation.add_argument("--overwrite-report", action="store_true")
    evaluation.add_argument("--save-predictions", action="store_true")
    prediction = commands.add_parser("predict", help="Получить список [тип, начало, конец]")
    inputs = prediction.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--file", type=Path, help="Текстовый файл UTF-8; переводы строк сохраняются")
    inputs.add_argument("--interactive", action="store_true")
    for command in (evaluation, prediction):
        command.add_argument("--model-dir", type=Path, default=default_model_directory())
        command.add_argument("--rules-only", action="store_true", help="Только диагностика правил, без распознавания ФИО")
        command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.data_dir, args.seed, args.train_count, args.validation_count, args.test_count, args.synthetic_only)
            print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
        elif args.command == "train":
            from .training import train
            result = train(args.data_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, accumulation=args.accumulation, learning_rate=args.learning_rate, max_length=args.max_length, seed=args.seed, device=args.device, resume_from=args.continue_from)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            report = run_evaluation(args)
            print(json.dumps({"weighted_f1_without_o": report["scores"]["token_weighted_f1_without_o"], "strict_entity_weighted_f1": report["scores"]["strict_entity_weighted_f1"], "requirements": report["requirements"], "lab_requirements_met_on_this_dataset": report["lab_requirements_met_on_this_dataset"], "report": str(args.output)}, ensure_ascii=False, indent=2))
            if not report["lab_requirements_met_on_this_dataset"]:
                return 2
        elif args.command == "predict":
            detector = _detector(args)
            if args.interactive:
                print("Введите текст одной строкой. Завершение: Ctrl+Z, затем Enter (Windows) или Ctrl+D (Linux).", file=sys.stderr)
                for line in sys.stdin:
                    print(json.dumps(detector.detect(line.rstrip("\r\n")), ensure_ascii=False), flush=True)
            else:
                if args.file:
                    with args.file.open("r", encoding="utf-8-sig", newline="") as stream:
                        text = stream.read()
                else:
                    text = args.text
                print(json.dumps(detector.detect(text), ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
