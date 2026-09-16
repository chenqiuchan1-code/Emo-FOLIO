#!/usr/bin/env python3
"""Validate the public data subset without calling any model API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EMOTIONS = {"快乐", "悲伤", "恐惧", "愤怒", "惊讶", "平静", "好奇"}
INTENTS = {"安抚", "激励", "逗趣", "引发好奇", "制造紧张", "表达悲悯", "信息传递", "教育启发"}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    return obj


def validate(repo_root: Path, first_id: int = 1, last_id: int = 50) -> list[str]:
    books_dir = repo_root / "data" / "books"
    annotations_dir = repo_root / "data" / "annotations"
    images_dir = repo_root / "data" / "images"
    errors: list[str] = []

    expected = {f"book_{index}" for index in range(first_id, last_id + 1)}
    actual_books = {path.stem for path in books_dir.glob("book_*.json")}
    actual_annotations = {
        path.name.removesuffix("_annotated.json")
        for path in annotations_dir.glob("book_*_annotated.json")
    }
    actual_image_dirs = {path.name for path in images_dir.glob("book_*") if path.is_dir()}

    for label, actual in (
        ("book JSON", actual_books),
        ("annotation JSON", actual_annotations),
        ("image directory", actual_image_dirs),
    ):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{label}: missing {missing}")
        if extra:
            errors.append(f"{label}: unexpected {extra}")

    for book_id in sorted(expected, key=lambda value: int(value.split("_")[-1])):
        book_path = books_dir / f"{book_id}.json"
        annotation_path = annotations_dir / f"{book_id}_annotated.json"
        if not book_path.exists() or not annotation_path.exists():
            continue

        try:
            book = _load(book_path)
            annotation = _load(annotation_path)
        except Exception as exc:
            errors.append(f"{book_id}: invalid JSON ({exc})")
            continue

        if book.get("book_id") != book_id:
            errors.append(f"{book_id}: book_id mismatch in {book_path.name}")
        if annotation.get("book_id") != book_id:
            errors.append(f"{book_id}: book_id mismatch in {annotation_path.name}")

        pages = book.get("pages")
        annotated_pages = annotation.get("pages")
        if not isinstance(pages, list) or not isinstance(annotated_pages, list):
            errors.append(f"{book_id}: pages must be lists")
            continue
        if len(pages) != len(annotated_pages):
            errors.append(
                f"{book_id}: {len(pages)} source pages != {len(annotated_pages)} annotated pages"
            )

        page_numbers = [page.get("page") for page in pages if isinstance(page, dict)]
        if page_numbers != list(range(1, len(pages) + 1)):
            errors.append(f"{book_id}: page numbers are not contiguous from 1")

        annotated_by_page = {
            page.get("page"): page for page in annotated_pages if isinstance(page, dict)
        }
        picture_dir_value = book.get("picture_dir", f"../images/{book_id}")
        picture_dir = (book_path.parent / picture_dir_value).resolve()

        for page in pages:
            if not isinstance(page, dict):
                errors.append(f"{book_id}: non-object page entry")
                continue
            page_number = page.get("page")
            image_name = page.get("image")
            if not isinstance(image_name, str) or not (picture_dir / image_name).is_file():
                errors.append(f"{book_id} page {page_number}: missing image {image_name!r}")

            annotated = annotated_by_page.get(page_number)
            if not annotated:
                errors.append(f"{book_id} page {page_number}: missing annotation")
                continue
            if page.get("text") != annotated.get("text") or image_name != annotated.get("image"):
                errors.append(f"{book_id} page {page_number}: source/annotation content mismatch")
            labels = annotated.get("labels")
            if not isinstance(labels, dict) or not labels.get("primary_emotion"):
                errors.append(f"{book_id} page {page_number}: incomplete labels")
                continue
            primary = labels.get("primary_emotion")
            secondary = labels.get("secondary_emotions", [])
            intent = labels.get("intent_tag")
            trends = labels.get("prev_trend_marks", [])
            if primary not in EMOTIONS or any(value not in EMOTIONS for value in secondary):
                errors.append(f"{book_id} page {page_number}: unknown emotion label")
            if primary in secondary or len(secondary) != len(set(secondary)):
                errors.append(f"{book_id} page {page_number}: duplicated emotion label")
            if intent not in INTENTS:
                errors.append(f"{book_id} page {page_number}: unknown intent label {intent!r}")
            if any(not isinstance(mark, str) or mark[:-1] not in EMOTIONS or mark[-1:] not in "+=-" for mark in trends):
                errors.append(f"{book_id} page {page_number}: malformed trend mark")

        segments = annotation.get("intent_segments", [])
        expected_start = 1
        for segment in segments:
            if not isinstance(segment, dict):
                errors.append(f"{book_id}: non-object intent segment")
                continue
            start = segment.get("start_page", segment.get("start"))
            end = segment.get("end_page", segment.get("end"))
            intent = segment.get("intent")
            if start != expected_start or not isinstance(end, int) or end < start:
                errors.append(f"{book_id}: non-contiguous or invalid intent segment")
                break
            if intent not in INTENTS:
                errors.append(f"{book_id}: unknown segment intent {intent!r}")
            expected_start = end + 1
        if segments and expected_start != len(pages) + 1:
            errors.append(f"{book_id}: intent segments do not cover all pages")

    return errors


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--first-id", type=int, default=1)
    parser.add_argument("--last-id", type=int, default=50)
    args = parser.parse_args()

    errors = validate(args.repo_root.resolve(), args.first_id, args.last_id)
    if errors:
        print(f"Validation failed with {len(errors)} issue(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Validation passed: books {args.first_id}-{args.last_id} have aligned source JSON, "
        "annotations, and page images."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
