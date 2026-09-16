#!/usr/bin/env python3
"""Resize release images in place while preserving aspect ratio and file names."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=repo_root / "data" / "images")
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()

    changed = 0
    total = 0
    for path in sorted(args.image_root.rglob("*.png")):
        total += 1
        with Image.open(path) as image:
            image.load()
            if max(image.size) <= args.max_side:
                continue
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            image.save(path, format="PNG", optimize=True)
            changed += 1
    print(f"Processed {total} PNG files; resized {changed} to a maximum side of {args.max_side}px.")


if __name__ == "__main__":
    main()
