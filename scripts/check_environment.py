"""Check one of the repository's runtime environments without network access."""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib.metadata import PackageNotFoundError, version


SCOPES = {
    "core": {
        "openai": "openai",
        "PIL": "Pillow",
        "matplotlib": "matplotlib",
    },
    "annotation": {
        "streamlit": "streamlit",
        "PIL": "Pillow",
    },
    "ocr": {
        "numpy": "numpy",
        "cv2": "opencv-python-headless",
        "paddle": "paddlepaddle",
        "paddleocr": "paddleocr",
        "PIL": "Pillow",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=tuple(SCOPES),
        default="core",
        help="environment to check (default: core)",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ is required; found {sys.version.split()[0]}")
        return 1

    print(f"Python {sys.version.split()[0]}")
    missing = []
    for module_name, distribution_name in SCOPES[args.scope].items():
        try:
            module = importlib.import_module(module_name)
            try:
                installed_version = version(distribution_name)
            except PackageNotFoundError:
                installed_version = "installed (distribution metadata unavailable)"
            if module_name == "cv2":
                installed_version = getattr(module, "__version__", installed_version)
                print(f"OpenCV {installed_version}")
            else:
                print(f"{distribution_name} {installed_version}")
        except Exception as exc:
            missing.append((distribution_name, str(exc)))

    if missing:
        for name, error in missing:
            print(f"ERROR: {name} is unavailable: {error}")
        install_commands = {
            "core": "pip install -r requirements.txt",
            "annotation": "pip install -r tools/annotation/requirements.txt",
            "ocr": "bash tools/data_processing/setup_ocr_env.sh <environment-name>",
        }
        print(f"Install this runtime with: {install_commands[args.scope]}")
        return 1

    print(f"{args.scope.capitalize()} runtime check passed. No model API was contacted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
