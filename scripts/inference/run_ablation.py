# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_FULL_MOSAIC_ROOT = REPO_ROOT / "results" / "mosaic" / "gpt-4o"
BOOKS_DIR_DEFAULT = str(REPO_ROOT / "data" / "books")
BOOK_GLOB_DEFAULT = "book_*.json"

EXPERIMENTS = {
    "A2": {
        "name": "A2__w_o_stepA",
        "reuse": [],
        "run_from_step": "stepB",
        "run_flags": [
            "--skip-stepA-run",
            "--disable-stepA-input-stepB",
            "--disable-stepA-input-stepC",
            "--inject-page-text-stepB",
            "--inject-page-images-stepB",
        ],
        "need_synthetic_merged": False,
    },
    "A3": {
        "name": "A3__A_fixed_window_C",
        "reuse": ["stepA"],
        "run_from_step": "stepC",
        "run_flags": [
            "--skip-stepB-run",
            "--disable-stepB-input-stepC",
            "--chunk-mode-stepC", "fixed_windows",
        ],
        "need_synthetic_merged": False,
    },
    "A4": {
        "name": "A4__A_full_book_C",
        "reuse": ["stepA"],
        "run_from_step": "stepC",
        "run_flags": [
            "--skip-stepB-run",
            "--disable-stepB-input-stepC",
            "--chunk-mode-stepC", "whole_book",
        ],
        "need_synthetic_merged": False,
    },
    "A5": {
        "name": "A5__A_B_C_full",
        "reuse": ["stepA", "stepB", "stepC", "merged"],
        "run_from_step": None,
        "run_flags": [],
        "need_synthetic_merged": False,
    },
    "B2": {
        "name": "B2__w_o_summary_context",
        "reuse": ["stepA"],
        "run_from_step": "stepB",
        "run_flags": [
            "--skip-stepC-run",
            "--no-book-summary-stepB",
            "--no-page-summaries-stepB",
        ],
        "need_synthetic_merged": True,
    },
    "B3": {
        "name": "B3__w_o_evidence_cues",
        "reuse": ["stepA"],
        "run_from_step": "stepB",
        "run_flags": [
            "--skip-stepC-run",
            "--no-s1-text-cues-stepB",
            "--no-s1-visual-cues-stepB",
        ],
        "need_synthetic_merged": True,
    },
    "B4": {
        "name": "B4__w_o_emotion_candidates",
        "reuse": ["stepA"],
        "run_from_step": "stepB",
        "run_flags": [
            "--skip-stepC-run",
            "--no-s1-candidates-stepB",
        ],
        "need_synthetic_merged": True,
    },
    "B5": {
        "name": "B5__w_o_rationale_output",
        "reuse": ["stepA"],
        "run_from_step": "stepB",
        "run_flags": [
            "--skip-stepC-run",
            "--no-rationale-output-stepB",
        ],
        "need_synthetic_merged": True,
    },
    "C2": {
        "name": "C2__w_o_evidence_cues",
        "reuse": ["stepA", "stepB"],
        "run_from_step": "stepC",
        "run_flags": [
            "--no-inject-book-summary-stepC",
            "--no-inject-s1-summary-stepC",
            "--no-inject-s1-text-cues-stepC",
            "--no-inject-s1-visual-cues-stepC",
        ],
        "need_synthetic_merged": False,
    },
    "C3": {
        "name": "C3__w_o_anchor_hint",
        "reuse": ["stepA", "stepB"],
        "run_from_step": "stepC",
        "run_flags": [
            "--no-inject-book-summary-stepC",
            "--no-inject-s1-summary-stepC",
            # anchor 通过下面的 stepC_base_flags 条件关闭
        ],
        "need_synthetic_merged": False,
    },
    "C4": {
        "name": "C4__w_o_evidence_and_anchor",
        "reuse": ["stepA", "stepB"],
        "run_from_step": "stepC",
        "run_flags": [
            "--no-inject-book-summary-stepC",
            "--no-inject-s1-summary-stepC",
            "--no-inject-s1-text-cues-stepC",
            "--no-inject-s1-visual-cues-stepC",
            # anchor 通过下面的 stepC_base_flags 条件关闭
        ],
        "need_synthetic_merged": False,
    },
}

def _split_specs(book_glob: str) -> List[str]:
    s = (book_glob or "").strip()
    if not s:
        return []
    parts = re.split(r"[,\s，、]+", s)
    return [p for p in parts if p]

def _expand_specs(books_dir: Path, spec: str) -> List[Path]:
    spec = spec.strip()

    if any(ch in spec for ch in "*?[]"):
        return sorted(books_dir.glob(spec))

    if "-" in spec:
        left, right = spec.rsplit("-", 1)
        left = left.strip()
        right = right.strip()

        left_stripped = re.sub(r"\.(json|JSON)$", "", left)
        right_stripped = re.sub(r"\.(json|JSON)$", "", right)

        mL = re.search(r"(\d+)$", left_stripped)
        mR = re.search(r"(\d+)$", right_stripped)
        if mL and mR:
            s_num, e_num = mL.group(1), mR.group(1)
            a, b = int(s_num), int(e_num)
            if a > b:
                a, b = b, a

            prefixL = left_stripped[:mL.start(1)]
            prefixR = right_stripped[:mR.start(1)]
            common_prefix = os.path.commonprefix([prefixL, prefixR]) if prefixL != prefixR else prefixL
            width = max(len(s_num), len(e_num))

            out = []
            seen = set()
            for i in range(a, b + 1):
                names = [
                    f"{common_prefix}{i}.json",
                    f"{common_prefix}{i}.JSON",
                    f"{common_prefix}{i:0{width}d}.json",
                    f"{common_prefix}{i:0{width}d}.JSON",
                ]
                found = False
                for n in names:
                    p = books_dir / n
                    if p.exists() and p not in seen:
                        out.append(p)
                        seen.add(p)
                        found = True
                        break
                if not found:
                    pass
            return sorted(out, key=lambda x: x.name)

    tried = [spec] if spec.lower().endswith(".json") else [spec, f"{spec}.json", f"{spec}.JSON"]
    for name in tried:
        p = books_dir / name
        if p.exists():
            return [p]
    return []

def resolve_books(books_dir: Path, book_glob: str, batch_limit: int) -> List[Path]:
    specs = _split_specs(book_glob)
    files = []
    seen = set()
    for sp in specs:
        for p in _expand_specs(books_dir, sp):
            if p not in seen:
                files.append(p)
                seen.add(p)
    files = sorted(files, key=lambda x: x.name)
    if batch_limit and batch_limit > 0:
        files = files[:batch_limit]
    return files

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)

def copy_matching_debug(src_dir: Path, dst_dir: Path, prefix: str):
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob(f"{prefix}*"):
        safe_link_or_copy(p, dst_dir / p.name)

def materialize_reuse(base_root: Path, out_root: Path, book_ids: List[str], steps: List[str]):
    finals_src = base_root / "finals"
    merged_src = base_root / "merged"
    debug_src = base_root / "debug"

    finals_dst = out_root / "finals"
    merged_dst = out_root / "merged"
    debug_dst = out_root / "debug"

    for book_id in book_ids:
        if "stepA" in steps:
            p = finals_src / f"{book_id}_STEPA_information.json"
            if p.exists():
                safe_link_or_copy(p, finals_dst / p.name)
            copy_matching_debug(debug_src / "stepA", debug_dst / "stepA", f"{book_id}_STEPA_")

        if "stepB" in steps:
            p = finals_src / f"{book_id}_STEPB_segments.json"
            if p.exists():
                safe_link_or_copy(p, finals_dst / p.name)
            copy_matching_debug(debug_src / "stepB", debug_dst / "stepB", f"{book_id}_STEPB_")

        if "stepC" in steps:
            p = finals_src / f"{book_id}_STEPC_emotions.json"
            if p.exists():
                safe_link_or_copy(p, finals_dst / p.name)
            copy_matching_debug(debug_src / "stepC", debug_dst / "stepC", f"{book_id}_STEPC_")

        if "merged" in steps:
            p = merged_src / f"{book_id}_MOSAIC_merged.json"
            if p.exists():
                safe_link_or_copy(p, merged_dst / p.name)

def synthesize_stepB_only_merged(base_root: Path, out_root: Path, book_ids: List[str]):
    base_merged_root = base_root / "merged"
    stepb_root = out_root / "finals"
    merged_root = out_root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)

    for book_id in book_ids:
        base_merged_path = base_merged_root / f"{book_id}_MOSAIC_merged.json"
        stepb_path = stepb_root / f"{book_id}_STEPB_segments.json"
        if not base_merged_path.exists() or not stepb_path.exists():
            continue

        base_merged = json.loads(base_merged_path.read_text(encoding="utf-8"))
        stepb = json.loads(stepb_path.read_text(encoding="utf-8"))
        segments = stepb.get("segments") or []

        pages = base_merged.get("pages") or []
        for pg in pages:
            p = int(pg.get("page", 0))
            tag = ""
            for seg in segments:
                a, b = int(seg.get("start", 0)), int(seg.get("end", 0))
                if a <= p <= b:
                    tag = str(seg.get("intent") or "")
                    break
            pg["intent_tag"] = tag

        merged = {
            "book_id": base_merged.get("book_id") or book_id,
            "intent_segments": segments,
            "pages": pages,
            "meta": {
                "synthetic_merged_for_stepB_ablation": True,
                "stepB": {"path": str(stepb_path)},
                "emotion_pages_reused_from": str(base_merged_path),
            },
        }
        (merged_root / f"{book_id}_MOSAIC_merged.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True, choices=list(EXPERIMENTS.keys()))

    # 这些参数可由 shell 包装脚本传入。
    ap.add_argument("--full-mosaic-root", default=str(BASE_FULL_MOSAIC_ROOT))
    ap.add_argument("--model-stepA", default="gpt-4o")
    ap.add_argument("--model-stepB", default="gpt-4o")
    ap.add_argument("--model-stepC", default="gpt-4o")

    ap.add_argument("--books-dir", default=BOOKS_DIR_DEFAULT)
    ap.add_argument("--book-glob", default=BOOK_GLOB_DEFAULT)
    ap.add_argument("--batch-limit", type=int, default=0)

    ap.add_argument("--window-size-stepC", type=int, default=8, help="仅 A3 fixed-window 使用")
    ap.add_argument("--dump-payload", action="store_true", help="新跑阶段是否写 payload；复用阶段会带上原 payload")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--text-verbosity", default=None)

    # 允许继续向 MOSAIC 入口透传额外参数。
    args, extra = ap.parse_known_args()

    exp = EXPERIMENTS[args.exp_id]

    full_mosaic_root = Path(args.full_mosaic_root)
    if not full_mosaic_root.exists():
        raise RuntimeError(f"full MOSAIC 根目录不存在：{full_mosaic_root}")

    # Derive the output tag from the selected backbone configuration.
    if args.model_stepA == args.model_stepB == args.model_stepC:
        out_tag = args.model_stepA
    else:
        out_tag = f"A_{args.model_stepA}__B_{args.model_stepB}__C_{args.model_stepC}"

    out_root = REPO_ROOT / "results" / "ablation" / exp["name"] / out_tag
    out_root.mkdir(parents=True, exist_ok=True)

    books = resolve_books(Path(args.books_dir), args.book_glob, args.batch_limit)
    if not books:
        raise RuntimeError("没有匹配到任何 books。")
    book_ids = [p.stem for p in books]

    # 1) 先把可复用结果 materialize 进来
    materialize_reuse(full_mosaic_root, out_root, book_ids, exp["reuse"])

    # 2) 写计划文件，方便核查当前实验配置
    plan = {
        "exp_id": args.exp_id,
        "exp_name": exp["name"],
        "base_full_root": str(full_mosaic_root),
        "out_root": str(out_root),
        "book_glob": args.book_glob,
        "batch_limit": args.batch_limit,
        "book_ids": book_ids,
        "reuse_steps": exp["reuse"],
        "run_from_step": exp["run_from_step"],
        "run_flags": exp["run_flags"],
        "dump_payload": bool(args.dump_payload),
        "window_size_stepC": args.window_size_stepC,
        "model_stepA": args.model_stepA,
        "model_stepB": args.model_stepB,
        "model_stepC": args.model_stepC,
    }
    (out_root / "ablation_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 3) 若该实验完全复用（A5），直接结束
    if exp["run_from_step"] is None:
        print(f"✅ {args.exp_id} 仅复用已有完整结果，无需模型调用：{out_root}")
        return

    # 4) 计算当前实验是否会真正运行 StepC
    will_run_stepC = ("--skip-stepC-run" not in exp["run_flags"])


    # Shared Step C settings for the ablation variants.
    stepC_base_flags = [
        "--no-include-prompt-head-stepC",
        "--no-anchor-images-stepC",
        "--no-anchor-s1-cues-stepC",
        "--no-inject-s1-candidates-stepC",
    ]
    if will_run_stepC and args.exp_id in {"C3", "C4", "A4"}:
        stepC_base_flags.append("--no-include-anchor-hint-stepC")

    # 6) 调用 MOSAIC 入口运行需要的新阶段
    cmd = [
        sys.executable, "-m", "scripts.inference.run_mosaic",
        "--books-dir", args.books_dir,
        "--book-glob", args.book_glob,
        "--batch-limit", str(args.batch_limit),
        "--out-root", str(out_root),
        "--from-step", exp["run_from_step"],

        "--model-stepA", args.model_stepA,
        "--model-stepB", args.model_stepB,
        "--model-stepC", args.model_stepC,

        "--allow-missing-images-stepA",
        "--image-detail-stepA", "auto",
        "--max-image-side-stepA", "512",

        "--image-detail-stepB", "auto",
        "--max-image-side-stepB", "512",
        "--max-retries-stepB", "2",

        "--allow-missing-images-stepC",
        "--image-detail-stepC", "auto",
        "--max-image-side-stepC", "512",
        "--max-retries-stepC", "2",

        # Final Step B input configuration.
        "--inject-page-text-stepB",
        "--inline-stepA-per-page-stepB",
    ] + stepC_base_flags

    if args.reuse_existing:
        cmd.append("--reuse-existing")
    if args.force:
        cmd.append("--force")
    if args.reasoning_effort:
        cmd += ["--reasoning-effort", args.reasoning_effort]
    if args.text_verbosity:
        cmd += ["--text-verbosity", args.text_verbosity]

    # 7) payload：只给“真正会新跑的阶段”开
    if args.dump_payload:
        if exp["run_from_step"] == "stepA" and "--skip-stepA-run" not in exp["run_flags"]:
            cmd.append("--dump-stepA-payload")

        if exp["run_from_step"] in {"stepA", "stepB"} and "--skip-stepB-run" not in exp["run_flags"]:
            cmd.append("--dump-stepB-payload")

        if exp["run_from_step"] in {"stepA", "stepB", "stepC"} and will_run_stepC:
            cmd.append("--dump-stepC-payload")

    # 8) 追加实验专属开关
    cmd += exp["run_flags"]

    # 9) A3 fixed-window 大小
    if args.exp_id == "A3":
        cmd += ["--window-size-stepC", str(args.window_size_stepC)]

    # 10) 继续透传额外参数
    cmd += extra

    print("🟢 RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 11) B2-B5：不跑 StepC，但合成 merged，方便 compare 继续 post/eval
    if exp["need_synthetic_merged"]:
        synthesize_stepB_only_merged(full_mosaic_root, out_root, book_ids)
        print(f"✅ 已为 {args.exp_id} 合成 merged（供 compare 使用）：{out_root / 'merged'}")

if __name__ == "__main__":
    main()
