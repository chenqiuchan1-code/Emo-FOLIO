# -*- coding: utf-8 -*-
"""Run the complete MOSAIC pipeline: Step A -> Step B -> Step C -> merge.

For routine use, edit and run ``run_mosaic.sh``. This module remains available
as the lower-level command-line entry point.
"""

import argparse
import json
import shutil
import inspect
import re
import os
from pathlib import Path
from typing import Dict, Any, List
from mosaic.step_a import run_step_a
from mosaic.step_b import run_step_b
from mosaic.step_c import run_step_c
from mosaic.runtime import save_run_status, RunStatus


# --------------------------
# 目录与搬运
# --------------------------
def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    finals = out_root / "finals"
    debug = out_root / "debug"
    stepa = debug / "stepA"
    stepb = debug / "stepB"
    stepc = debug / "stepC"
    merged = out_root / "merged"

    for p in [finals, debug, stepa, stepb, stepc, merged]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "finals": finals,
        "debug": debug,
        "stepA": stepa,
        "stepB": stepb,
        "stepC": stepc,
        "merged": merged,
    }


def sweep_runtime_to_scratch(project_root: Path, scratch_dir: Path):
    src = project_root / "results" / ".runtime"
    if not src.exists():
        return
    for p in src.iterdir():
        dst = scratch_dir / p.name
        try:
            if dst.exists():
                if dst.is_dir(): shutil.rmtree(dst)
                else: dst.unlink()
            shutil.move(str(p), str(dst))
        except Exception:
            pass
    try:
        src.rmdir()
    except Exception:
        pass

def move_by_glob(src_dir: Path, pattern: str, dst_dir: Path):
    for p in src_dir.glob(pattern):
        try:
            dst = dst_dir / p.name
            if dst.exists():
                if dst.is_dir(): shutil.rmtree(dst)
                else: dst.unlink()
            shutil.move(str(p), str(dst))
        except Exception:
            pass


# --------------------------
# Merge step outputs into the evaluation input format.
# --------------------------
def merge_stepB_stepC(stepC_json: Path, stepB_json: Path | None = None) -> Dict[str, Any]:
    """
    StepB 可选：
    - 有 StepB：pages + intent_segments + intent_tag
    - 无 StepB：只保留 pages 情绪，intent_segments 为空，intent_tag 置空
    """
    sb = {}
    if stepB_json and Path(stepB_json).exists():
        sb = json.loads(Path(stepB_json).read_text(encoding="utf-8"))
    sc = json.loads(Path(stepC_json).read_text(encoding="utf-8"))

    segments = sb.get("segments") or []
    scores = (sc.get("scores") or sc.get("page_scores") or [])

    page2emo: Dict[int, Dict[str, Any]] = {}
    for row in scores:
        if not isinstance(row, dict):
            continue
        p = row.get("page")
        em = row.get("emotions")
        if isinstance(p, int) and isinstance(em, dict):
            page2emo[p] = em

    def _intent_for_page(p: int) -> str:
        for s in segments:
            if not isinstance(s, dict):
                continue
            a, b = s.get("start"), s.get("end")
            if isinstance(a, int) and isinstance(b, int) and a <= p <= b:
                return str(s.get("intent") or "")
        return ""

    all_pages = sorted(page2emo.keys())
    pages_out = []
    for p in all_pages:
        pages_out.append(
            {
                "page": p,
                "emotion_intensity": page2emo.get(p, {}),
                "intent_tag": _intent_for_page(p),
            }
        )

    merged = {
        "book_id": sb.get("book_id") or sc.get("book_id") or "",
        "intent_segments": segments,
        "pages": pages_out,
        "meta": {
            "stepB": {"path": str(stepB_json) if stepB_json else "", "used": bool(stepB_json and Path(stepB_json).exists())},
            "stepC": {"path": str(stepC_json)},
        },
    }
    return merged


def save_merged_to_csv(merged: Dict[str, Any], csv_path: Path):
    import csv
    EMOS = ["快乐","悲伤","恐惧","愤怒","惊讶","平静","好奇"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["page","segment_id","intent"] + EMOS + ["confidence"])
        for row in merged["pages"]:
            emo = row.get("emotions", {})
            w.writerow([
                row["page"], row.get("segment_id"), row.get("intent","")
            ] + [emo.get(e, 0) for e in EMOS] + [row.get("confidence", 0)]
            )


# --------------------------
# Pass only keyword arguments supported by the selected step function.
# --------------------------
def call_with_optional_kwargs(func, required_kwargs: dict, optional_kwargs: dict):
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    filtered_optional = {k: v for k, v in optional_kwargs.items() if k in allowed}
    return func(**required_kwargs, **filtered_optional)

# Step artifact paths and execution checks.
def _artifact_paths(book_id: str, dirs: Dict[str, Path]) -> Dict[str, Path]:
    """
    统一产物命名（使用 MOSAIC 标识）：
      StepA -> finals/<book>_STEPA_information.json
      StepB -> finals/<book>_STEPB_segments.json
      StepC -> finals/<book>_STEPC_emotions.json
      Merge -> merged/<book>_MOSAIC_merged.json
    """
    return {
        "stepA": dirs["finals"] / f"{book_id}_STEPA_information.json",
        "stepB": dirs["finals"] / f"{book_id}_STEPB_segments.json",
        "stepC": dirs["finals"] / f"{book_id}_STEPC_emotions.json",
        "merge": dirs["merged"] / f"{book_id}_MOSAIC_merged.json",
    }

def _should_run(step: str, from_step: str, force: bool, reuse: bool, artifact: Path) -> bool:
    """
    决定某一步是否需要运行：
      - order 顺序：[step1, step2, hints, step3, merge]
      - 若 force=True：从 from_step 开始以及其后的步均运行（覆盖）
      - 若 force=False 且 reuse=True：若目标 artifact 已存在则跳过该步
      - 若 step 在 from_step 之前则跳过（不回跑前序）
    """
    order = ["stepA", "stepB", "stepC", "merge"]
    if from_step not in order and from_step != "auto":
        # 容错：若传入未知 from_step（不大可能），回退到 auto（即按 artifact 存在判断）
        from_step = "auto"

    if force:
        # 若强制，则只要 step 在 from_step 的后面或相等就跑
        if from_step == "auto":
            # force + auto 的语义等同于强制跑所有步
            return True
        return order.index(step) >= order.index(from_step)

    # 非强制
    if from_step != "auto":
        # 若用户明确指定从某步开始，前序跳过
        if order.index(step) < order.index(from_step):
            return False

    # 如果允许复用且产物已存在，则跳过
    if reuse and artifact.exists():
        return False

    # 默认跑（产物缺失或不复用）
    return True



# --------------------------
# Book selection: wildcard, range, or explicit list.
# --------------------------
_SPLIT_SEP_RE = re.compile(r"[,\s，、]+")

def _split_specs(book_glob: str) -> List[str]:
    """
    将 book_glob 拆成若干“规格”：
    - 若包含逗号/空格/中文分隔符，则按分隔切分多个规格
    - 否则保留原样作为单一规格（可为通配 * 或范围 book_1-book_4 或单名 book_3.json）
    """
    s = (book_glob or "").strip()
    if not s:
        return []
    parts = [p for p in _SPLIT_SEP_RE.split(s) if p]
    return parts if len(parts) > 1 else [s]

def _expand_specs(books_dir: Path, spec: str) -> List[Path]:
    """
    单规格展开：
      - 含 * ? [] → 直接 glob
      - 含 '-'  → 区间（如 'book_1-book_4'）：
          先“拆左右端点→提取数字→枚举构造文件名（含前导0/.json/.JSON）”逐一尝试；
          若仍为空则兜底：扫描目录，按“文件名最后一个数字”在区间内筛选。
      - 其他    → 单文件名；若无后缀也尝试 .json/.JSON
    """
    spec = spec.strip()

    # 1) 通配：* ? []
    if any(ch in spec for ch in "*?[]"):
        return sorted(books_dir.glob(spec))

    # 2) 区间：形如 'xxxA-yyyB' —— 我们按最后一个 '-' 拆分两端
    if "-" in spec:
        left, right = spec.rsplit("-", 1)
        left = left.strip()
        right = right.strip()

        # 允许左右端以 .json/.JSON 结尾，先去掉尾缀再取尾部数字
        left_stripped  = re.sub(r"\.(json|JSON)$", "", left)
        right_stripped = re.sub(r"\.(json|JSON)$", "", right)

        # 提取左右端的“尾部数字串”
        mL = re.search(r"(\d+)$", left_stripped)
        mR = re.search(r"(\d+)$", right_stripped)
        if mL and mR:
            s_num, e_num = mL.group(1), mR.group(1)
            a, b = int(s_num), int(e_num)
            if a > b:
                a, b = b, a

            # 左端“前缀”（把尾部数字去掉）与右端“前缀”
            prefixL = left_stripped[:mL.start(1)]
            prefixR = right_stripped[:mR.start(1)]

            # 如果左右前缀不同，以“较短公共前缀”为主；否则用左前缀
            common_prefix = os.path.commonprefix([prefixL, prefixR]) if prefixL != prefixR else prefixL

            # 以左右端数字宽度的最大值决定零填充宽度（兼容 01/003）
            width = max(len(s_num), len(e_num))

            def candidate_paths(i: int):
                # 同时尝试：不补零/补零 + .json/.JSON
                base_plain = f"{common_prefix}{i}"
                base_pad = f"{common_prefix}{i:0{width}d}"
                names = [
                    f"{base_plain}.json", f"{base_plain}.JSON",
                    f"{base_pad}.json",  f"{base_pad}.JSON",
                ]
                return [books_dir / n for n in names]

            out: List[Path] = []
            seen = set()
            for i in range(a, b + 1):
                for p in candidate_paths(i):
                    if p.exists() and p not in seen:
                        out.append(p); seen.add(p)
                        break  # 命中一种命名即接受该编号

            if out:
                # 最终按“文件名里最后一个数字”的自然序排序
                return sorted(out, key=lambda x: int(re.findall(r"(\d+)", x.stem)[-1]))

            # —— 枚举仍未命中：做兜底扫描 —— #
            candidates = []
            seen2 = set()
            for p in list(books_dir.glob("*.json")) + list(books_dir.glob("*.JSON")):
                if p in seen2:  # 去重
                    continue
                seen2.add(p)
                nums = re.findall(r"(\d+)", p.stem)
                if not nums:
                    continue
                idx = int(nums[-1])
                if a <= idx <= b:
                    candidates.append(p)
            if candidates:
                return sorted(candidates, key=lambda x: int(re.findall(r"(\d+)", x.stem)[-1]))
        # 若左右端无法提取数字，则视为普通文件名处理（落到下面 3) 的分支）

    # 3) 单文件名：尝试不带后缀/.json/.JSON
    tried = [spec] if spec.lower().endswith(".json") else [spec, f"{spec}.json", f"{spec}.JSON"]
    for name in tried:
        p = books_dir / name
        if p.exists():
            return [p]
    return []

def _status_failed(book_id: str, step: str, scratch_dir: Path) -> tuple[bool, str]:
    import json
    candidates = [
        scratch_dir / f"{book_id}_{step}_STATUS.json",
        scratch_dir.parent / f"{book_id}_{step}_STATUS.json",
        Path("results/.runtime") / f"{book_id}_{step}_STATUS.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
                if st.get("hard_fail"):
                    return True, st.get("reason", f"{step}_hard_fail")
            except Exception:
                pass
    return False, ""

# --------------------------
# 主流程（单本）
# --------------------------
def run_one_book(args, book_json: Path, dirs: Dict[str, Path], proj_root: Path):
    """
    Run the MOSAIC pipeline for one book:
    - 支持从中间步骤开始（--from-step）
    - 支持强制覆盖（--force）
    - 支持复用已存在产物（--reuse-existing）
    """
    book_id = book_json.stem
    print(f"▶️ MOSAIC 开始：{book_id}")
    print(f"📁 输出根目录：{dirs['finals'].parent}")

    # 早停检查
    if Path("STOP_ALL").exists():
        print("⛔ 检测到全局 STOP_ALL 文件，跳过本书并停止运行。")
        return

    # 解析 from/force/reuse 三要素
    from_step = getattr(args, "from_step", "auto")
    force = bool(getattr(args, "force", False))
    reuse = bool(getattr(args, "reuse_existing", False))

    # 各步目标 artifact 路径
    arts = _artifact_paths(book_id, dirs)

    # ==== StepA ====
    print("—— Step A ——")
    stepA_json = arts["stepA"]

    if bool(getattr(args, "skip_stepA_run", False)):
        print("⏭️ 跳过 StepA（实验配置要求）")
        stepA_json = None

    elif _should_run("stepA", from_step, force, reuse, arts["stepA"]):
        if Path("STOP_ALL").exists():
            print("⛔ STOP_ALL 存在，跳过 StepA。")
            return

        call_with_optional_kwargs(
            run_step_a,
            dict(
                book_json=book_json,
                model=args.model_stepA,
                allow_missing=args.allow_missing_images_stepA,
                out_dir=dirs["finals"],
            ),
            dict(
                dump_payload=args.dump_stepA_payload,
                dry_run=args.dry_run_stepA,
                include_visual_style_guide=(not args.no_visual_style_guide_stepA),
                image_detail_stepA=args.image_detail_stepA,
                max_image_side_stepA=args.max_image_side_stepA,
                trace_tokens=args.trace_tokens,
                reasoning_effort=args.reasoning_effort,
                text_verbosity=args.text_verbosity,
            ),
        )
        if args.dry_run_stepA:
            move_by_glob(dirs["finals"], f"{book_id}_STEPA_payload_preview.json", dirs["stepA"])
            print("🧪 Step A dry run complete; later steps were not executed.")
            return
        failed, why = _status_failed(book_id, "STEPA", dirs["finals"])
        if failed:
            print(f"⛔ 本书早停：StepA 硬失败（{why}），跳过后续步骤。")
            return
        move_by_glob(dirs["finals"], f"{book_id}_STEPA_payload_preview.json", dirs["stepA"])
        move_by_glob(dirs["finals"], f"{book_id}_STEPA_raw.txt", dirs["stepA"])

    else:
        print(f"⏭️ 跳过 StepA（复用现有产物）：{arts['stepA']}")

    # ==== StepB ====
    print("—— Step B ——")
    stepB_json = arts["stepB"]

    if bool(getattr(args, "skip_stepB_run", False)):
        print("⏭️ 跳过 StepB（实验配置要求）")
        stepB_json = None

    elif _should_run("stepB", from_step, force, reuse, arts["stepB"]):
        if Path("STOP_ALL").exists():
            print("⛔ STOP_ALL 存在，跳过 StepB。")
            return

        call_with_optional_kwargs(
            run_step_b,
            dict(
                book_json=book_json,
                stepA_json=(None if bool(getattr(args, "disable_stepA_input_stepB", False)) else stepA_json),
                out_json=arts["stepB"],
                out_txt=(dirs["stepB"] / f"{book_id}_STEPB_raw.txt"),
                model=args.model_stepB,
                dump_payload=args.dump_stepB_payload,
                max_retries=args.max_retries_stepB,
                allow_missing_images=True,
                image_detail=args.image_detail_stepB,
                max_image_side=args.max_image_side_stepB,
                inject_page_text=args.inject_page_text_stepB,
                inject_page_images=args.inject_page_images_stepB,
                inline_stepA_per_page=args.inline_stepA_per_page_stepB,
                emit_missing_image_placeholders=args.emit_missing_image_placeholders_stepB,
                reasoning_effort=args.reasoning_effort,
                text_verbosity=args.text_verbosity,

                include_book_summary=args.include_book_summary_stepB,
                include_page_summaries=args.include_page_summaries_stepB,
                include_emotion_candidates=args.include_emotion_candidates_stepB,
                include_text_cues=args.include_text_cues_stepB,
                include_visual_cues=args.include_visual_cues_stepB,
                require_rationale_output=args.require_rationale_output_stepB,
            ),
            dict(),
        )

        failed, why = _status_failed(book_id, "STEPB", arts["stepB"].parent)
        if failed:
            print(f"⛔ 本书早停：StepB 硬失败（{why}），跳过后续步骤。")
            return
        move_by_glob(dirs["finals"], f"{book_id}_STEPB_request_preview.json", dirs["stepB"])
        move_by_glob(dirs["finals"], f"{book_id}_STEPB_raw.txt", dirs["stepB"])

    else:
        print(f"⏭️ 跳过 StepB（复用现有产物）：{arts['stepB']}")

    # ==== StepC ====
    print("—— Step C ——")
    stepC_json = arts["stepC"]

    if bool(getattr(args, "skip_stepC_run", False)):
        print("⏭️ 跳过 StepC（实验配置要求）")
        stepC_json = None

    elif _should_run("stepC", from_step, force, reuse, arts["stepC"]):
        if Path("STOP_ALL").exists():
            print("⛔ STOP_ALL 存在，跳过 StepC。")
            return

        for p in [
            dirs["finals"] / f"{book_id}_STEPC_STATUS.json",
            dirs["finals"].parent / f"{book_id}_STEPC_STATUS.json",
            Path("results/.runtime") / f"{book_id}_STEPC_STATUS.json",
        ]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        call_with_optional_kwargs(
            run_step_c,
            dict(
                book_json=book_json,
                stepA_json=(None if bool(getattr(args, "disable_stepA_input_stepC", False)) else stepA_json),
                stepB_json=(None if bool(getattr(args, "disable_stepB_input_stepC", False)) else stepB_json),
                model=args.model_stepC,
                dump_payload=args.dump_stepC_payload,
                reasoning_effort=args.reasoning_effort,
                text_verbosity=args.text_verbosity,
                max_retries_stepC=args.max_retries_stepC,
                image_detail_stepC=args.image_detail_stepC,
                max_image_side_stepC=args.max_image_side_stepC,
                inject_page_text=args.inject_page_text_stepC,
                inject_page_images=args.inject_page_images_stepC,
                allow_missing_images=args.allow_missing_images_stepC,
                emit_missing_image_placeholders=args.emit_missing_image_placeholders_stepC,
                include_window_intent=args.include_window_intent_stepC,
                include_anchor_images=(not bool(getattr(args, "no_anchor_images_stepC", False))),
                include_anchor_s1_cues=(not bool(getattr(args, "no_anchor_s1_cues_stepC", False))),
                include_anchor_hint=args.include_anchor_hint_stepC,
                inject_book_summary=args.inject_book_summary_stepC,
                inject_s1_summary=args.inject_s1_summary_stepC,
                inject_s1_candidates=args.inject_s1_candidates_stepC,
                inject_s1_text_cues=args.inject_s1_text_cues_stepC,
                inject_s1_visual_cues=args.inject_s1_visual_cues_stepC,
                include_prompt_head=args.include_prompt_head_stepC,
                chunk_mode=args.chunk_mode_stepC,
                window_size_stepC=args.window_size_stepC,
                dry_run=args.dry_run_stepC,
                out_dir=dirs["finals"],
                out_dir_debug=dirs["stepC"],
                reuse_existing_windows=reuse,
            ),
            dict(),
        )

        failed, why = _status_failed(book_id, "STEPC", dirs["finals"])
        if failed:
            print(f"⛔ 本书早停：StepC 硬失败（{why}），跳过合并。")
            return
        move_by_glob(dirs["finals"], f"{book_id}_STEPC_*_payload_preview.json", dirs["stepC"])
        move_by_glob(dirs["finals"], f"{book_id}_STEPC_*_raw.txt", dirs["stepC"])

    else:
        print(f"⏭️ 跳过 StepC（复用现有产物）：{arts['stepC']}")

    # ==== Merge ====
    print("—— Merge Results ——")
    merged_json = arts["merge"]

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return -1.0

    def _should_run_merge(force_flag: bool, merged_path: Path, inputs: list[Path]) -> bool:
        if force_flag:
            return True
        if not merged_path.exists():
            return True
        m_merged = _mtime(merged_path)
        return any(ip and Path(ip).exists() and _mtime(Path(ip)) > m_merged for ip in inputs)

    if stepC_json is None:
        print("⏭️ 跳过 Merge（当前实验未生成 StepC）")
        return

    effective_stepB_json = None
    if stepB_json is not None and (not bool(getattr(args, "disable_stepB_input_stepC", False))):
        effective_stepB_json = stepB_json

    upstreams = [effective_stepB_json, stepC_json]
    if _should_run_merge(force, merged_json, upstreams):
        merged = merge_stepB_stepC(
            stepC_json=stepC_json,
            stepB_json=effective_stepB_json,
        )
        merged_json.parent.mkdir(parents=True, exist_ok=True)
        merged_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Merge 完成：{merged_json}")
    else:
        print(f"⏭️ 跳过 Merge（复用现有产物）：{merged_json}")

# --------------------------
# CLI
# --------------------------
def main():
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Run the MOSAIC pipeline (Step A -> Step B -> Step C -> merge).")

    # 输入
    ap.add_argument("--book", help="单本：books/book_x.json")
    ap.add_argument("--books-dir", default=str(repo_root / "data" / "books"), help="批量：目录（与 --book-glob 搭配）")
    ap.add_argument("--book-glob", default="book_*.json", help="批量：支持通配 *；或范围 'book_1-book_4'；或逗号分隔 'book_1,book_4,book_8'")
    ap.add_argument("--batch-limit", type=int, default=0, help="批量：最多处理 N 本（0=不限制）")

    # 输出
    ap.add_argument("--out-root", default=str(repo_root / "results" / "mosaic"), help="统一输出根目录")
    ap.add_argument("--project-root", default=str(repo_root), help="项目根目录")

    # ---------------- StepA ----------------
    ap.add_argument("--model-stepA", default="gpt-4o")
    ap.add_argument("--allow-missing-images-stepA", action="store_true")
    ap.add_argument("--dump-stepA-payload", action="store_true")
    ap.add_argument("--dry-run-stepA", action="store_true")
    ap.add_argument("--no-visual-style-guide-stepA", action="store_true")
    ap.add_argument("--image-detail-stepA", choices=["auto", "low", "high"], default="auto")
    ap.add_argument("--max-image-side-stepA", type=int, default=512)

    # ---------------- StepB ----------------
    ap.add_argument("--model-stepB", default="gpt-4o")
    ap.add_argument("--dump-stepB-payload", action="store_true")
    ap.add_argument("--max-retries-stepB", type=int, default=5)
    ap.add_argument("--image-detail-stepB", choices=["auto", "low", "high"], default="auto")
    ap.add_argument("--max-image-side-stepB", type=int, default=512)

    # Final Step B configuration injects ordered page text and page-aligned
    # Step A information, without re-injecting page images.
    ap.add_argument("--inject-page-text-stepB", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--inject-page-images-stepB", action="store_true")
    ap.add_argument("--inline-stepA-per-page-stepB", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--emit-missing-image-placeholders-stepB", action="store_true")

    # StepB：StepA 线索内容开关（B2/B3/B4/B5）
    ap.add_argument("--no-book-summary-stepB", action="store_false", dest="include_book_summary_stepB")
    ap.add_argument("--no-page-summaries-stepB", action="store_false", dest="include_page_summaries_stepB")
    ap.add_argument("--no-s1-candidates-stepB", action="store_false", dest="include_emotion_candidates_stepB")
    ap.add_argument("--no-s1-text-cues-stepB", action="store_false", dest="include_text_cues_stepB")
    ap.add_argument("--no-s1-visual-cues-stepB", action="store_false", dest="include_visual_cues_stepB")
    ap.add_argument("--no-rationale-output-stepB", action="store_false", dest="require_rationale_output_stepB")
    ap.set_defaults(
        include_book_summary_stepB=True,
        include_page_summaries_stepB=True,
        include_emotion_candidates_stepB=True,
        include_text_cues_stepB=True,
        include_visual_cues_stepB=True,
        require_rationale_output_stepB=True,
    )

    # 实验级跳过 / 断开
    ap.add_argument("--skip-stepA-run", action="store_true")
    ap.add_argument("--skip-stepB-run", action="store_true")
    ap.add_argument("--skip-stepC-run", action="store_true")
    ap.add_argument("--disable-stepA-input-stepB", action="store_true")
    ap.add_argument("--disable-stepA-input-stepC", action="store_true")
    ap.add_argument("--disable-stepB-input-stepC", action="store_true")


    # ---------------- StepC ----------------
    ap.add_argument("--model-stepC", default="gpt-4o")
    ap.add_argument("--dump-stepC-payload", action="store_true")
    ap.add_argument("--dry-run-stepC", action="store_true")
    ap.add_argument("--max-retries-stepC", type=int, default=3)
    ap.add_argument("--image-detail-stepC", choices=["auto", "low", "high"], default="auto")
    ap.add_argument("--max-image-side-stepC", type=int, default=512)

    # Final Step C configuration uses original page content, text/visual cues,
    # and continuity anchors. Summaries and emotion candidates are excluded.
    ap.add_argument("--inject-page-text-stepC", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--inject-page-images-stepC", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--inject-book-summary-stepC", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--inject-s1-summary-stepC", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--inject-s1-candidates-stepC", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--inject-s1-text-cues-stepC", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--inject-s1-visual-cues-stepC", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-prompt-head-stepC", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--chunk-mode-stepC", choices=["stepb_windows", "whole_book", "fixed_windows"],
                    default="stepb_windows")
    ap.add_argument("--window-size-stepC", type=int, default=8)
    ap.add_argument("--allow-missing-images-stepC", action="store_true")
    ap.add_argument("--emit-missing-image-placeholders-stepC", action="store_true")
    ap.add_argument("--include-window-intent-stepC", action="store_true")
    ap.add_argument("--include-anchor-hint-stepC", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--anchor-images-stepC", action="store_false", dest="no_anchor_images_stepC",
                    help="Inject anchor-page images (disabled in the paper configuration).")
    ap.add_argument("--no-anchor-images-stepC", action="store_true", dest="no_anchor_images_stepC",
                    help="Do not inject anchor-page images.")
    ap.add_argument("--anchor-s1-cues-stepC", action="store_false", dest="no_anchor_s1_cues_stepC",
                    help="Inject Step A information for anchor pages (disabled in the paper configuration).")
    ap.add_argument("--no-anchor-s1-cues-stepC", action="store_true", dest="no_anchor_s1_cues_stepC",
                    help="Do not inject Step A information for anchor pages.")
    ap.set_defaults(no_anchor_images_stepC=True, no_anchor_s1_cues_stepC=True)

    # 通用 tracing / reasoning
    ap.add_argument("--trace-tokens", action="store_true")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--text-verbosity", choices=["low", "medium", "high"], default=None)

    # ---------- 从中间开始 / 强制 / 复用 ----------
    ap.add_argument("--from-step", choices=["auto", "stepA", "stepB", "stepC", "merge"], default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    dirs = ensure_dirs(out_root)
    proj_root = Path(args.project_root)

    # 单本 or 批量
    if args.book:
        run_one_book(args, Path(args.book), dirs, proj_root)
        return

    bdir = Path(args.books_dir) if args.books_dir else None
    if not bdir or not bdir.exists():
        print("❌ 请指定 --book 或 --books-dir")
        return

    # ==== 新：支持多规格 ====
    specs = _split_specs(args.book_glob)  # 可能是 ["book_*.json"] 或 ["book_1-book_4"] 或 ["book_1","book_4","book_8"]
    selected: List[Path] = []
    missing_specs: List[str] = []

    if len(specs) == 1 and any(ch in specs[0] for ch in "*?"):
        # 单一通配：保持原实现
        selected = sorted(bdir.glob(specs[0]))
    else:
        # 范围 / 多选 / 单名
        seen = set()
        for sp in specs:
            paths = _expand_specs(bdir, sp)
            if paths:
                for p in paths:
                    if p not in seen:
                        selected.append(p)
                        seen.add(p)
            else:
                # 记录没展开成功的项（可能文件不存在）
                missing_specs.append(sp)

    if args.batch_limit and args.batch_limit > 0:
        selected = selected[: args.batch_limit]

    if not selected:
        print("❌ 未匹配到任何绘本文件，请检查 --books-dir 与 --book-glob")
        if missing_specs:
            print("（提示：以下规格未找到）", ", ".join(missing_specs))
        return

    # 列出本次将处理的书单
    names = [p.stem for p in selected]
    print(f"📚 本次将处理 {len(selected)} 本：{', '.join(names)}")
    if missing_specs:
        print("ℹ️ 跳过未找到的项：", ", ".join(missing_specs))

    for bj in selected:
        run_one_book(args, bj, dirs, proj_root)


if __name__ == "__main__":
    main()
