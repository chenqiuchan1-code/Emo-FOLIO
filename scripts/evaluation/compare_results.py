# -*- coding: utf-8 -*-

import json, csv, sys, subprocess, shutil, time, re, os
from pathlib import Path
from datetime import datetime
from typing import List
from scripts.inference.run_ablation import EXPERIMENTS

REPO_ROOT = Path(__file__).resolve().parents[2]

# ===== 参数区（集中配置；其余代码不动） =========================================
# 选书范围
BOOKS_DIR = str(REPO_ROOT / "data" / "books")
BOOK_GLOB = "book_*.json"
BATCH_LIMIT = 0    # 0 表示不限制
NUM_RUNS = 2

# 模型：先定当前要评哪个模型
BASELINE_MODEL = "gpt-4o"
MOSAIC_MODEL_STEPA = "gpt-4o"
MOSAIC_MODEL_STEPB = "gpt-4o"
MOSAIC_MODEL_STEPC = "gpt-4o"
# 常用可写示例：
# - "gpt-4o"
# - "gpt-4o-mini"
# - "gemini-2.5-flash"
# - "gemini-2.5-pro"
# - "qwen3-vl-8b-instruct"
# - "qwen3-vl-32b-instruct"
# - "internvl35-8b"
# - "internvl35-38b"

# Modes included in the main comparison.
DEFAULT_SELECTED_VARIANTS = ["baseline", "cot", "mosaic"]

# MOSAIC 来源：main 读完整方法，ablation 读消融结果
ACTIVE_MOSAIC_SOURCE = {
    "mode": "main",   # "main" / "ablation"
    "exp_id": "C4",     #填消融实验编号，如 "A2" / "C4"
}

# compare 分析模式：
# - normal：常规 baseline / cot / mosaic 对比
# - length：Table 6 长短书分组分析；只做已有结果的 postprocess + evaluate，不重新调用模型
COMPARE_MODE = "normal"

# Table 6 长短书分割线：
# - 页数 <= LENGTH_SPLIT_PAGES：Short Books
# - 页数 >  LENGTH_SPLIT_PAGES：Long Books
LENGTH_SPLIT_PAGES = 17

# 后处理参数：副情感阈值
BASELINE_SECONDARY_THRESHOLD = 1
COT_SECONDARY_THRESHOLD = 1
MOSAIC_SECONDARY_THRESHOLD = 1

# 评估口径
EVAL_ARGS_COMMON = [
    "--trend_equal_tol", "0",        # 相邻页强度差在多少以内算“=”
    "--trend_min_intensity", "0",    # 趋势评估的最小强度门槛
    "--segment_iou_thresh", "0.3",   # 段落匹配的 IoU 阈值
    "--boundary_tol", "0",           # 边界命中的页码容差
]
TREND_EXCLUDE_EQ = True          # 趋势评估时是否排除“=”趋势
TREND_RELAX_NO_OPPOSITE = False  # 是否启用“无相反趋势”宽松判定
TREND_EVAL_MODE = "primary_involved"   # "all" / "primary_only" / "primary_involved"
GOLD_DIR = str(REPO_ROOT / "data" / "annotations")

# compare 默认职责：通常固定
DEFAULT_DO_GEN = False
DEFAULT_DO_POST = True
DEFAULT_DO_EVAL = True
DEFAULT_DO_SUMMARY = True

# 仅 do_gen=True 且 selected_variants 包含 mosaic 时会用到
# When generation is enabled, each trial must repeat all model calls rather
# than reuse predictions from a previous run.
DEFAULT_MOSAIC_FROM_STEP = "auto"   # "auto" / "stepA" / "stepB" / "stepC" / "merge"
DEFAULT_MOSAIC_FORCE = True
DEFAULT_MOSAIC_REUSE_EXISTING = False

# ---- 低频兼容参数：通常不改，但后面逻辑会用到 ----

def make_mosaic_out_tag(stepa: str, stepb: str, stepc: str) -> str:
    if stepa == stepb == stepc:
        return stepa
    return f"A_{stepa}__B_{stepb}__C_{stepc}"

MOSAIC_OUT_TAG = make_mosaic_out_tag(MOSAIC_MODEL_STEPA, MOSAIC_MODEL_STEPB, MOSAIC_MODEL_STEPC)

BASELINE_RAW_RESULTS_DIR = str(REPO_ROOT / "results" / "baselines" / BASELINE_MODEL)
COT_RAW_RESULTS_DIR = str(REPO_ROOT / "results" / "cot" / BASELINE_MODEL)
DEFAULT_MOSAIC_RAW_RESULTS_DIR = str(REPO_ROOT / "results" / "mosaic" / MOSAIC_OUT_TAG)

def resolve_mosaic_raw_results_dir_from_source(source: dict | None = None) -> str:
    src = source or ACTIVE_MOSAIC_SOURCE
    mode = str(src.get("mode", "main") or "main").strip().lower()
    out_tag = str(src.get("out_tag", "") or MOSAIC_OUT_TAG).strip() or MOSAIC_OUT_TAG

    if mode == "main":
        return str(REPO_ROOT / "results" / "mosaic" / out_tag)

    if mode == "ablation":
        exp_id = str(src.get("exp_id", "") or "").strip()
        if not exp_id:
            raise ValueError("ACTIVE_MOSAIC_SOURCE['mode']='ablation' 时必须填写 exp_id。")
        if exp_id not in EXPERIMENTS:
            raise ValueError(f"未知 exp_id: {exp_id}")
        exp_name = EXPERIMENTS[exp_id]["name"]
        return str(REPO_ROOT / "results" / "ablation" / exp_name / out_tag)

    raise ValueError(f"未知 ACTIVE_MOSAIC_SOURCE['mode']: {mode}")

def resolve_effective_mosaic_tag_from_source(source: dict | None = None) -> str:
    src = source or ACTIVE_MOSAIC_SOURCE
    return str(src.get("out_tag", "") or MOSAIC_OUT_TAG).strip() or MOSAIC_OUT_TAG

# 这里先解析一次；main() 里还会按 ACTIVE_MOSAIC_SOURCE 再覆盖
MOSAIC_RAW_RESULTS_DIR = resolve_mosaic_raw_results_dir_from_source()

CONFIG = {
    "analysis": {
        "mode": COMPARE_MODE,
        "length_split_pages": LENGTH_SPLIT_PAGES,
    },

    "default_run": {
        "selected_variants": DEFAULT_SELECTED_VARIANTS,
        "do_gen": DEFAULT_DO_GEN,
        "do_post": DEFAULT_DO_POST,
        "do_eval": DEFAULT_DO_EVAL,
        "do_summary": DEFAULT_DO_SUMMARY,
        "mosaic_from_step": DEFAULT_MOSAIC_FROM_STEP,
        "mosaic_force": DEFAULT_MOSAIC_FORCE,
        "mosaic_reuse_existing": DEFAULT_MOSAIC_REUSE_EXISTING,
    },

    "books_dir": BOOKS_DIR,
    "book_glob": BOOK_GLOB,
    "batch_limit": BATCH_LIMIT,
    "num_runs": NUM_RUNS,

    "variants": {
        "baseline": {
            "kind": "baseline_like",
            "display_name": "Full-book E2E",
            "module": "scripts.inference.run_baselines",
            "raw_results_dir": BASELINE_RAW_RESULTS_DIR,
            "gen_args": [
                "--out", BASELINE_RAW_RESULTS_DIR,
                "--model", BASELINE_MODEL,
                "--trace-tokens",
                "--image-detail", "auto",
                "--max-image-side", "512",
            ],
            "post_in_dir_subpath": ".",
            "post_out_dir_name": "postprocess_results",
            "secondary_threshold": BASELINE_SECONDARY_THRESHOLD,
        },

        "cot": {
            "kind": "baseline_like",
            "display_name": "CoT",
            "module": "scripts.inference.run_baselines",
            "raw_results_dir": COT_RAW_RESULTS_DIR,
            "gen_args": [
                "--out", COT_RAW_RESULTS_DIR,
                "--model", BASELINE_MODEL,
                "--cot",
                "--dump-payload",
                "--trace-tokens",
                "--image-detail", "auto",
                "--max-image-side", "512",
            ],
            "post_in_dir_subpath": ".",
            "post_out_dir_name": "postprocess_results",
            "secondary_threshold": COT_SECONDARY_THRESHOLD,
        },

        "mosaic": {
            "kind": "mosaic",
            "display_name": "MOSAIC",
            "module": "scripts.inference.run_mosaic",
            "raw_results_dir": MOSAIC_RAW_RESULTS_DIR,

            "inject_page_text_stepB": True,
            "inject_page_images_stepB": False,
            "inline_stepA_per_page_stepB": True,
            "emit_missing_image_placeholders_stepB": False,

            "inject_page_text_stepC": True,
            "inject_page_images_stepC": True,

            "inject_book_summary_stepC": False,
            "inject_s1_summary_stepC": False,
            "inject_s1_candidates_stepC": False,
            "inject_s1_text_cues_stepC": True,
            "inject_s1_visual_cues_stepC": True,
            "include_prompt_head_stepC": False,

            "chunk_mode_stepC": "stepb_windows",
            "include_window_intent_stepC": False,
            "include_anchor_hint_stepC": True,
            "no_anchor_images_stepC": True,
            "no_anchor_s1_cues_stepC": True,
            "emit_missing_image_placeholders_stepC": False,

            "trace_tokens": False,
            "reasoning_effort": None,
            "text_verbosity": None,

            "gen_args": [
                "--model-stepA", MOSAIC_MODEL_STEPA,
                "--allow-missing-images-stepA",
                "--dump-stepA-payload",
                "--image-detail-stepA", "auto",
                "--max-image-side-stepA", "512",

                "--model-stepB", MOSAIC_MODEL_STEPB,
                "--dump-stepB-payload",
                "--max-retries-stepB", "2",
                "--image-detail-stepB", "auto",
                "--max-image-side-stepB", "512",

                "--model-stepC", MOSAIC_MODEL_STEPC,
                "--allow-missing-images-stepC",
                "--dump-stepC-payload",
                "--max-retries-stepC", "2",
                "--image-detail-stepC", "auto",
                "--max-image-side-stepC", "512",

                "--out-root", MOSAIC_RAW_RESULTS_DIR,
            ],

            "post_in_dir_subpath": "merged",
            "post_out_dir_name": "postprocess_results",
            "secondary_threshold": MOSAIC_SECONDARY_THRESHOLD,
        },
    },

    "postprocess_module": "scripts.evaluation.postprocess",

    "evaluate_module": "scripts.evaluation.evaluate",
    "eval": {
        "gold_dir": GOLD_DIR,
        "args_common": EVAL_ARGS_COMMON,
        "trend_exclude_eq": TREND_EXCLUDE_EQ,
        "trend_relax_no_opposite": TREND_RELAX_NO_OPPOSITE,
        "trend_eval_mode": TREND_EVAL_MODE,
    }
}

VALID_VARIANTS = ("baseline", "cot", "mosaic")

# ===== 参数区结束 =============================================================

# 读取 runtime_guards 写入的全局熔断文件（若存在则中止）
STOP_ALL_FILE = Path("STOP_ALL")
STOP_REASON_FILE = Path("STOP_ALL_REASON.txt")

# ---------- 选书解析：与 baseline 和 MOSAIC 入口的口径一致 ----------
def _split_specs(spec: str) -> List[str]:
    if not spec:
        return []
    s = spec.replace("，", ",").replace("、", ",").replace(";", ",").replace(" ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or [spec.strip()]

def _expand_specs(books_dir: Path, spec: str) -> List[Path]:
    """
    单规格展开：
      - 含 * ? [] → 直接 glob
      - 含 '-'  → 区间（如 'book_1-book_4'）：
          先拆左右端点→提取数字→按公共前缀枚举（含前导0/.json/.JSON）；
          若仍为空则兜底：扫描目录，按“文件名最后一个数字”在区间内筛选。
      - 其他    → 单文件名；若无后缀也尝试 .json/.JSON
    """
    spec = spec.strip()

    # 通配
    if any(ch in spec for ch in "*?[]"):
        return sorted(books_dir.glob(spec))

    # 区间
    if "-" in spec:
        left, right = spec.rsplit("-", 1)
        left = left.strip(); right = right.strip()
        mL = re.search(r"(\d+)$", left)
        mR = re.search(r"(\d+)$", right)
        if mL and mR:
            s_num, e_num = mL.group(1), mR.group(1)
            a, b = int(s_num), int(e_num)
            if a > b: a, b = b, a
            prefixL = left[:mL.start(1)]
            prefixR = right[:mR.start(1)]
            common_prefix = os.path.commonprefix([prefixL, prefixR]) if prefixL != prefixR else prefixL
            width = max(len(s_num), len(e_num))

            def candidates(i: int):
                base_plain = f"{common_prefix}{i}"
                base_pad = f"{common_prefix}{i:0{width}d}"
                names = [f"{base_plain}.json", f"{base_plain}.JSON", f"{base_pad}.json", f"{base_pad}.JSON"]
                return [books_dir / n for n in names]

            out, seen = [], set()
            for i in range(a, b + 1):
                for p in candidates(i):
                    if p.exists() and p not in seen:
                        out.append(p); seen.add(p)
                        break
            if out:
                return sorted(out, key=lambda x: int(re.findall(r"(\d+)", x.stem)[-1]))

            # 兜底：目录扫描
            cands = []
            for p in list(books_dir.glob("*.json")) + list(books_dir.glob("*.JSON")):
                nums = re.findall(r"(\d+)", p.stem)
                if not nums: continue
                idx = int(nums[-1])
                if a <= idx <= b:
                    cands.append(p)
            if cands:
                return sorted(cands, key=lambda x: int(re.findall(r"(\d+)", x.stem)[-1]))

    # 单文件名：尝试不带后缀/.json/.JSON
    tried = [spec] if spec.lower().endswith(".json") else [spec, f"{spec}.json", f"{spec}.JSON"]
    for name in tried:
        p = books_dir / name
        if p.exists():
            return [p]
    return []

def _resolve_all_books(books_dir: Path, book_glob: str, batch_limit: int) -> List[Path]:
    specs = _split_specs(book_glob)
    files: List[Path] = []
    seen = set()
    for sp in specs:
        for p in _expand_specs(books_dir, sp):
            if p not in seen:
                files.append(p); seen.add(p)
    files = sorted(files, key=lambda x: x.name)
    if batch_limit > 0:
        files = files[:batch_limit]
    return files

def _parse_variants(spec: str) -> list[str]:
    raw = re.split(r"[,\s，、]+", str(spec or "").strip())
    out = []
    seen = set()
    for x in raw:
        v = x.strip().lower()
        if not v:
            continue
        if v not in VALID_VARIANTS:
            raise ValueError(f"Unsupported variant: {v} (allowed: {', '.join(VALID_VARIANTS)})")
        if v not in seen:
            out.append(v)
            seen.add(v)
    if not out:
        raise ValueError("No valid variants specified.")
    return out


def _variant_cfg(name: str) -> dict:
    return CONFIG["variants"][name]


def _variant_scan_dir(name: str) -> Path:
    cfg = _variant_cfg(name)
    root = Path(cfg["raw_results_dir"]).resolve()
    sub = cfg.get("post_in_dir_subpath", ".")
    return root if sub in ("", ".") else root / sub

def _variant_raw_patterns(name: str) -> list[str]:
    """
    不同 variant 只识别各自的主结果文件，避免 txt/json 混扫导致同一本书重复后处理。
    - baseline / cot: baseline 入口的原始主输出，通常是 *_WHOLE.txt
    - mosaic: 只使用 merged 目录下的 *_MOSAIC_merged.json
    """
    if name in ("baseline", "cot"):
        return ["*_WHOLE.txt", "*_WHOLE.TXT"]
    if name == "mosaic":
        return ["*_MOSAIC_merged.json", "*_MOSAIC_merged.JSON"]
    return ["*.json", "*.JSON"]

def _filter_effective_books_by_variants(
    selected: set[str],
    present_by_variant: dict[str, set[str]],
    gold_present: set[str],
    use_fail_fast_for_mosaic: bool = False,
) -> tuple[set[str], dict]:
    pred_common = set(selected)
    for _, present in present_by_variant.items():
        pred_common &= set(present)

    effective = pred_common & set(gold_present)

    fail_fast_dropped = []
    if use_fail_fast_for_mosaic:
        keep = []
        for b in sorted(effective):
            failed, why = _is_book_failed(b)
            if failed:
                fail_fast_dropped.append((b, why))
            else:
                keep.append(b)
        effective = set(keep)

    diag = {
        "selected": sorted(selected),
        "gold_present": sorted(gold_present),
        "present_by_variant": {
            v: sorted(present_by_variant.get(v, set()))
            for v in present_by_variant
        },
        "missing_by_variant": {
            v: sorted(set(selected) - set(present_by_variant.get(v, set())))
            for v in present_by_variant
        },
        "dropped_by_missing_gold": sorted(pred_common - set(gold_present)),
        "fail_fast_dropped": fail_fast_dropped,
        "effective": sorted(effective),
    }
    return effective, diag


def _sanitize_tag(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^\w\-.]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _make_summary_filename(
    selected_variants: list[str],
    baseline_model: str,
    mosaic_tag: str,
    effective_books_n: int,
    exp_id: str = "",
) -> str:
    variants_tag = "-".join(selected_variants)

    baseline_tag = _sanitize_tag(baseline_model)
    mosaic_tag_clean = _sanitize_tag(mosaic_tag)
    exp_tag = _sanitize_tag(exp_id)

    # 单模式
    if len(selected_variants) == 1:
        v = selected_variants[0]
        if v in ("baseline", "cot"):
            return f"summary_{v}_{baseline_tag}_n{effective_books_n}.csv"
        if v == "mosaic":
            if exp_tag:
                return f"summary_mosaic_{exp_tag}_{mosaic_tag_clean}_n{effective_books_n}.csv"
            return f"summary_mosaic_{mosaic_tag_clean}_n{effective_books_n}.csv"

    # 多模式
    if "mosaic" in selected_variants:
        if exp_tag:
            return f"compare_{variants_tag}_{exp_tag}_{baseline_tag}__{mosaic_tag_clean}_n{effective_books_n}.csv"
        return f"compare_{variants_tag}_{baseline_tag}__{mosaic_tag_clean}_n{effective_books_n}.csv"

    return f"compare_{variants_tag}_{baseline_tag}_n{effective_books_n}.csv"


def _write_csv_metadata_block(
    f,
    *,
    selected_variants: list[str],
    baseline_model: str,
    mosaic_tag: str,
    book_glob: str,
    effective_books_n: int,
    exp_id: str = "",
):
    meta_rows = [
        ["# variants", ",".join(selected_variants)],
        ["# baseline_model", baseline_model],
        ["# mosaic_tag", mosaic_tag if "mosaic" in selected_variants else ""],
        ["# exp_id", exp_id if "mosaic" in selected_variants else ""],
        ["# book_glob", book_glob],
        ["# effective_books_n", effective_books_n],
        [],
    ]
    w = csv.writer(f)
    w.writerows(meta_rows)


def _write_summary_tables(
    metrics_runs_by_variant: dict[str, list[dict]],
    selected_variants: list[str],
    summary_dir: Path,
    *,
    baseline_model: str,
    mosaic_tag: str,
    book_glob: str,
    effective_books_n: int,
    exp_id: str = "",
):
    avg_by_variant = {}
    for v in selected_variants:
        runs = metrics_runs_by_variant.get(v, [])
        if runs:
            avg_by_variant[v] = _avg_metrics(runs)

    if not avg_by_variant:
        return None

    out_name = _make_summary_filename(
        selected_variants=selected_variants,
        baseline_model=baseline_model,
        mosaic_tag=mosaic_tag,
        effective_books_n=effective_books_n,
        exp_id=exp_id,
    )
    out_csv = summary_dir / out_name

    # 单模式：输出单列表
    if len(avg_by_variant) == 1:
        v = next(iter(avg_by_variant.keys()))
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            _write_csv_metadata_block(
                f,
                selected_variants=selected_variants,
                baseline_model=baseline_model,
                mosaic_tag=mosaic_tag,
                book_glob=book_glob,
                effective_books_n=effective_books_n,
                exp_id=exp_id,
            )
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k in sorted(avg_by_variant[v].keys()):
                w.writerow([k, avg_by_variant[v][k]])
        return out_csv

    # 多模式：输出宽表；以第一个 variant 为 anchor，给其它模式附 delta
    anchor = next(v for v in selected_variants if v in avg_by_variant)
    others = [v for v in selected_variants if v in avg_by_variant and v != anchor]

    keys = sorted(set().union(*[m.keys() for m in avg_by_variant.values()]))
    header = ["metric", f"{anchor}_avg"]
    for v in others:
        header += [f"{v}_avg", f"delta({v}-{anchor})"]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        _write_csv_metadata_block(
            f,
            selected_variants=selected_variants,
            baseline_model=baseline_model,
            mosaic_tag=mosaic_tag,
            book_glob=book_glob,
            effective_books_n=effective_books_n,
            exp_id=exp_id,
        )
        w = csv.writer(f)
        w.writerow(header)
        for k in keys:
            row = [k, avg_by_variant[anchor].get(k, "")]
            anchor_val = avg_by_variant[anchor].get(k, "")
            for v in others:
                val = avg_by_variant[v].get(k, "")
                delta = ""
                if isinstance(anchor_val, (int, float)) and isinstance(val, (int, float)):
                    delta = val - anchor_val
                row += [val, delta]
            w.writerow(row)

    return out_csv

    # 单模式：输出单列表
    if len(avg_by_variant) == 1:
        v = next(iter(avg_by_variant.keys()))
        out_csv = summary_dir / f"summary_{v}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k in sorted(avg_by_variant[v].keys()):
                w.writerow([k, avg_by_variant[v][k]])
        return out_csv

    # 多模式：输出宽表；以第一个 variant 为 anchor，给其它模式附 delta
    anchor = next(v for v in selected_variants if v in avg_by_variant)
    others = [v for v in selected_variants if v in avg_by_variant and v != anchor]

    keys = sorted(set().union(*[m.keys() for m in avg_by_variant.values()]))
    out_csv = summary_dir / "compare_summary.csv"

    header = ["metric", f"{anchor}_avg"]
    for v in others:
        header += [f"{v}_avg", f"delta({v}-{anchor})"]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in keys:
            row = [k, avg_by_variant[anchor].get(k, "")]
            anchor_val = avg_by_variant[anchor].get(k, "")
            for v in others:
                val = avg_by_variant[v].get(k, "")
                delta = ""
                if isinstance(anchor_val, (int, float)) and isinstance(val, (int, float)):
                    delta = val - anchor_val
                row += [val, delta]
            w.writerow(row)

    return out_csv

def run_cmd(args, log_file: Path):
    """
    运行命令：同时写日志文件 & 实时把子进程输出打印到控制台。
    失败时抛异常并指出日志路径。
    """
    import os, subprocess, sys

    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"🟢 RUN: {' '.join(args)}")
    print(f"📝 LOG: {log_file}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        p.wait()

        if p.returncode != 0:
            raise RuntimeError(
                f"Command failed ({p.returncode}): {' '.join(args)}\nSee log: {log_file}"
            )

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _dir_has_any_json_or_txt(d: Path) -> bool:
    if not d.exists():
        return False
    return any(d.glob("*.json")) or any(d.glob("*.JSON")) or any(d.glob("*.txt"))


def _materialize_subset_files(src_dir: Path, dst_dir: Path, stems: set[str], patterns: list[str]) -> Path:
    """
    从 src_dir 里把属于 stems 的文件（按 patterns 匹配）“实体化”到 dst_dir：
    - 优先使用 symlink（快、节省空间）
    - symlink 失败则 copy2
    返回 dst_dir。
    """
    import shutil, os, re
    safe_mkdir(dst_dir)

    def _link_or_copy(src: Path, dst: Path):
        if dst.exists():
            return
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    copied = 0
    for pat in patterns:
        for f in src_dir.glob(pat):
            m = re.match(r"^(book_[^_]+)", f.stem)
            if not m:
                continue
            bid = m.group(1)
            if bid in stems:
                _link_or_copy(f, dst_dir / f.name)
                copied += 1
    # print(f"📦 subset materialized: {copied} files -> {dst_dir}")
    return dst_dir


def _is_book_failed(book_id: str) -> tuple[bool, str]:
    """
    读取 results/.runtime 下由各阶段写出的 *_STATUS.json，
    任一阶段标记 hard_fail=True，就认为该书本轮应跳过评测。
    """
    tmp = Path("results/.runtime")
    for s in ("STEPA", "STEPB", "STEPC", "MERGE"):
        p = tmp / f"{book_id}_{s}_STATUS.json"
        if p.exists():
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
                if st.get("hard_fail"):
                    return True, st.get("reason", f"{s}_hard_fail")
            except Exception:
                pass
    return False, ""


def _make_gold_subset_dir(gold_root: Path, pred_dir: Path, subset_dir: Path) -> Path:
    """
    根据预测目录中的文件名抽取书名集合（如 book_2），
    在 subset_dir 下只复制这些书对应的 *_annotated.json。
    返回 subset_dir 作为新的 gold_dir。
    """
    import re, shutil
    subset_dir.mkdir(parents=True, exist_ok=True)

    # 从预测文件名中提取 book_* 前缀（兼容如 book_2_MOSAIC_merged.json / book_2_baseline.json 等）
    books = set()
    for p in pred_dir.glob("*.json"):
        m = re.match(r"^(book_[^_]+)", p.stem)  # e.g., book_2_MOSAIC_merged -> book_2
        if m:
            books.add(m.group(1))
    # —— 过滤 Fail-Fast 书目：若任一 *_STATUS.json 标记 hard_fail，就从评测子集中剔除
    _filtered = []
    for b in sorted(books):
        failed, why = _is_book_failed(b)
        if failed:
            print(f"🚫 跳过评测：{b}（{why}）")
        else:
            _filtered.append(b)
    books = set(_filtered)

    # 复制金标子集
    for b in sorted(books):
        src = gold_root / f"{b}_annotated.json"
        if src.exists():
            shutil.copy(src, subset_dir / src.name)

    return subset_dir


def dump_failure_snapshot(snapshot_dir: Path, config: dict, all_books: list):
    """仅在子流程失败时调用，保存最小快照，方便复现。"""
    try:
        (snapshot_dir / "config_snapshot.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (snapshot_dir / "books_snapshot.json").write_text(
            json.dumps([p.name for p in all_books], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _read_eval_metrics(eval_out_dir: Path) -> dict:
    """
    从 evaluate 输出目录读取 overall_summary.csv，提取 __AVERAGE__ 行作为指标 dict。
    若不存在则返回 {}。
    """
    overall_csv = eval_out_dir / "overall_summary.csv"
    if not overall_csv.exists():
        return {}

    with overall_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}

    header = rows[0]
    # 找 __AVERAGE__ 行（一般在最后一行）
    avg_row = None
    for r in rows[1:]:
        if r and r[0].strip() == "__AVERAGE__":
            avg_row = r
            break
    if avg_row is None:
        # 兜底：取最后一行
        avg_row = rows[-1]

    out = {}
    for k, v in zip(header, avg_row):
        if k in ("gold_file", "pred_file"):
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out



def _avg_metrics(metrics_list: list[dict]) -> dict:
    """
    对多次 trial 的 metrics 做均值（仅对 float 项）。
    """
    if not metrics_list:
        return {}
    keys = set().union(*[m.keys() for m in metrics_list])
    out = {}
    for k in sorted(keys):
        vals = [m.get(k) for m in metrics_list if isinstance(m.get(k), (int, float))]
        if vals:
            out[k] = sum(vals) / len(vals)
        else:
            # 保留非数值字段（取首个）
            for m in metrics_list:
                if k in m:
                    out[k] = m[k]
                    break
    return out

# ---------- Table 6：长短书分组分析 ----------

TABLE6_METRICS = [
    ("primary_acc", "Primary Acc"),
    ("ml_micro_f1", "ML F1"),
    ("segments_micro_f1", "Segment F1"),
    ("boundary_f1", "Boundary F1"),
    ("intent_acc", "Intent Acc"),
]


def _safe_float(v):
    return v if isinstance(v, (int, float)) else ""


def _fmt_metric(v):
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return ""


def _count_pages_in_book_json(book_path: Path) -> int:
    """
    统计绘本页数。
    优先使用 pages 列表；若 page 字段存在，则取最大 page 编号，兼容缺页或非连续页码。
    """
    try:
        data = json.loads(book_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    pages = data.get("pages", []) if isinstance(data, dict) else data
    if not isinstance(pages, list):
        return 0

    page_nums = []
    for i, p in enumerate(pages, start=1):
        if isinstance(p, dict):
            try:
                page_nums.append(int(p.get("page", i)))
            except Exception:
                page_nums.append(i)
        else:
            page_nums.append(i)

    return max(page_nums) if page_nums else len(pages)


def _split_books_by_length(book_files: list[Path], split_pages: int):
    """
    Short Books: page_count <= split_pages
    Long Books:  page_count >  split_pages
    """
    groups = {"Short Books": [], "Long Books": []}
    page_count_map = {}

    for p in book_files:
        n = _count_pages_in_book_json(p)
        page_count_map[p.stem] = n
        if n <= split_pages:
            groups["Short Books"].append(p)
        else:
            groups["Long Books"].append(p)

    return groups, page_count_map


def _write_table6_length_summary(
    group_results: dict,
    summary_dir: Path,
    *,
    model_name: str,
    mosaic_tag: str,
    book_glob: str,
    split_pages: int,
):
    """
    输出两个 block：
    1) Paper table block：Short / Long 内部比较 Full-book E2E、MOSAIC 及其差值
    2) Auxiliary block：Long-Short 差异，供正文分析使用，不一定放入论文表格
    """
    out_csv = summary_dir / f"table6_length_{_sanitize_tag(model_name)}__{_sanitize_tag(mosaic_tag)}_cut{split_pages}.csv"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)

        w.writerow(["# table", "Table 6 length-group analysis"])
        w.writerow(["# model", model_name])
        w.writerow(["# mosaic_tag", mosaic_tag])
        w.writerow(["# book_glob", book_glob])
        w.writerow(["# split_rule", f"Short Books <= {split_pages}; Long Books > {split_pages}"])
        w.writerow([])

        # ---------- Block 1：论文主表 ----------
        w.writerow(["## Paper table: within-group MOSAIC gains"])
        header = ["Model", "Length Group", "N", "Mode"] + [label for _, label in TABLE6_METRICS]
        w.writerow(header)

        for group_name in ["Short Books", "Long Books"]:
            g = group_results.get(group_name, {})
            n = g.get("n", 0)
            e2e = g.get("baseline", {})
            mosaic = g.get("mosaic", {})

            rows = {
                "Full-book E2E": [model_name, group_name, n, "Full-book E2E"],
                "MOSAIC": [model_name, group_name, n, "MOSAIC"],
                "Gain": [model_name, group_name, n, "Gain"],
            }

            for key, _label in TABLE6_METRICS:
                ev = _safe_float(e2e.get(key, ""))
                hv = _safe_float(mosaic.get(key, ""))

                rows["Full-book E2E"].append(_fmt_metric(ev))
                rows["MOSAIC"].append(_fmt_metric(hv))

                if isinstance(ev, (int, float)) and isinstance(hv, (int, float)):
                    rows["Gain"].append(_fmt_metric(hv - ev))
                else:
                    rows["Gain"].append("")

            w.writerow(rows["Full-book E2E"])
            w.writerow(rows["MOSAIC"])
            w.writerow(rows["Gain"])

        w.writerow([])

        # ---------- Block 2：辅助分析，不一定放论文表格 ----------
        w.writerow(["## Auxiliary: long-short differences by mode"])
        aux_header = ["Model", "Mode", "Short N", "Long N"] + [f"Long-Short {label}" for _, label in TABLE6_METRICS]
        w.writerow(aux_header)

        short = group_results.get("Short Books", {})
        long = group_results.get("Long Books", {})
        short_n = short.get("n", 0)
        long_n = long.get("n", 0)

        for variant, display_name in [("baseline", "Full-book E2E"), ("mosaic", "MOSAIC")]:
            s_metrics = short.get(variant, {})
            l_metrics = long.get(variant, {})
            row = [model_name, display_name, short_n, long_n]

            for key, _label in TABLE6_METRICS:
                sv = _safe_float(s_metrics.get(key, ""))
                lv = _safe_float(l_metrics.get(key, ""))
                if isinstance(sv, (int, float)) and isinstance(lv, (int, float)):
                    row.append(_fmt_metric(lv - sv))
                else:
                    row.append("")

            w.writerow(row)

    return out_csv


def _run_post_eval_for_effective_subset(
    *,
    selected_variants: list[str],
    effective: set[str],
    subset_dir: Path,
    logs_dir: Path,
    log_prefix: str,
):
    """
    对给定 effective book_id 集合运行 postprocess + evaluate。
    只服务 Table 6 length mode，不影响常规 compare 流程。
    """
    post_dirs = {}
    metrics_by_variant = {}

    # ---------- Postprocess ----------
    for variant in selected_variants:
        cfg = _variant_cfg(variant)
        variant_trial_dir = subset_dir / variant
        safe_mkdir(variant_trial_dir)

        raw_src_dir = _variant_scan_dir(variant)
        raw_subset_dir = variant_trial_dir / "raw_subset"
        post_out_dir = variant_trial_dir / cfg["post_out_dir_name"]
        post_dirs[variant] = post_out_dir

        _materialize_subset_files(
            src_dir=raw_src_dir,
            dst_dir=raw_subset_dir,
            stems=effective,
            patterns=_variant_raw_patterns(variant),
        )

        if not _dir_has_any_json_or_txt(raw_subset_dir):
            print(f"⚠️ {log_prefix}/{variant} raw_subset 为空，跳过 postprocess：{raw_subset_dir}")
            continue

        log_file = logs_dir / f"{log_prefix}_{variant}_post.log"
        cmd = [
            sys.executable, "-m", CONFIG["postprocess_module"],
            "--in_dir", str(raw_subset_dir),
            "--out_dir", str(post_out_dir),
            "--secondary_threshold", str(cfg["secondary_threshold"]),
        ]
        run_cmd(cmd, log_file)

    # ---------- Gold subset ----------
    common_gold_subset = subset_dir / "gold_subset_common"
    shutil.rmtree(common_gold_subset, ignore_errors=True)
    common_gold_subset.mkdir(parents=True, exist_ok=True)

    gold_root = Path(CONFIG["eval"]["gold_dir"]).resolve()
    for b in sorted(effective):
        src = gold_root / f"{b}_annotated.json"
        if src.exists():
            shutil.copy(src, common_gold_subset / src.name)

    # ---------- Evaluate ----------
    for variant in selected_variants:
        variant_trial_dir = subset_dir / variant
        pred_dir = post_dirs.get(variant, variant_trial_dir / _variant_cfg(variant)["post_out_dir_name"])

        if not _dir_has_any_json_or_txt(pred_dir):
            print(f"⚠️ {log_prefix}/{variant} pred_dir 为空，跳过 evaluate：{pred_dir}")
            continue

        if not _dir_has_any_json_or_txt(common_gold_subset):
            print(f"⚠️ {log_prefix} gold_subset 为空，跳过 evaluate：{common_gold_subset}")
            continue

        eval_out_dir = variant_trial_dir / "eval_results"
        safe_mkdir(eval_out_dir)

        log_file = logs_dir / f"{log_prefix}_{variant}_eval.log"
        cmd = [
            sys.executable, "-m", CONFIG["evaluate_module"],
            "--pred_dir", str(pred_dir),
            "--gold_dir", str(common_gold_subset),
            "--out_dir", str(eval_out_dir),
        ] + CONFIG["eval"]["args_common"]

        if CONFIG["eval"].get("trend_exclude_eq", True):
            cmd.append("--trend_exclude_eq")
        else:
            cmd.append("--no-trend_exclude_eq")
        if CONFIG["eval"].get("trend_relax_no_opposite", False):
            cmd.append("--trend_relax_no_opposite")
        cmd += ["--trend_eval_mode", CONFIG["eval"].get("trend_eval_mode", "all")]

        run_cmd(cmd, log_file)

        metrics = _read_eval_metrics(eval_out_dir)
        if metrics:
            metrics_by_variant[variant] = metrics

    return metrics_by_variant


def _run_length_analysis(
    *,
    args,
    selected_variants: list[str],
    all_books: list[Path],
    run_root: Path,
    logs_dir: Path,
    summary_dir: Path,
    effective_mosaic_tag: str,
):
    """
    Table 6 专用：按页数阈值划分 short / long，
    分别运行 postprocess + evaluate，并输出 length summary CSV。
    """
    if set(selected_variants) != {"baseline", "mosaic"}:
        raise ValueError("length 模式要求 --variants baseline,mosaic，用于计算 MOSAIC 相对 Full-book E2E 的增益。")

    groups, page_count_map = _split_books_by_length(all_books, args.length_split_pages)

    (run_root / "length_page_counts.json").write_text(
        json.dumps(page_count_map, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    group_plan = {
        "split_pages": args.length_split_pages,
        "split_rule": f"Short Books <= {args.length_split_pages}; Long Books > {args.length_split_pages}",
        "groups": {
            g: {
                "raw_n": len(files),
                "book_ids": [p.stem for p in files],
            }
            for g, files in groups.items()
        }
    }
    (run_root / "length_group_plan.json").write_text(
        json.dumps(group_plan, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    group_results = {}

    for group_name in ["Short Books", "Long Books"]:
        group_files = groups.get(group_name, [])
        print(f"\n==================== LENGTH GROUP: {group_name} ====================")

        subset_dir = run_root / f"length_{group_name.replace(' ', '_')}"
        safe_mkdir(subset_dir)

        selected_books = _book_stems_from_files(group_files)

        gold_root = Path(CONFIG["eval"]["gold_dir"]).resolve()
        gold_present = _books_present_in_dir(
            gold_root,
            patterns=["*_annotated.json", "*_annotated.JSON"]
        )

        present_by_variant = {
            v: _books_present_in_dir(
                _variant_scan_dir(v),
                patterns=_variant_raw_patterns(v)
            )
            for v in selected_variants
        }

        effective, diag = _filter_effective_books_by_variants(
            selected=selected_books,
            present_by_variant=present_by_variant,
            gold_present=gold_present,
            use_fail_fast_for_mosaic=("mosaic" in selected_variants),
        )

        (subset_dir / "effective_books_diag.json").write_text(
            json.dumps(diag, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (subset_dir / "books_selected_for_post_eval.json").write_text(
            json.dumps(sorted(effective), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"📌 {group_name}: raw={len(group_files)}, effective={len(effective)}")

        group_results[group_name] = {"n": len(effective)}

        if not effective:
            continue

        metrics_by_variant = _run_post_eval_for_effective_subset(
            selected_variants=selected_variants,
            effective=effective,
            subset_dir=subset_dir,
            logs_dir=logs_dir,
            log_prefix=group_name.replace(" ", "_"),
        )

        for variant in selected_variants:
            group_results[group_name][variant] = metrics_by_variant.get(variant, {})

    out_csv = _write_table6_length_summary(
        group_results,
        summary_dir,
        model_name=BASELINE_MODEL,
        mosaic_tag=effective_mosaic_tag,
        book_glob=args.book_glob,
        split_pages=args.length_split_pages,
    )

    print(f"\n✅ Table 6 length summary saved: {out_csv}")
    return out_csv

def _ensure_stop_all():
    if STOP_ALL_FILE.exists():
        reason = ""
        if STOP_REASON_FILE.exists():
            try:
                reason = STOP_REASON_FILE.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        raise RuntimeError(f"STOP_ALL triggered. Reason: {reason}")


def _book_stems_from_files(files: list[Path]) -> set[str]:
    """
    从 books json 文件提取 stem（book_16.json -> book_16）。
    """
    stems = set()
    for p in files:
        stems.add(p.stem)
    return stems

def _books_present_in_dir(src_dir: Path, patterns: list[str]) -> set[str]:
    """
    扫描 src_dir，按 patterns 找到文件，提取其 book_id（book_12 这种 stem 前缀）。
    """
    import re
    out = set()
    if not src_dir.exists():
        return out
    for pat in patterns:
        for f in src_dir.glob(pat):
            m = re.match(r"^(book_[^_]+)", f.stem)
            if m:
                out.add(m.group(1))
    return out


def _filter_effective_books(
    selected: set[str],
    baseline_present: set[str],
    mosaic_present: set[str],
    gold_present: set[str],
    use_fail_fast: bool = True,
) -> tuple[set[str], dict]:
    """
    计算最终用于 post/eval 的共同有效集合：
      selected ∩ baseline_present ∩ mosaic_present ∩ gold_present

    返回诊断信息：
      - missing_in_baseline / missing_in_mosaic / missing_in_gold
      - dropped_by_pred_intersection / dropped_by_missing_gold
      - （可选）fail_fast_dropped
      - effective
    """
    pred_common = set(selected) & set(baseline_present) & set(mosaic_present)
    effective = pred_common & set(gold_present)

    fail_fast_dropped = []
    if use_fail_fast:
        keep = []
        for b in sorted(effective):
            failed, why = _is_book_failed(b)
            if failed:
                fail_fast_dropped.append((b, why))
            else:
                keep.append(b)
        effective = set(keep)

    diag = {
        "selected": sorted(selected),
        "baseline_present": sorted(baseline_present),
        "mosaic_present": sorted(mosaic_present),
        "gold_present": sorted(gold_present),
        "missing_in_baseline": sorted(set(selected) - set(baseline_present)),
        "missing_in_mosaic": sorted(set(selected) - set(mosaic_present)),
        "missing_in_gold": sorted(pred_common - set(gold_present)),
        "dropped_by_pred_intersection": sorted(set(selected) - pred_common),
        "dropped_by_missing_gold": sorted(pred_common - set(gold_present)),
        "fail_fast_dropped": fail_fast_dropped,
        "effective": sorted(effective),
    }
    return effective, diag

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="", help="消融实验 ID，如 A2/A3/A4/A5/B2-B5/C2-C4")
    ap.add_argument("--book-glob", default=CONFIG["book_glob"], help="同 CONFIG['book_glob']")
    ap.add_argument("--books-dir", default=CONFIG["books_dir"], help="同 CONFIG['books_dir']")
    ap.add_argument("--batch-limit", type=int, default=CONFIG["batch_limit"], help="同 CONFIG['batch_limit']")
    ap.add_argument("--num-runs", type=int, default=CONFIG["num_runs"], help="重复跑几次求均值")
    ap.add_argument(
        "--compare-mode",
        choices=["normal", "length"],
        default=CONFIG["analysis"]["mode"],
        help="normal=常规对比；length=Table 6 长短书分组分析"
    )
    ap.add_argument(
        "--length-split-pages",
        type=int,
        default=CONFIG["analysis"]["length_split_pages"],
        help="Table 6 长短书分割线：页数 <= 阈值为 Short Books，> 阈值为 Long Books"
    )

    ap.add_argument(
        "--variants",
        default=",".join(CONFIG["default_run"]["selected_variants"]),
        help="要处理的实验变体，可选：baseline / cot / mosaic，支持逗号分隔，如 baseline,cot,mosaic"
    )

    ap.add_argument("--gen", action="store_true", help="运行所选 variants 的生成")
    ap.add_argument("--post", action="store_true", help="运行所选 variants 的 postprocess")
    ap.add_argument("--eval", action="store_true", help="运行所选 variants 的 evaluate")
    ap.add_argument("--summary", action="store_true", help="输出所选 variants 的汇总表")

    ap.add_argument("--mosaic-from-step", choices=["auto", "stepA", "stepB", "stepC", "merge"], default="auto")
    ap.add_argument("--mosaic-force", action="store_true")
    ap.add_argument("--mosaic-no-reuse", action="store_true")

    ap.add_argument("--trend-exclude-eq", action="store_true")
    ap.add_argument("--trend-relax-no-opposite", action="store_true")
    ap.add_argument(
        "--trend-eval-mode",
        choices=["all", "primary_only", "primary_involved"],
        default=CONFIG["eval"]["trend_eval_mode"],
    )
    ap.add_argument("--trend-primary-only", action="store_true")
    ap.add_argument("--trend-primary-involved", action="store_true")

    args = ap.parse_args()

    effective_mosaic_tag = resolve_effective_mosaic_tag_from_source(ACTIVE_MOSAIC_SOURCE)

    # 允许 CLI 用 --exp-id 临时覆盖参数区里的 ACTIVE_MOSAIC_SOURCE
    # 但不再维护第二套 ablation 路径映射表
    if args.exp_id.strip():
        exp_id = args.exp_id.strip()
        if exp_id not in EXPERIMENTS:
            raise ValueError(f"未知 exp-id: {exp_id}")

        ACTIVE_MOSAIC_SOURCE["mode"] = "ablation"
        ACTIVE_MOSAIC_SOURCE["exp_id"] = exp_id

    # 根据 ACTIVE_MOSAIC_SOURCE 统一解析当前要读取的 MOSAIC 目录
    resolved_mosaic_raw_dir = resolve_mosaic_raw_results_dir_from_source(ACTIVE_MOSAIC_SOURCE)
    effective_exp_id = ACTIVE_MOSAIC_SOURCE.get("exp_id", "").strip() if ACTIVE_MOSAIC_SOURCE.get("mode", "main") == "ablation" else ""
    effective_mosaic_tag = resolve_effective_mosaic_tag_from_source(ACTIVE_MOSAIC_SOURCE)

    CONFIG["variants"]["mosaic"]["raw_results_dir"] = resolved_mosaic_raw_dir

    # 若 compare 未来也要拿来做 gen，顺手把 mosaic 的 --out-root 同步到同一路径
    gen_args = CONFIG["variants"]["mosaic"]["gen_args"]
    if "--out-root" in gen_args:
        idx = gen_args.index("--out-root")
        if idx + 1 < len(gen_args):
            gen_args[idx + 1] = resolved_mosaic_raw_dir

    default_run = CONFIG["default_run"].copy()

    default_run = CONFIG["default_run"].copy()
    if args.gen:
        default_run["do_gen"] = True
    if args.post:
        default_run["do_post"] = True
    if args.eval:
        default_run["do_eval"] = True
    if args.summary:
        default_run["do_summary"] = True

    if args.mosaic_from_step:
        default_run["mosaic_from_step"] = args.mosaic_from_step
    if args.mosaic_force:
        default_run["mosaic_force"] = True
    if args.mosaic_no_reuse:
        default_run["mosaic_reuse_existing"] = False

    if args.trend_exclude_eq:
        CONFIG["eval"]["trend_exclude_eq"] = True
    if args.trend_relax_no_opposite:
        CONFIG["eval"]["trend_relax_no_opposite"] = True

    if args.trend_primary_only and args.trend_primary_involved:
        raise ValueError("--trend-primary-only 与 --trend-primary-involved 不能同时使用")

    CONFIG["eval"]["trend_eval_mode"] = args.trend_eval_mode
    if args.trend_primary_only:
        CONFIG["eval"]["trend_eval_mode"] = "primary_only"
    elif args.trend_primary_involved:
        CONFIG["eval"]["trend_eval_mode"] = "primary_involved"

    selected_variants = _parse_variants(args.variants)

    books_dir = Path(args.books_dir)
    all_books = _resolve_all_books(books_dir, args.book_glob, args.batch_limit)
    if not all_books:
        raise RuntimeError(f"No books matched: dir={books_dir}, spec={args.book_glob}")

    if args.exp_id.strip():
        run_root = Path("ab_runs_ablation") / args.exp_id.strip() / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_root = Path("ab_runs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logs_dir = run_root / "logs"
    summary_dir = run_root / "summary"
    safe_mkdir(logs_dir)
    safe_mkdir(summary_dir)

    (run_root / "books_requested.json").write_text(
        json.dumps([p.name for p in all_books], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (run_root / "variants_requested.json").write_text(
        json.dumps(selected_variants, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # ------------ Table 6：长短书分组分析模式 ------------
    if args.compare_mode == "length":
        if default_run.get("do_gen", False):
            raise ValueError("length 模式只基于已有结果运行 postprocess/evaluate，不支持 --gen。")

        _run_length_analysis(
            args=args,
            selected_variants=selected_variants,
            all_books=all_books,
            run_root=run_root,
            logs_dir=logs_dir,
            summary_dir=summary_dir,
            effective_mosaic_tag=effective_mosaic_tag,
        )
        print("\n🎉 Done.")
        return

    num_runs = int(args.num_runs)
    metrics_runs_by_variant = {v: [] for v in selected_variants}

    for trial in range(1, num_runs + 1):
        print(f"\n==================== TRIAL {trial}/{num_runs} ====================")
        trial_dir = run_root / f"trial_{trial}"
        safe_mkdir(trial_dir)

        # ------------ ① Generation ------------
        if default_run["do_gen"]:
            for variant in selected_variants:
                _ensure_stop_all()
                cfg = _variant_cfg(variant)
                log_file = logs_dir / f"{variant}_gen_{trial}.log"

                if cfg["kind"] == "baseline_like":
                    cmd = [
                        sys.executable, "-m", cfg["module"],
                        str(books_dir),
                        "--book-glob", args.book_glob,
                    ] + cfg["gen_args"]
                    run_cmd(cmd, log_file)
                    continue

                if cfg["kind"] == "mosaic":
                    cmd = [
                        sys.executable, "-m", cfg["module"],
                        "--books-dir", str(books_dir),
                        "--book-glob", args.book_glob,
                        "--from-step", default_run["mosaic_from_step"],
                    ]
                    if default_run["mosaic_force"]:
                        cmd.append("--force")
                    if default_run["mosaic_reuse_existing"]:
                        cmd.append("--reuse-existing")

                    cmd += cfg["gen_args"]

                    if cfg.get("inject_page_text_stepB", False):
                        cmd.append("--inject-page-text-stepB")
                    if cfg.get("inject_page_images_stepB", False):
                        cmd.append("--inject-page-images-stepB")
                    if cfg.get("inline_stepA_per_page_stepB", False):
                        cmd.append("--inline-stepA-per-page-stepB")
                    if cfg.get("emit_missing_image_placeholders_stepB", False):
                        cmd.append("--emit-missing-image-placeholders-stepB")

                    if not cfg.get("inject_page_text_stepC", True):
                        cmd.append("--no-inject-page-text-stepC")
                    if not cfg.get("inject_page_images_stepC", True):
                        cmd.append("--no-inject-page-images-stepC")

                    if not cfg.get("inject_book_summary_stepC", True):
                        cmd.append("--no-inject-book-summary-stepC")
                    if not cfg.get("inject_s1_summary_stepC", True):
                        cmd.append("--no-inject-s1-summary-stepC")
                    if not cfg.get("inject_s1_candidates_stepC", True):
                        cmd.append("--no-inject-s1-candidates-stepC")
                    if not cfg.get("inject_s1_text_cues_stepC", True):
                        cmd.append("--no-inject-s1-text-cues-stepC")
                    if not cfg.get("inject_s1_visual_cues_stepC", True):
                        cmd.append("--no-inject-s1-visual-cues-stepC")
                    if not cfg.get("include_prompt_head_stepC", True):
                        cmd.append("--no-include-prompt-head-stepC")

                    cmd += ["--chunk-mode-stepC", str(cfg.get("chunk_mode_stepC", "stepb_windows"))]

                    if cfg.get("include_window_intent_stepC", False):
                        cmd.append("--include-window-intent-stepC")
                    if cfg.get("include_anchor_hint_stepC", False):
                        cmd.append("--include-anchor-hint-stepC")
                    if cfg.get("emit_missing_image_placeholders_stepC", False):
                        cmd.append("--emit-missing-image-placeholders-stepC")
                    if cfg.get("no_anchor_images_stepC", False):
                        cmd.append("--no-anchor-images-stepC")
                    if cfg.get("no_anchor_s1_cues_stepC", False):
                        cmd.append("--no-anchor-s1-cues-stepC")

                    if cfg.get("trace_tokens", False):
                        cmd.append("--trace-tokens")
                    if cfg.get("reasoning_effort", None):
                        cmd += ["--reasoning-effort", str(cfg["reasoning_effort"])]
                    if cfg.get("text_verbosity", None):
                        cmd += ["--text-verbosity", str(cfg["text_verbosity"])]

                    run_cmd(cmd, log_file)
                    continue

                raise ValueError(f"Unknown variant kind: {cfg['kind']}")

        # ------------ ② Effective books ------------
        effective = None
        diag = None
        need_effective = bool(default_run.get("do_post", False) or default_run.get("do_eval", False))

        if need_effective:
            selected_books = _book_stems_from_files(all_books)
            gold_root = Path(CONFIG["eval"]["gold_dir"]).resolve()
            gold_present = _books_present_in_dir(
                gold_root,
                patterns=["*_annotated.json", "*_annotated.JSON"]
            )

            present_by_variant = {
                v: _books_present_in_dir(
                    _variant_scan_dir(v),
                    patterns=_variant_raw_patterns(v)
                )
                for v in selected_variants
            }

            effective, diag = _filter_effective_books_by_variants(
                selected=selected_books,
                present_by_variant=present_by_variant,
                gold_present=gold_present,
                use_fail_fast_for_mosaic=("mosaic" in selected_variants),
            )

            print("\n📌 [COMPARE] 样本对齐（按本次所选 variants 与 gold 求共同有效集合）")
            for v in selected_variants:
                missing = diag["missing_by_variant"].get(v, [])
                if missing:
                    print(f"⚠️ 缺 {v} 结果，过滤：{missing}")
            if diag["dropped_by_missing_gold"]:
                print(f"⚠️ 缺 gold 标注，过滤：{diag['dropped_by_missing_gold']}")
            if diag["fail_fast_dropped"]:
                print("🚫 Fail-fast 过滤：")
                for b, why in diag["fail_fast_dropped"]:
                    print(f"  - {b}: {why}")
            print(f"✅ 最终用于 post/eval 的共同有效书目数：{len(effective)}")

            (trial_dir / "effective_books_diag.json").write_text(
                json.dumps(diag, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            (trial_dir / "books_selected_for_post_eval.json").write_text(
                json.dumps(sorted(effective), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            if not effective:
                print("❌ 本次所选 variants 与 gold 没有共同有效样本，本 trial 跳过 postprocess / evaluate。")
                continue

        # ------------ ③ Postprocess ------------
        post_dirs = {}

        if default_run["do_post"]:
            _ensure_stop_all()

            for variant in selected_variants:
                cfg = _variant_cfg(variant)
                variant_trial_dir = trial_dir / variant
                safe_mkdir(variant_trial_dir)

                raw_src_dir = _variant_scan_dir(variant)
                raw_subset_dir = variant_trial_dir / "raw_subset"
                post_out_dir = variant_trial_dir / cfg["post_out_dir_name"]
                post_dirs[variant] = post_out_dir

                _materialize_subset_files(
                    src_dir=raw_src_dir,
                    dst_dir=raw_subset_dir,
                    stems=effective,
                    patterns=_variant_raw_patterns(variant),
                )

                if not _dir_has_any_json_or_txt(raw_subset_dir):
                    print(f"⚠️ {variant} raw_subset 为空，跳过 postprocess：{raw_subset_dir}")
                    continue

                log_file = logs_dir / f"{variant}_post_{trial}.log"
                cmd = [
                    sys.executable, "-m", CONFIG["postprocess_module"],
                    "--in_dir", str(raw_subset_dir),
                    "--out_dir", str(post_out_dir),
                    "--secondary_threshold", str(cfg["secondary_threshold"]),
                ]
                run_cmd(cmd, log_file)

        # ------------ ④ Evaluate ------------
        if default_run["do_eval"]:
            _ensure_stop_all()

            common_gold_subset = trial_dir / "gold_subset_common"
            shutil.rmtree(common_gold_subset, ignore_errors=True)
            common_gold_subset.mkdir(parents=True, exist_ok=True)

            gold_root = Path(CONFIG["eval"]["gold_dir"]).resolve()
            for b in sorted(effective):
                src = gold_root / f"{b}_annotated.json"
                if src.exists():
                    shutil.copy(src, common_gold_subset / src.name)

            for variant in selected_variants:
                variant_trial_dir = trial_dir / variant
                pred_dir = post_dirs.get(variant, variant_trial_dir / _variant_cfg(variant)["post_out_dir_name"])

                if not _dir_has_any_json_or_txt(pred_dir):
                    print(f"⚠️ {variant} pred_dir 为空，跳过 evaluate：{pred_dir}")
                    continue

                if not _dir_has_any_json_or_txt(common_gold_subset):
                    print(f"⚠️ gold_subset 为空，跳过 {variant} evaluate：{common_gold_subset}")
                    continue

                eval_out_dir = variant_trial_dir / "eval_results"
                safe_mkdir(eval_out_dir)

                log_file = logs_dir / f"{variant}_eval_{trial}.log"
                cmd = [
                    sys.executable, "-m", CONFIG["evaluate_module"],
                    "--pred_dir", str(pred_dir),
                    "--gold_dir", str(common_gold_subset),
                    "--out_dir", str(eval_out_dir),
                ] + CONFIG["eval"]["args_common"]

                if CONFIG["eval"].get("trend_exclude_eq", True):
                    cmd.append("--trend_exclude_eq")
                else:
                    cmd.append("--no-trend_exclude_eq")
                if CONFIG["eval"].get("trend_relax_no_opposite", False):
                    cmd.append("--trend_relax_no_opposite")
                cmd += ["--trend_eval_mode", CONFIG["eval"].get("trend_eval_mode", "all")]

                run_cmd(cmd, log_file)

                metrics = _read_eval_metrics(eval_out_dir)
                if metrics:
                    metrics_runs_by_variant[variant].append(metrics)

    # ------------ ⑤ Summary ------------
    if default_run["do_summary"]:
        # 若本轮没走 post/eval，effective 可能不存在；此时退回到请求书目数
        try:
            effective_books_n = len(effective) if effective is not None else len(all_books)
        except Exception:
            effective_books_n = len(all_books)

        out_csv = _write_summary_tables(
            metrics_runs_by_variant=metrics_runs_by_variant,
            selected_variants=selected_variants,
            summary_dir=summary_dir,
            baseline_model=BASELINE_MODEL,
            mosaic_tag=effective_mosaic_tag,
            book_glob=args.book_glob,
            effective_books_n=effective_books_n,
            exp_id=effective_exp_id,
        )
        if out_csv is not None:
            print(f"\n✅ Summary saved: {out_csv}")

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
