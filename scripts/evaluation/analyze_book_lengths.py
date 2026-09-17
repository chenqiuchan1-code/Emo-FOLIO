# -*- coding: utf-8 -*-
"""
analyze_book_lengths.py

用途：
统计绘本 JSON 的页数分布，辅助设定长短书分割阈值。

默认读取 `data/books` 中的全部书目，并将结果写入
`results/book_length_analysis`。

运行示例：
python -m scripts.evaluation.analyze_book_lengths
python -m scripts.evaluation.analyze_book_lengths --book-glob 'book_1-book_34'
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


# ---------- 书目解析：兼容编号区间、通配符和逗号分隔列表 ----------
def _split_specs(book_glob: str):
    if not book_glob:
        return []
    s = book_glob.replace("，", ",").replace("、", ",").replace(";", ",").replace(" ", ",")
    return [p.strip() for p in s.split(",") if p.strip()]


def _expand_specs(books_dir: Path, spec: str):
    spec = spec.strip()

    # 通配
    if any(ch in spec for ch in "*?[]"):
        return sorted(books_dir.glob(spec))

    # 区间：book_1-book_34
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
                for name in names:
                    p = books_dir / name
                    if p.exists() and p not in seen:
                        out.append(p)
                        seen.add(p)
                        break

            return sorted(out, key=lambda x: int(re.findall(r"(\d+)", x.stem)[-1]))

    # 单本：book_1 / book_1.json
    tried = [spec] if spec.lower().endswith(".json") else [spec, f"{spec}.json", f"{spec}.JSON"]
    for name in tried:
        p = books_dir / name
        if p.exists():
            return [p]

    return []


def resolve_books(books_dir: Path, book_glob: str):
    files = []
    seen = set()
    for spec in _split_specs(book_glob):
        for p in _expand_specs(books_dir, spec):
            if p not in seen:
                files.append(p)
                seen.add(p)

    def sort_key(p: Path):
        nums = re.findall(r"(\d+)", p.stem)
        return int(nums[-1]) if nums else p.stem

    return sorted(files, key=sort_key)


# ---------- 页数统计 ----------
def count_pages(book_path: Path) -> int:
    """
    支持两种结构：
    1. {"pages": [...]}
    2. [...]
    若 page 字段存在，则取最大 page 编号；
    否则取 pages 列表长度。
    """
    try:
        data = json.loads(book_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ 读取失败：{book_path} | {e}")
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

    if page_nums:
        return max(page_nums)
    return len(pages)


def percentile(values, q):
    """
    简单 percentile，不依赖 numpy。
    q: 0-100
    """
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def make_threshold_candidates(records):
    """
    对每一个可能页数阈值 t，计算：
    Short: page_count <= t
    Long:  page_count > t
    """
    counts = [r["page_count"] for r in records]
    candidates = []
    for t in sorted(set(counts)):
        short_n = sum(1 for c in counts if c <= t)
        long_n = sum(1 for c in counts if c > t)
        total = len(counts)
        if total == 0:
            continue
        candidates.append({
            "threshold": t,
            "short_n": short_n,
            "long_n": long_n,
            "short_ratio": short_n / total,
            "long_ratio": long_n / total,
            "balance_gap": abs(short_n - long_n),
        })
    return candidates


# ---------- 可视化 ----------
def plot_histogram(counts, out_path: Path, median_value: float):
    plt.figure(figsize=(8, 5))
    bins = range(min(counts), max(counts) + 2)
    plt.hist(counts, bins=bins, edgecolor="black")
    plt.axvline(median_value, linestyle="--", linewidth=1.5, label=f"Median = {median_value:.1f}")
    plt.xlabel("Page count")
    plt.ylabel("Number of books")
    plt.title("Distribution of Book Lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sorted_curve(records, out_path: Path):
    sorted_counts = sorted([r["page_count"] for r in records])

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(sorted_counts) + 1), sorted_counts, marker="o", markersize=3)
    plt.xlabel("Books sorted by page count")
    plt.ylabel("Page count")
    plt.title("Sorted Book Length Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_threshold_balance(candidates, out_path: Path):
    thresholds = [c["threshold"] for c in candidates]
    short_ns = [c["short_n"] for c in candidates]
    long_ns = [c["long_n"] for c in candidates]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, short_ns, marker="o", markersize=3, label="Short Books")
    plt.plot(thresholds, long_ns, marker="o", markersize=3, label="Long Books")
    plt.xlabel("Length threshold: Short <= threshold, Long > threshold")
    plt.ylabel("Number of books")
    plt.title("Short/Long Split under Different Thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------- 主流程 ----------
def main():
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-dir", default=str(repo_root / "data" / "books"), help="绘本 JSON 所在目录")
    ap.add_argument("--book-glob", default="book_*.json", help="书目范围，如 book_1-book_34 / book_*.json")
    ap.add_argument("--out-dir", default=str(repo_root / "results" / "book_length_analysis"), help="输出目录")
    args = ap.parse_args()

    books_dir = Path(args.books_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    book_files = resolve_books(books_dir, args.book_glob)
    if not book_files:
        raise RuntimeError(f"没有找到任何书：books_dir={books_dir}, book_glob={args.book_glob}")

    records = []
    for p in book_files:
        n = count_pages(p)
        records.append({
            "book_id": p.stem,
            "file": str(p),
            "page_count": n,
        })

    records = [r for r in records if r["page_count"] > 0]
    counts = [r["page_count"] for r in records]

    if not counts:
        raise RuntimeError("没有统计到有效页数。")

    summary = {
        "book_glob": args.book_glob,
        "num_books": len(records),
        "min": min(counts),
        "max": max(counts),
        "mean": mean(counts),
        "median": median(counts),
        "p25": percentile(counts, 25),
        "p50": percentile(counts, 50),
        "p75": percentile(counts, 75),
        "p90": percentile(counts, 90),
    }

    candidates = make_threshold_candidates(records)
    candidates_sorted = sorted(candidates, key=lambda x: (x["balance_gap"], x["threshold"]))

    # 写每本书页数
    counts_csv = out_dir / "book_page_counts.csv"
    with counts_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["book_id", "page_count", "file"])
        for r in sorted(records, key=lambda x: x["page_count"]):
            w.writerow([r["book_id"], r["page_count"], r["file"]])

    # 写阈值候选
    cand_csv = out_dir / "threshold_candidates.csv"
    with cand_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "short_n", "long_n", "short_ratio", "long_ratio", "balance_gap"])
        for c in candidates:
            w.writerow([
                c["threshold"],
                c["short_n"],
                c["long_n"],
                f"{c['short_ratio']:.4f}",
                f"{c['long_ratio']:.4f}",
                c["balance_gap"],
            ])

    # 写 summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 画图
    plot_histogram(counts, out_dir / "page_count_histogram.png", summary["median"])
    plot_sorted_curve(records, out_dir / "page_count_sorted_curve.png")
    plot_threshold_balance(candidates, out_dir / "threshold_balance_curve.png")

    # 控制台输出
    print("\n📚 Book length summary")
    print(f"Books:  {summary['num_books']}")
    print(f"Min:    {summary['min']}")
    print(f"Max:    {summary['max']}")
    print(f"Mean:   {summary['mean']:.2f}")
    print(f"Median: {summary['median']:.2f}")
    print(f"P25:    {summary['p25']:.2f}")
    print(f"P75:    {summary['p75']:.2f}")
    print(f"P90:    {summary['p90']:.2f}")

    print("\n🔎 Most balanced threshold candidates")
    for c in candidates_sorted[:10]:
        print(
            f"threshold={c['threshold']:>3} | "
            f"short={c['short_n']:>3} ({c['short_ratio']:.2%}) | "
            f"long={c['long_n']:>3} ({c['long_ratio']:.2%}) | "
            f"gap={c['balance_gap']}"
        )

    print("\n✅ Outputs saved to:")
    print(f"- {counts_csv}")
    print(f"- {cand_csv}")
    print(f"- {summary_path}")
    print(f"- {out_dir / 'page_count_histogram.png'}")
    print(f"- {out_dir / 'page_count_sorted_curve.png'}")
    print(f"- {out_dir / 'threshold_balance_curve.png'}")


if __name__ == "__main__":
    main()
