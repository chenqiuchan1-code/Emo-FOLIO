# -*- coding: utf-8 -*-
"""Convert raw baseline or MOSAIC outputs into the evaluation schema.

The deterministic transformation derives co-primary and secondary emotions,
cross-page intensity trends, normalized stage--intent records, and page-level
intent assignments. The paper-aligned secondary-emotion threshold is 1.

Example:
    python -m scripts.evaluation.postprocess \
      --in_dir results/mosaic/gpt-4o/merged \
      --out_dir results/postprocessed/mosaic-gpt-4o \
      --secondary_threshold 1
"""

import json, argparse, csv, sys
from pathlib import Path

EMOS = ["快乐","悲伤","恐惧","愤怒","惊讶","平静","好奇"]
INTENTS = {"安抚","激励","逗趣","引发好奇","制造紧张","表达悲悯","信息传递","教育启发"}

INTENT_ALIASES = {
    "快乐": "逗趣",
    "开心": "逗趣",
    "高兴": "逗趣",
    "有趣": "逗趣",
    "轻松": "逗趣",
    "平静": "安抚",
    "平和": "安抚",
    "温和": "安抚",
    "安慰": "安抚",
    "安心": "安抚",
}

def normalize_intent_label(x: str) -> str:
    x = str(x or "").strip()
    if not x:
        return ""
    if x in INTENTS:
        return x
    return INTENT_ALIASES.get(x, x)

# ---------- JSON 抽取 ----------

# ---------- JSON 抽取 ----------
def extract_first_json(text: str) -> str:
    """从文本中抽取首个完整 JSON（配对大括号，忽略字符串内的大括号）"""
    in_str = False
    escape = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return text[start:i+1]
    raise ValueError("未能在文本中找到完整的 JSON 对象。")

def load_pred_json(input_path: Path) -> dict:
    txt = input_path.read_text(encoding="utf-8")
    try:
        return json.loads(txt)  # 整段即 JSON
    except Exception:
        jtxt = extract_first_json(txt)  # 抽取第一个 JSON
        return json.loads(jtxt)

# ---------- 合并相邻相同意图的阶段 ----------
def merge_intent_segments(segments, max_page: int):
    """
    输入：原始 intent 段列表（start_page/end_page/intent）
    输出：合并后的段列表（相邻或重叠，且 intent 相同即合并）
         合并条件：cur.intent == last.intent 且 cur.start_page <= last.end_page + 1
         合并后 end_page 取区间最大值；同时裁剪到 [1, max_page]
    """
    if not segments:
        return []
    cleaned = []
    for s in segments:
        try:
            st = int(s.get("start_page") or s.get("start") or 0)
            ed = int(s.get("end_page") or s.get("end") or 0)
            it = normalize_intent_label(s.get("intent", ""))
        except Exception:
            continue
        if not it or st <= 0 or ed <= 0:
            continue
        if st > ed:
            st, ed = ed, st
        st = max(1, st); ed = min(max_page, ed)
        cleaned.append({"start_page": st, "end_page": ed, "intent": it})

    cleaned.sort(key=lambda x: (x["start_page"], x["end_page"]))
    merged = []
    for seg in cleaned:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if (seg["intent"] == last["intent"]) and (seg["start_page"] <= last["end_page"] + 1):
            last["end_page"] = max(last["end_page"], seg["end_page"])
        else:
            merged.append(seg)
    return merged

# ---------- 派生计算 ----------
def derive_candidates_and_secondary(emodict, secondary_threshold: int):
    scores = {k: int(emodict.get(k, 0)) for k in EMOS}
    max_val = max(scores.values()) if scores else 0
    primary_candidates = [k for k in EMOS if scores.get(k, 0) == max_val]
    secs = [(k, scores[k]) for k in EMOS
            if (k not in primary_candidates) and (scores[k] >= secondary_threshold)]
    secs.sort(key=lambda kv: (-kv[1], EMOS.index(kv[0])))
    secondary_list = [k for k, _ in secs]
    return primary_candidates, secondary_list

def compute_prev_trend(prev_emodict, cur_emodict, interest):
    if prev_emodict is None:
        return []
    marks = []
    for emo in EMOS:
        if emo in interest:
            p, c = int(prev_emodict.get(emo, 0)), int(cur_emodict.get(emo, 0))
            if c > p: marks.append(f"{emo}+")
            elif c < p: marks.append(f"{emo}-")
            else: marks.append(f"{emo}=")
    return marks

def expand_intent_tags(intent_segments, total_pages: int):
    tags = [""] * total_pages
    for seg in (intent_segments or []):
        s = int(seg.get("start_page", 0))
        e = int(seg.get("end_page", 0))
        it = normalize_intent_label(seg.get("intent", ""))
        if it not in INTENTS:
            continue
        s = max(1, s); e = min(total_pages, e)
        for pg in range(s, e + 1):
            tags[pg - 1] = it
    return tags

def normalize_one_json(raw: dict, secondary_threshold: int) -> dict:
    pages_in = raw.get("pages", [])
    if not pages_in:
        raise ValueError("输入 JSON 中缺少 'pages' 字段或为空。")
    max_page = max(int(p.get("page", 0)) for p in pages_in)

    # 逐页情感强度（兼容：baseline=emotions；MOSAIC merged=emotion_intensity）
    def _pick_strength_dict(p: dict) -> dict:
        d = p.get("emotions")
        if isinstance(d, dict) and d:
            return d
        d = p.get("emotion_intensity")
        if isinstance(d, dict) and d:
            return d
        # 兜底：允许其它命名
        d = p.get("emotion_intensities") or p.get("emotion_strengths") or {}
        return d if isinstance(d, dict) else {}

    strength_by_page = {
        int(p["page"]): {k: int(_pick_strength_dict(p).get(k, 0)) for k in EMOS}
        for p in pages_in
    }

    for pg in range(1, max_page + 1):
        strength_by_page.setdefault(pg, {k: 0 for k in EMOS})

    # ✅ 合并 intent 段（相邻/重叠且同意图）
    intent_merged = merge_intent_segments(raw.get("intent_segments", []), max_page)

    # 段落展开到逐页（用“合并后”的段”）
    intent_tags = expand_intent_tags(intent_merged, max_page)

    # 逐页派生
    pages_out = []
    prev_emodict = None
    for pg in range(1, max_page + 1):
        emod = strength_by_page[pg]
        primary_candidates, secondary = derive_candidates_and_secondary(emod, secondary_threshold)
        # Derive a direction for every emotion from its intensity difference.
        # Evaluation later selects the gold-anchored eligible categories. This
        # keeps trend estimation independent of predicted page-level labels,
        # as specified by Eq. (B.3) and GA-Trend Acc in the paper.
        prev_marks = compute_prev_trend(prev_emodict, emod, set(EMOS))
        pages_out.append({
            "page": pg,
            "emotions": {k: int(emod.get(k, 0)) for k in EMOS},
            "primary_candidates": primary_candidates,
            "secondary_emotions": secondary,
            "prev_trend_marks": prev_marks,
            "intent_tag": intent_tags[pg - 1] if 1 <= pg <= len(intent_tags) else ""
        })
        prev_emodict = emod

    return {
        "book_id": raw.get("book_id", ""),
        "pages": pages_out,
        # 只保留“已合并”的 intent 段落（不再另存原始段）
        "intent_segments": intent_merged
    }

# ---------- I/O 主流程 ----------
def process_file(in_path: Path, out_dir: Path, secondary_threshold: int) -> None:
    raw = load_pred_json(in_path)
    normalized = normalize_one_json(raw, secondary_threshold)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem  # e.g., book_1_MOSAIC_merged
    out_json = out_dir / f"{stem}_normalized.json"
    out_csv  = out_dir / f"{stem}_report.csv"

    # 写 JSON
    out_json.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    # 写 CSV（逐页对照）
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "page","primary_candidates","secondary_list","prev_trend_marks","intent_tag",
            "快乐","悲伤","恐惧","愤怒","惊讶","平静","好奇"
        ])
        for p in normalized["pages"]:
            w.writerow([
                p["page"],
                " ".join(p.get("primary_candidates", [])),
                " ".join(p.get("secondary_emotions", [])),
                " ".join(p.get("prev_trend_marks", [])),
                p.get("intent_tag", ""),
                p["emotions"]["快乐"], p["emotions"]["悲伤"], p["emotions"]["恐惧"],
                p["emotions"]["愤怒"], p["emotions"]["惊讶"], p["emotions"]["平静"], p["emotions"]["好奇"]
            ])

    print(f"✅ {in_path.name} → JSON: {out_json.name}, CSV: {out_csv.name}")

def main():
    repo_root = Path(__file__).resolve().parents[2]
    # 你可以按需更改默认输入/输出目录
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  default=str(repo_root / "results" / "mosaic"), help="输入目录（含 *_MOSAIC_merged.json/.txt）")
    ap.add_argument("--out_dir", default=str(repo_root / "results" / "postprocessed"), help="输出目录")
    ap.add_argument("--in_file", default="", help="指定单个文件处理（优先级更高）")
    ap.add_argument("--secondary_threshold", type=int, default=1, help="次要情绪强度阈值（默认：1）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.in_file:
        in_path = Path(args.in_file)
        if not in_path.exists():
            print(f"❌ 指定文件不存在：{in_path}", file=sys.stderr)
            sys.exit(1)
        process_file(in_path, out_dir, args.secondary_threshold)
        return

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"❌ 输入目录不存在：{in_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(list(in_dir.glob("*.txt")) + list(in_dir.glob("*.json")))
    if not files:
        print(f"⚠️ 在 {in_dir} 未发现 .txt 或 .json 文件")
        return

    ok, fail = 0, 0
    for fp in files:
        try:
            process_file(fp, out_dir, args.secondary_threshold)
            ok += 1
        except Exception as e:
            print(f"❌ 处理失败：{fp.name}  → {e}", file=sys.stderr)
            fail += 1
    print(f"\n== 完成 ==  成功: {ok}  失败: {fail}  输出目录: {out_dir}")

if __name__ == "__main__":
    main()
