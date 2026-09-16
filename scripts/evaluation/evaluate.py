# -*- coding: utf-8 -*-
"""
evaluate_predictions.py  (PURE CANDIDATES MODE, micro-only outputs)
批量评估人工标注与后处理后的模型预测。

口径（与 postprocess_gpt_json.py 的纯 candidates 模式一致）：
- 主情感准确率（tie-aware）：gold 主情感 ∈ primary_candidates 即判对
- 多标签 Micro-F1：预测集合 = primary_candidates ∪ secondary_emotions
- prev_trend_marks：Jaccard（仅 micro 输出）
- Gold-Anchored 趋势：以强度为准（仅 micro 输出）
- intent_tag 准确率（页级）
- intent_segments：Hungarian + IoU 阈值（仅 micro 输出）
- Boundary F1@±k

用法（在仓库根目录）：
  python -m scripts.evaluation.evaluate \
  --trend_equal_tol 0 \
  --trend_min_intensity 1 \
  --segment_iou_thresh 0.3 \
  --prev_trend_exclude_eq \
  --trend_exclude_eq
"""

import json, argparse, csv, sys
from pathlib import Path
from collections import defaultdict

EMOS = ["快乐","悲伤","恐惧","愤怒","惊讶","平静","好奇"]
INTENTS = {"安抚","激励","逗趣","引发好奇","制造紧张","表达悲悯","信息传递","教育启发"}
DIRS = {"+","-","="}

# ---------- 读取 JSON 或从 .txt 中抽取 JSON ----------
def extract_first_json(text: str) -> str:
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
    raise ValueError("未在文本中找到完整 JSON 片段。")

def load_json_any(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except Exception:
        jtxt = extract_first_json(txt)
        return json.loads(jtxt)

# ---------- 规范化 & 字段抽取 ----------
def norm_trend_mark(s: str):
    if s is None:
        return None
    s = str(s).strip()
    s = (s.replace("＋","+").replace("－","-").replace("＝","=").replace(" ", ""))
    if len(s) < 2:
        return None
    emo = s[:-1]
    direc = s[-1]
    if emo in EMOS and direc in DIRS:
        return emo + direc
    return None

def get_gold_page_fields(page: dict):
    """兼容 gold 两种结构：带 labels{} 或平铺字段"""
    labels = page.get("labels", {}) if isinstance(page.get("labels", {}), dict) else page
    primary = labels.get("primary_emotion","")
    secondary = labels.get("secondary_emotions",[]) or []
    trends = labels.get("prev_trend_marks",[]) or []
    intent_tag = labels.get("intent_tag","") or ""
    return primary, list(secondary), list(trends), intent_tag

def get_pred_page_fields(page: dict):
    """
    预测（normalized）标准结构（纯 candidates）：
    - primary_candidates: list[str]
    - secondary_emotions: list[str]
    - prev_trend_marks: list[str]
    - intent_tag: str
    """
    primary_candidates = page.get("primary_candidates", []) or []
    secondary = page.get("secondary_emotions",[]) or []
    trends = page.get("prev_trend_marks",[]) or []
    intent_tag = page.get("intent_tag","") or ""
    return list(primary_candidates), list(secondary), list(trends), intent_tag

def build_page_map(pages_list):
    """pages list -> {page_num: page_dict}"""
    d = {}
    for p in pages_list:
        try:
            pg = int(p.get("page", 0))
            if pg > 0:
                d[pg] = p
        except Exception:
            continue
    return d


def _norm_trend_set(trend_list):
    out = {norm_trend_mark(x) for x in (trend_list or [])}
    out.discard(None)
    return out


TREND_EVAL_MODES = {"all", "primary_only", "primary_involved"}


def _get_gold_adjacent_emotion_context(gold_pages: dict, pg: int):
    if pg <= 1:
        return set(), set(), "", ""
    prev = gold_pages.get(pg - 1, {})
    cur = gold_pages.get(pg, {})
    prev_primary, prev_secs, _, _ = get_gold_page_fields(prev)
    cur_primary, cur_secs, _, _ = get_gold_page_fields(cur)

    prev_all = set(prev_secs)
    curr_all = set(cur_secs)
    if prev_primary:
        prev_all.add(prev_primary)
    if cur_primary:
        curr_all.add(cur_primary)

    return prev_all, curr_all, prev_primary, cur_primary


def _eligible_trend_emotions(gold_pages: dict, pg: int, trend_eval_mode: str = "all"):
    if pg <= 1:
        return set()
    if trend_eval_mode not in TREND_EVAL_MODES:
        raise ValueError(f"Unsupported trend_eval_mode: {trend_eval_mode}")
    if trend_eval_mode == "all":
        return set()

    prev_all, curr_all, prev_primary, cur_primary = _get_gold_adjacent_emotion_context(gold_pages, pg)
    overlap = prev_all & curr_all

    if trend_eval_mode == "primary_only":
        if prev_primary and cur_primary and prev_primary == cur_primary:
            return {cur_primary}
        return set()

    # primary_involved：相邻页有同一情感，且该情感在两页中至少有一页是主情感
    return {
        emo for emo in overlap
        if emo == prev_primary or emo == cur_primary
    }


def _is_trend_page_eligible(gold_pages: dict, pg: int, trend_eval_mode: str = "all") -> bool:
    if pg <= 1:
        return False
    if trend_eval_mode == "all":
        return True
    return bool(_eligible_trend_emotions(gold_pages, pg, trend_eval_mode=trend_eval_mode))


def _gold_trend_set_by_mode(gold_pages: dict, pg: int, trend_eval_mode: str = "all"):
    cur = gold_pages.get(pg, {})
    _, _, g_trends, _ = get_gold_page_fields(cur)
    gset = _norm_trend_set(g_trends)
    if trend_eval_mode == "all":
        return gset
    eligible = _eligible_trend_emotions(gold_pages, pg, trend_eval_mode=trend_eval_mode)
    return {m for m in gset if m[:-1] in eligible}


def _pred_trend_set_by_gold_mode(gold_pages: dict, pred_pages: dict, pg: int, trend_eval_mode: str = "all"):
    pred_page = pred_pages.get(pg, {})
    _, _, p_trends, _ = get_pred_page_fields(pred_page)
    pset = _norm_trend_set(p_trends)
    if trend_eval_mode == "all":
        return pset
    eligible = _eligible_trend_emotions(gold_pages, pg, trend_eval_mode=trend_eval_mode)
    return {m for m in pset if m[:-1] in eligible}


# ---------- 指标：primary（tie-aware）准确率 ----------
def compute_primary_accuracy_tieaware(gold_pages: dict, pred_pages: dict):
    correct = total = 0
    for pg, g in gold_pages.items():
        g_primary, _, _, _ = get_gold_page_fields(g)
        if not g_primary:
            continue
        total += 1
        p = pred_pages.get(pg, {})
        p_cands, _, _, _ = get_pred_page_fields(p)
        if p_cands and (g_primary in p_cands):
            correct += 1
    acc = correct/total if total else 0.0
    return acc, correct, total

# ---------- 指标：多标签 Micro/Macro F1（仅输出 micro） ----------
def multilabel_f1(gold_pages: dict, pred_pages: dict):
    """
    多标签集合：
      - GOLD: {primary} ∪ secondary_emotions
      - PRED: primary_candidates ∪ secondary_emotions   ← 纯 candidates 口径
    返回：micro_f1, macro_f1, (micro_prec, micro_rec), per_label_f1(dict)
           （宏值仅内部计算，不对外输出）
    """
    tp = fp = fn = 0
    per_label = {emo: {"tp":0,"fp":0,"fn":0} for emo in EMOS}

    for pg, g in gold_pages.items():
        g_primary, g_secs, _, _ = get_gold_page_fields(g)
        g_set = set(g_secs)
        if g_primary: g_set.add(g_primary)

        p = pred_pages.get(pg, {})
        p_cands, p_secs, _, _ = get_pred_page_fields(p)
        p_set = set(p_secs) | set(p_cands)

        inter = g_set & p_set
        tp += len(inter)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)

        for emo in EMOS:
            g_has = (emo in g_set)
            p_has = (emo in p_set)
            if g_has and p_has: per_label[emo]["tp"] += 1
            elif (not g_has) and p_has: per_label[emo]["fp"] += 1
            elif g_has and (not p_has): per_label[emo]["fn"] += 1

    def safe_prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp+fp) else 0.0
        rec  = tp / (tp + fn) if (tp+fn) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        return prec, rec, f1

    micro_prec, micro_rec, micro_f1 = safe_prf(tp, fp, fn)

    f1s = []
    per_label_f1 = {}
    for emo, stats in per_label.items():
        _, _, f1 = safe_prf(stats["tp"], stats["fp"], stats["fn"])
        f1s.append(f1); per_label_f1[emo] = f1
    macro_f1 = sum(f1s)/len(f1s) if f1s else 0.0

    return micro_f1, macro_f1, (micro_prec, micro_rec), per_label_f1

def multilabel_scope_stats(gold_pages: dict, pred_pages: dict):
    """
    统计多标签评估所覆盖的数据量：
    - pages: 参与页数（= gold 页数）
    - gold_label_total: gold 侧标签总数（primary ∪ secondary）
    - pred_label_total: pred 侧标签总数（primary_candidates ∪ secondary）
    """
    pages = 0
    gold_label_total = 0
    pred_label_total = 0

    for pg, g in gold_pages.items():
        g_primary, g_secs, _, _ = get_gold_page_fields(g)
        g_set = set(g_secs)
        if g_primary:
            g_set.add(g_primary)

        p = pred_pages.get(pg, {})
        p_cands, p_secs, _, _ = get_pred_page_fields(p)
        p_set = set(p_secs) | set(p_cands)

        pages += 1
        gold_label_total += len(g_set)
        pred_label_total += len(p_set)

    return {
        "pages": pages,
        "gold_label_total": gold_label_total,
        "pred_label_total": pred_label_total,
    }


# ---------- 指标：Gold-Anchored 趋势（仅输出 micro） ----------
def _pred_sign_from_intensity(pred_pages: dict, pg: int, emo: str, equal_tol: int = 0, min_intensity: int = 0):
    if pg <= 1:
        return None
    prev = pred_pages.get(pg-1, {}).get("emotions", {})
    cur  = pred_pages.get(pg,   {}).get("emotions", {})
    a = int(prev.get(emo, 0))
    b = int(cur.get(emo, 0))
    if max(a, b) < min_intensity or min(a, b) < min_intensity:
        return None
    diff = b - a
    if abs(diff) <= equal_tol:
        return "="
    return "+" if diff > 0 else "-"

def trend_gold_anchored_metrics(gold_pages: dict, pred_pages: dict,
                                equal_tol: int = 0, min_intensity: int = 0,
                                relax_no_opposite: bool = False,
                                trend_eval_mode: str = "all"):

    def _gold_trend_dict(trend_list):
        d = {}
        for s in (trend_list or []):
            s = norm_trend_mark(s)
            if s:
                d[s[:-1]] = s[-1]
        return d

    def _is_correct(gdir: str, pdir: str) -> bool:
        if not relax_no_opposite:
            return pdir == gdir
        if gdir == "+":
            return pdir in {"+", "="}
        if gdir == "-":
            return pdir in {"-", "="}
        return pdir == "="

    correct_all = total_all = 0
    correct_all_noeq = total_all_noeq = 0

    emo_stat = {emo: {"c":0,"t":0, "c_noeq":0,"t_noeq":0} for emo in EMOS}
    page_stats = {}

    for pg, g in gold_pages.items():
        if pg <= 1:
            continue
        if not _is_trend_page_eligible(gold_pages, pg, trend_eval_mode=trend_eval_mode):
            continue

        gset = _gold_trend_set_by_mode(gold_pages, pg, trend_eval_mode=trend_eval_mode)
        gmap = _gold_trend_dict(list(gset))
        if not gmap:
            continue

        c = t = c_noeq = t_noeq = 0
        for emo, gdir in gmap.items():
            pdir = _pred_sign_from_intensity(pred_pages, pg, emo, equal_tol=equal_tol, min_intensity=min_intensity)
            if pdir is None:
                continue
            if _is_correct(gdir, pdir):
                c += 1

            t += 1
            if gdir != "=":
                if _is_correct(gdir, pdir):
                    c_noeq += 1
                t_noeq += 1

            emo_stat[emo]["t"] += 1
            if gdir != "=":
                emo_stat[emo]["t_noeq"] += 1
            if _is_correct(gdir, pdir):
                emo_stat[emo]["c"] += 1
                if gdir != "=":
                    emo_stat[emo]["c_noeq"] += 1

        acc = (c/t) if t else None
        acc_noeq = (c_noeq/t_noeq) if t_noeq else None
        page_stats[pg] = (c, t, acc, c_noeq, t_noeq, acc_noeq)

        correct_all += c
        total_all += t
        correct_all_noeq += c_noeq
        total_all_noeq += t_noeq

    micro = (correct_all/total_all) if total_all else 1.0
    micro_noeq = (correct_all_noeq/total_all_noeq) if total_all_noeq else None

    emos = [emo for emo in EMOS if emo_stat[emo]["t"] > 0]
    macro_emo = sum(emo_stat[e]["c"]/emo_stat[e]["t"] for e in emos)/len(emos) if emos else 1.0
    emos_noeq = [emo for emo in EMOS if emo_stat[emo]["t_noeq"] > 0]
    macro_emo_noeq = (sum(emo_stat[e]["c_noeq"]/emo_stat[e]["t_noeq"] for e in emos_noeq)/len(emos_noeq)) if emos_noeq else None

    pages = [pg for pg in page_stats if page_stats[pg][1] > 0]
    macro_page = sum(page_stats[pg][2] for pg in pages) / len(pages) if pages else 1.0
    pages_noeq = [pg for pg in page_stats if page_stats[pg][4] > 0]
    macro_page_noeq = (sum(page_stats[pg][5] for pg in pages_noeq) / len(pages_noeq)) if pages_noeq else None

    return {
        "trend_gold_micro_acc": micro,
        "trend_gold_macro_acc_emotion": macro_emo,
        "trend_gold_macro_acc_page": macro_page,
        "trend_gold_micro_acc_noeq": micro_noeq,
        "trend_gold_macro_acc_emotion_noeq": macro_emo_noeq,
        "trend_gold_macro_acc_page_noeq": macro_page_noeq,
        "trend_gold_per_page": page_stats,
        "trend_gold_eval_pages": len(pages),
        "trend_gold_total_marks": total_all,
        "trend_gold_eval_pages_noeq": len(pages_noeq),
        "trend_gold_total_marks_noeq": total_all_noeq,
    }

# ---------- intent_tag 准确率（页级） ----------
def intent_accuracy(gold_pages: dict, pred_pages: dict):
    correct = total = 0
    for pg, g in gold_pages.items():
        _, _, _, g_intent = get_gold_page_fields(g)
        p = pred_pages.get(pg, {})
        _, _, _, p_intent = get_pred_page_fields(p)
        if not g_intent:
            continue
        total += 1
        if g_intent == p_intent:
            correct += 1
    acc = correct/total if total else 0.0
    return acc, correct, total

# ---------- 段级评估：工具 ----------
def _seg_len(a, b):
    return max(0, int(b) - int(a) + 1)

def _seg_iou(s1, e1, s2, e2):
    s1, e1, s2, e2 = int(s1), int(e1), int(s2), int(e2)
    inter = _seg_len(max(s1, s2), min(e1, e2))
    union = _seg_len(s1, e1) + _seg_len(s2, e2) - inter
    return (inter / union) if union > 0 else 0.0

def _normalize_segments(seg_list):
    out = []
    for seg in seg_list or []:
        try:
            s = int(seg.get("start_page", 0))
            e = int(seg.get("end_page", 0))
            it = str(seg.get("intent","")).strip()
            if s > 0 and e >= s and it:
                out.append((s,e,it))
        except Exception:
            continue
    return out

# ---------- 匈牙利算法（纯 Python，无第三方依赖） ----------
def hungarian_maximize(matrix):
    n = len(matrix)
    m = max(len(row) for row in matrix) if n > 0 else 0
    if n == 0 or m == 0:
        return []
    maxW = 0.0
    for r in matrix:
        for w in r:
            if w is not None and w > maxW:
                maxW = w
    size = max(n, m)
    cost = [[0.0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i < n and j < len(matrix[i]):
                w = matrix[i][j]
                if w is None:
                    cost[i][j] = maxW + 1.0
                else:
                    cost[i][j] = maxW - w
            else:
                cost[i][j] = maxW + 1.0
    u = [0.0]*(size+1)
    v = [0.0]*(size+1)
    p = [0]*(size+1)
    way = [0]*(size+1)
    for i in range(1, size+1):
        p[0] = i
        j0 = 0
        minv = [float('inf')]*(size+1)
        used = [False]*(size+1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, size+1):
                if not used[j]:
                    cur = cost[i0-1][j-1]-u[i0]-v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    match = [0]*(size+1)
    for j in range(1, size+1):
        match[p[j]] = j
    result = []
    for i in range(1, size+1):
        j = match[i]
        if i <= n and j <= m:
            w = matrix[i-1][j-1] if j-1 < len(matrix[i-1]) else None
            if w is not None:
                result.append((i-1, j-1, w))
    return result

# ---------- 段级评估：匈牙利最优匹配 + IoU 阈值（仅输出 micro） ----------
def segments_iou_f1_hungarian(gold_obj: dict, pred_obj: dict, iou_thresh: float = 0.3):
    gold = _normalize_segments(gold_obj.get("intent_segments", []))
    pred = _normalize_segments(pred_obj.get("intent_segments", []))

    gold_by_intent = defaultdict(list)
    pred_by_intent = defaultdict(list)
    for s,e,it in gold: gold_by_intent[it].append((s,e))
    for s,e,it in pred: pred_by_intent[it].append((s,e))

    TP = FP = FN = 0
    per_intent = {}
    detail_rows = []  # [intent, pred_idx, pred_span, gold_idx, gold_span, IoU, matched(>=thresh)]

    all_intents = set(list(gold_by_intent.keys()) + list(pred_by_intent.keys()))
    for intent in sorted(all_intents):
        g_list = gold_by_intent.get(intent, [])
        p_list = pred_by_intent.get(intent, [])

        matrix = []
        for (gs, ge) in g_list:
            row = []
            for (ps, pe) in p_list:
                row.append(_seg_iou(gs, ge, ps, pe))
            matrix.append(row)

        matches = hungarian_maximize(matrix) if (g_list and p_list) else []

        used_g = set(); used_p = set()
        for pi, (ps,pe) in enumerate(p_list):
            gi = None; iou = 0.0
            for (gii, pjj, w) in matches:
                if pjj == pi:
                    gi = gii; iou = w; break
            matched = (gi is not None and iou >= iou_thresh)
            if matched:
                used_g.add(gi); used_p.add(pi)
            detail_rows.append([
                intent, pi, f"[{ps},{pe}]",
                (gi if matched else ""), (f"[{g_list[gi][0]},{g_list[gi][1]}]" if matched else ""),
                round(iou,4), int(matched)
            ])

        tps = sum(1 for (gi,pj,w) in matches if w >= iou_thresh)
        fps = len(p_list) - tps
        fns = len(g_list) - tps

        TP += tps; FP += fps; FN += fns

        prec = tps / (tps + fps) if (tps+fps) else 0.0
        rec  = tps / (tps + fns) if (tps+fns) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        per_intent[intent] = {
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tps, "fp": fps, "fn": fns,
            "gold_n": len(g_list), "pred_n": len(p_list)
        }

    micro_prec = TP / (TP + FP) if (TP+FP) else 0.0
    micro_rec  = TP / (TP + FN) if (TP+FN) else 0.0
    micro_f1   = (2*micro_prec*micro_rec)/(micro_prec+micro_rec) if (micro_prec+micro_rec) else 0.0
    macro_f1   = (sum(v["f1"] for v in per_intent.values())/len(per_intent)) if per_intent else 1.0

    return {
        "segments_micro_precision": micro_prec,
        "segments_micro_recall": micro_rec,
        "segments_micro_f1": micro_f1,
        "segments_macro_f1": macro_f1,
        "segments_per_intent": per_intent,
        "segments_detail_rows": detail_rows,

        # 供可靠性验证使用
        "segments_gold_total": len(gold),
        "segments_pred_total": len(pred),
    }

# ---------- 边界指标：Boundary F1@±k ----------
def _extract_boundaries(segments):
    bounds = set()
    for s,e,it in segments:
        if e is not None:
            bounds.add(int(e))
    return bounds

def boundary_f1(gold_obj: dict, pred_obj: dict, tol: int = 0):
    g_raw = _normalize_segments(gold_obj.get("intent_segments", []))
    p_raw = _normalize_segments(pred_obj.get("intent_segments", []))

    g_bounds = sorted(list(_extract_boundaries(g_raw)))
    p_bounds = sorted(list(_extract_boundaries(p_raw)))

    used_g = set()
    tp = 0
    detail = []  # [pred_idx, pred_bound, matched_gold_idx, gold_bound, hit(±tol)]
    for i, pb in enumerate(p_bounds):
        matched_idx = -1
        best_dist = None
        for j, gb in enumerate(g_bounds):
            if j in used_g:
                continue
            d = abs(pb - gb)
            if d <= tol and (best_dist is None or d < best_dist):
                best_dist = d
                matched_idx = j
        if matched_idx >= 0:
            used_g.add(matched_idx)
            tp += 1
            detail.append([i, pb, matched_idx, g_bounds[matched_idx], 1])
        else:
            detail.append([i, pb, "", "", 0])

    fp = len(p_bounds) - tp
    fn = len(g_bounds) - tp

    prec = tp / (tp + fp) if (tp+fp) else 0.0
    rec  = tp / (tp + fn) if (tp+fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0

    return {
        "boundary_precision": prec,
        "boundary_recall": rec,
        "boundary_f1": f1,
        "boundary_tp": tp, "boundary_fp": fp, "boundary_fn": fn,
        "boundary_tol": tol,
        "boundary_detail_rows": detail
    }

# ---------- 配对工具 ----------
# ---------- 配对工具 ----------
def _normalize_match_stem(stem: str) -> str:
    s = str(stem or "").strip().lower()
    for suf in ["_annotated", "_normalized", "_whole_normalized", "_whole"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def _dedupe_keep_order(items):
    out = []
    seen = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def stem_key_candidates(path: Path, loaded: dict):
    """
    返回“有序去重”的候选 key，不能用 set。
    否则后续配对时遍历顺序会随进程变化，导致同一数据多次评估时结果漂移。
    """
    cands = []
    bid = str(loaded.get("book_id", "")).strip().lower()
    if bid:
        cands.append(bid)

    stem_raw = path.stem.lower()
    stem_norm = _normalize_match_stem(stem_raw)

    cands.append(stem_norm)
    cands.append(stem_raw)

    return _dedupe_keep_order(cands)

def _pred_match_rank(gold_path: Path, pred_path: Path):
    """
    给候选 pred 一个固定排序规则，保证每次都选同一个：
    1) 归一化后的 stem 完全相同优先
    2) 原始 stem 以前缀匹配优先
    3) stem 更短的优先
    4) 文件名字典序兜底
    """
    g_raw = gold_path.stem.lower()
    g_norm = _normalize_match_stem(g_raw)

    p_raw = pred_path.stem.lower()
    p_norm = _normalize_match_stem(p_raw)

    exact_norm = 0 if p_norm == g_norm else 1
    prefix_raw = 0 if p_raw.startswith(g_norm) else 1
    shorter = len(p_raw)

    return (exact_norm, prefix_raw, shorter, pred_path.name.lower())

# ---------- 单对评估 ----------
def evaluate_pair(gold_path: Path, pred_path: Path, equal_tol: int = 0, min_intensity: int = 0,
                  seg_iou_thresh: float = 0.3, boundary_tol: int = 0,
                  prev_trend_exclude_eq: bool = False, trend_exclude_eq: bool = True,
                  trend_relax_no_opposite: bool = False,
                  trend_eval_mode: str = "primary_involved"):
    gold = load_json_any(gold_path)
    pred = load_json_any(pred_path)
    gold_pages = build_page_map(gold.get("pages", []))
    pred_pages = build_page_map(pred.get("pages", []))

    # 1) primary（tie-aware）准确率
    prim_acc, prim_hit, prim_tot = compute_primary_accuracy_tieaware(gold_pages, pred_pages)
    # 2) 多标签 F1（candidates ∪ secondary）
    micro_f1, macro_f1, (micro_prec, micro_rec), per_label_f1 = multilabel_f1(gold_pages, pred_pages)
    ml_scope = multilabel_scope_stats(gold_pages, pred_pages)

    # 3) Gold-Anchored 趋势（以强度为准）
    gold_anchor = trend_gold_anchored_metrics(gold_pages, pred_pages,
                                              equal_tol=equal_tol,
                                              min_intensity=min_intensity,
                                              relax_no_opposite=trend_relax_no_opposite,
                                              trend_eval_mode=trend_eval_mode)
    # 4) intent_tag 准确率
    intent_acc, intent_hit, intent_tot = intent_accuracy(gold_pages, pred_pages)
    # 5) intent_segments（分段）Hungarian IoU/F1
    seg_metrics = segments_iou_f1_hungarian(gold, pred, iou_thresh=seg_iou_thresh)
    # 6) Boundary F1@±k
    bdy_metrics = boundary_f1(gold, pred, tol=boundary_tol)

    # 每页对照（便于排查）
    max_pg = max(gold_pages.keys()) if gold_pages else 0
    per_page_rows = []
    for pg in range(1, max_pg+1):
        g = gold_pages.get(pg, {})
        p = pred_pages.get(pg, {})
        g_primary, g_secs, g_trends, g_intent = get_gold_page_fields(g)
        p_cands, p_secs, p_trends, p_intent = get_pred_page_fields(p)

        # Retain the trend-set display; page-level Jaccard is not reported.
        if pg == 1:
            g_trends_view = []
            p_trends_view = []
        else:
            if _is_trend_page_eligible(gold_pages, pg, trend_eval_mode=trend_eval_mode):
                G = _gold_trend_set_by_mode(gold_pages, pg, trend_eval_mode=trend_eval_mode)
                P = _pred_trend_set_by_gold_mode(gold_pages, pred_pages, pg, trend_eval_mode=trend_eval_mode)
                g_trends_view = sorted(G)
                p_trends_view = sorted(P)
            else:
                g_trends_view = []
                p_trends_view = []

        # Gold-Anchored（页级）
        ga_stats = gold_anchor["trend_gold_per_page"].get(pg, (0,0,None,0,0,None))
        _, _, ga_acc, _, _, ga_acc_noeq = ga_stats

        per_page_rows.append([
            pg,
            g_primary,
            " ".join(p_cands), int(g_primary in p_cands) if g_primary else "",
            " ".join(sorted(set(g_secs))),
            " ".join(sorted(set(set(p_secs) | set(p_cands)))),
            " ".join(g_trends_view),
            " ".join(p_trends_view),
            ga_acc, ga_acc_noeq,
            g_intent, p_intent, int(g_intent == p_intent and g_intent != "")
        ])

    # —— 仅输出 micro 指标到 metrics —— #
    # 根据开关选择 Gold-Anchored 主口径（含“=”或去“=”）
    _trend_gold_main = gold_anchor["trend_gold_micro_acc_noeq"] if (trend_exclude_eq and gold_anchor["trend_gold_micro_acc_noeq"] is not None) else gold_anchor["trend_gold_micro_acc"]

    metrics = {
        "primary_accuracy": prim_acc,  # tie-aware
        "primary_correct": prim_hit,
        "primary_total": prim_tot,

        "multilabel_micro_precision": micro_prec,
        "multilabel_micro_recall": micro_rec,
        "multilabel_micro_f1": micro_f1,
        "per_label_f1": per_label_f1,  # 保留明细，便于排查

        "trend_gold_micro_acc": _trend_gold_main,
        "trend_gold_micro_acc_noeq": gold_anchor["trend_gold_micro_acc_noeq"],

        "intent_tag_accuracy": intent_acc,
        "intent_tag_correct": intent_hit,
        "intent_tag_total": intent_tot,

        "segments_micro_precision": seg_metrics["segments_micro_precision"],
        "segments_micro_recall": seg_metrics["segments_micro_recall"],
        "segments_micro_f1": seg_metrics["segments_micro_f1"],

        "boundary_precision": bdy_metrics["boundary_precision"],
        "boundary_recall": bdy_metrics["boundary_recall"],
        "boundary_f1": bdy_metrics["boundary_f1"],
        "boundary_tol": bdy_metrics["boundary_tol"],

        # ===== 可靠性验证用统计 =====
        "primary_eval_pages": prim_tot,

        "multilabel_eval_pages": ml_scope["pages"],
        "multilabel_gold_labels_total": ml_scope["gold_label_total"],
        "multilabel_pred_labels_total": ml_scope["pred_label_total"],

        "trend_gold_eval_pages": gold_anchor["trend_gold_eval_pages"],
        "trend_gold_total_marks": gold_anchor["trend_gold_total_marks"],
        "trend_gold_eval_pages_noeq": gold_anchor["trend_gold_eval_pages_noeq"],
        "trend_gold_total_marks_noeq": gold_anchor["trend_gold_total_marks_noeq"],

        "segments_gold_total": seg_metrics["segments_gold_total"],
        "segments_pred_total": seg_metrics["segments_pred_total"],
    }
    return metrics, per_page_rows, seg_metrics["segments_detail_rows"], bdy_metrics["boundary_detail_rows"]

# ---------- 主流程 ----------
def main():
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_dir", default=str(repo_root / "data" / "annotations"), help="人工标注目录")
    ap.add_argument("--pred_dir", default=str(repo_root / "results" / "postprocessed"), help="预测结果目录")
    ap.add_argument("--out_dir",  default=str(repo_root / "results" / "evaluation"), help="输出目录")
    ap.add_argument("--trend_equal_tol", type=int, default=0, help="判定“=”的容差 |Δ|≤tol（默认0）")
    ap.add_argument("--trend_min_intensity", type=int, default=0, help="仅当两页强度都≥该值才评估趋势（默认0=不限制）")
    ap.add_argument("--segment_iou_thresh", type=float, default=0.3, help="intent_segments 配对的 IoU 阈值（默认0.3）")
    ap.add_argument("--boundary_tol", type=int, default=0, help="边界命中容差（默认精确匹配）")
    # Control whether unchanged ("=") transitions are excluded.
    ap.add_argument(
        "--trend_exclude_eq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gold-anchored 趋势评估是否过滤稳定（=）趋势（默认过滤）",
    )
    ap.add_argument(
        "--trend_relax_no_opposite",
        action="store_true",
        help="方法二（Gold-anchored）宽松判对：gold为+时pred=+或'='算对；gold为-时pred=-或'='算对；gold为'='仍需pred为'='"
    )
    ap.add_argument(
        "--trend_eval_mode",
        choices=["all", "primary_only", "primary_involved"],
        default="primary_involved",
        help="趋势评估口径：all / primary_only / primary_involved"
    )
    ap.add_argument(
        "--trend_primary_only",
        action="store_true",
        help="兼容旧调用：等价于 --trend_eval_mode primary_only"
    )
    ap.add_argument(
        "--trend_primary_involved",
        action="store_true",
        help="兼容旧调用：等价于 --trend_eval_mode primary_involved"
    )
    args = ap.parse_args()

    if args.trend_primary_only and args.trend_primary_involved:
        raise ValueError("--trend_primary_only 与 --trend_primary_involved 不能同时使用")
    if args.trend_primary_only:
        args.trend_eval_mode = "primary_only"
    elif args.trend_primary_involved:
        args.trend_eval_mode = "primary_involved"

    gold_dir = Path(args.gold_dir); pred_dir = Path(args.pred_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_files = sorted(list(gold_dir.glob("*.json")) + list(gold_dir.glob("*.txt")))
    pred_files = sorted(list(pred_dir.glob("*.json")) + list(pred_dir.glob("*.txt")))
    if not gold_files or not pred_files:
        print(f"❌ 未找到文件：gold={gold_dir}, pred={pred_dir}", file=sys.stderr); sys.exit(1)

    gold_entries = []
    pred_entries = []
    for g in gold_files:
        try:
            gj = load_json_any(g)
        except Exception as e:
            print(f"⚠️ 跳过（无法解析）：{g.name} → {e}"); continue
        gold_entries.append((g, gj, stem_key_candidates(g, gj)))

    for p in pred_files:
        try:
            pj = load_json_any(p)
        except Exception as e:
            print(f"⚠️ 跳过（无法解析）：{p.name} → {e}"); continue
        pred_entries.append((p, pj, stem_key_candidates(p, pj)))

    key2pred = defaultdict(list)
    for p, pj, keys in pred_entries:
        for k in keys:
            key2pred[k].append((p, pj))

    # 每个 key 下的候选 pred 先按文件名字典序固定下来
    for k in list(key2pred.keys()):
        key2pred[k].sort(key=lambda item: item[0].name.lower())

    pairs = []
    used_pred = set()

    for g, gj, gkeys in gold_entries:
        candidate_pool = []
        seen_pred = set()

        # 第一层：按有序 key 精确找候选
        for k in gkeys:
            for cand_p, cand_pj in key2pred.get(k, []):
                if cand_p in used_pred or cand_p in seen_pred:
                    continue
                candidate_pool.append((cand_p, cand_pj))
                seen_pred.add(cand_p)

        # 第二层：兜底前缀匹配，但也要走固定排序
        if not candidate_pool:
            gst = _normalize_match_stem(g.stem.lower())
            for p, pj, _ in pred_entries:
                if p in used_pred or p in seen_pred:
                    continue
                pst = p.stem.lower()
                if pst.startswith(gst):
                    candidate_pool.append((p, pj))
                    seen_pred.add(p)

        matched = None
        if candidate_pool:
            candidate_pool.sort(key=lambda item: _pred_match_rank(g, item[0]))
            matched = candidate_pool[0]

        if matched:
            pairs.append((g, gj, matched[0], matched[1]))
            used_pred.add(matched[0])
        else:
            print(f"⚠️ 未找到匹配的预测文件：{g.name}")

    if not pairs:
        print("❌ 没有形成任何 gold-pred 配对。")
        sys.exit(1)

    overall_rows = []
    all_metric_objs = []
    for idx, (g_path, gj, p_path, pj) in enumerate(pairs, 1):
        print(f"▶️ 评估配对 {idx}/{len(pairs)}：{g_path.name}  vs  {p_path.name}")
        metrics, per_page_rows, seg_detail_rows, bdy_detail_rows = evaluate_pair(
            g_path, p_path,
            equal_tol=args.trend_equal_tol,
            min_intensity=args.trend_min_intensity,
            seg_iou_thresh=args.segment_iou_thresh,
            boundary_tol=args.boundary_tol,
            trend_exclude_eq=args.trend_exclude_eq,
            trend_relax_no_opposite=args.trend_relax_no_opposite,
            trend_eval_mode=args.trend_eval_mode
        )

        stem = g_path.stem
        csv_path = out_dir / f"{stem}_pages_compare.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "page",
                "gold_primary", "pred_primary_candidates", "primary_tie_correct",
                "gold_labels", "pred_labels",
                "gold_prev_trend", "pred_prev_trend",
                "trend_gold_acc", "trend_gold_acc_noeq",
                "gold_intent", "pred_intent", "intent_correct"
            ])
            w.writerows(per_page_rows)

        seg_csv = out_dir / f"{stem}_segments_match.csv"
        with seg_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["intent","pred_idx","pred_span","gold_idx","gold_span","IoU","matched(>=thresh)"])
            w.writerows(seg_detail_rows)

        bdy_csv = out_dir / f"{stem}_boundary_match.csv"
        with bdy_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)  # ← 补上这行
            w.writerow(["pred_idx", "pred_boundary", "gold_idx", "gold_boundary", "hit(±tol)"])
            w.writerows(bdy_detail_rows)

        all_metric_objs.append(metrics)


        summary_path = out_dir / f"{stem}_summary.json"
        summary = {
            "gold_file": g_path.name,
            "pred_file": p_path.name,
            **metrics,
            "segments_iou_threshold": args.segment_iou_thresh,
            "boundary_tol": args.boundary_tol
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        # overall_rows 仅保留 micro 指标（列顺序要与表头一致）
        overall_rows.append([
            g_path.name, p_path.name,
            metrics["primary_accuracy"],  # tie-aware
            metrics["multilabel_micro_precision"],
            metrics["multilabel_micro_recall"],
            metrics["multilabel_micro_f1"],
            metrics["trend_gold_micro_acc"],
            metrics["segments_micro_f1"],
            metrics["boundary_f1"],
            metrics["intent_tag_accuracy"]
        ])

    # ======= 总体汇总 CSV（逐书）=======
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_csv = out_dir / "overall_summary.csv"
    with overall_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "gold_file", "pred_file",
            "primary_acc",  # tie-aware
            "ml_micro_prec", "ml_micro_rec", "ml_micro_f1",
            "trend_gold_micro_acc",
            "segments_micro_f1",
            "boundary_f1",
            "intent_acc"
        ])
        w.writerows(overall_rows)

    # ======= 先把逐书明细转为 list[dict]，用于计算平均 =======
    overall_json = out_dir / "overall_summary.json"
    overall_list = []
    for row in overall_rows:
        overall_list.append({
            "gold_file": row[0],
            "pred_file": row[1],
            "primary_acc": row[2],
            "ml_micro_prec": row[3],
            "ml_micro_rec": row[4],
            "ml_micro_f1": row[5],
            "trend_gold_micro_acc": row[6],
            "segments_micro_f1": row[7],
            "boundary_f1": row[8],
            "intent_acc": row[9],
        })

    # 需要求平均的数值字段（micro-only）
    numeric_fields = [
        "primary_acc",
        "ml_micro_prec", "ml_micro_rec", "ml_micro_f1",
        "trend_gold_micro_acc",
        "segments_micro_f1",
        "boundary_f1",
        "intent_acc"
    ]

    def _safe_mean(values):
        vals = [v for v in values if isinstance(v, (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    # ======= 计算逐书宏平均（macro over books）=======
    avg_entry = {
        "gold_file": "__AVERAGE__",
        "pred_file": "__AVERAGE__",
        "books_count": len(overall_list)
    }
    for k in numeric_fields:
        avg_entry[k] = _safe_mean([d.get(k) for d in overall_list])

    # ======= 写 JSON：逐书 + 平均行 =======
    overall_list.append(avg_entry)
    overall_json.write_text(json.dumps(overall_list, ensure_ascii=False, indent=2), encoding="utf-8")

    # ======= 在 overall_summary.csv 中追加平均结果行 =======
    with overall_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        avg_row = [
            avg_entry.get("gold_file", "__AVERAGE__"),
            avg_entry.get("pred_file", "__AVERAGE__"),
            avg_entry.get("primary_acc"),
            avg_entry.get("ml_micro_prec"),
            avg_entry.get("ml_micro_rec"),
            avg_entry.get("ml_micro_f1"),
            avg_entry.get("trend_gold_micro_acc"),
            avg_entry.get("segments_micro_f1"),
            avg_entry.get("boundary_f1"),
            avg_entry.get("intent_acc")
        ]
        w.writerow(avg_row)

    # ======= 终端打印平均摘要 =======
    print("\n====== 模型整体能力（逐书平均）======")
    print(f"样本书本数: {avg_entry['books_count']}")
    for k in numeric_fields:
        v = avg_entry[k]
        if isinstance(v, float):
            print(f"{k:>24s}: {v:.4f}")
        else:
            print(f"{k:>24s}: {v}")
    print("==================================\n")
    # ======= 终端打印全局样本量统计 =======
    def _sum_metric(key):
        vals = [m.get(key, 0) for m in all_metric_objs if isinstance(m.get(key, 0), (int, float))]
        return sum(vals)

    print("====== 评估样本量统计（全局） ======")
    print(f"参与情感评估的页面总数：{_sum_metric('primary_eval_pages')}")
    print(
        f"人工标注情感标签总数：{_sum_metric('multilabel_gold_labels_total')}，"
        f"模型输出情感标签总数：{_sum_metric('multilabel_pred_labels_total')}"
    )
    print(
        f"人工标注段落总数：{_sum_metric('segments_gold_total')}，"
        f"模型预测段落总数：{_sum_metric('segments_pred_total')}"
    )
    print("==================================\n")

    print(f"\n✅ 评估完成！输出目录：{out_dir}")
    print(" - 每本书明细 CSV：*_pages_compare.csv（含 pred_primary_candidates & tie 命中标识）")
    print(" - 每本书分段对齐 CSV（Hungarian）：*_segments_match.csv")
    print(" - 每本书边界对齐 CSV：*_boundary_match.csv")
    print(" - 每本书摘要 JSON：*_summary.json")
    print(" - 总体汇总：overall_summary.csv（逐书 + 平均行）/ overall_summary.json（逐书 + 平均项）")
    print(" - 可调参数：--trend_equal_tol, --trend_min_intensity, --segment_iou_thresh, --boundary_tol, "
          "--trend_exclude_eq")

if __name__ == "__main__":
    main()
