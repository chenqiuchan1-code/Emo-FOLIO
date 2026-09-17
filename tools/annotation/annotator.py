# -*- coding: utf-8 -*-
"""Streamlit interface for page-, transition-, and stage-level annotation."""

import json
import base64
import io
import re
import html
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

# ===== 常量 =====
EMOTIONS = ["快乐", "悲伤", "恐惧", "愤怒", "惊讶", "平静", "好奇"]
TREND_CHOICES = {"递增(+)": "+", "递减(-)": "-", "不变(=)": "="}
INTENTS  = ["安抚", "激励", "逗趣", "引发好奇", "制造紧张", "表达悲悯", "信息传递", "教育启发"]

st.set_page_config(page_title="绘本情感标注器（优化版）", layout="wide")

# ===== 样式 =====
st.markdown("""
<style>
/* 意图标签一行显示（必要时横向滚动） */
div.intent-row .stRadio > div { flex-wrap: nowrap; overflow-x: auto; }
div.intent-row .stMarkdown { margin-bottom: 0 !important; }

/* 顶部行中按钮更紧凑 */
div.intent-actions .stButton>button { padding: 0.35rem 0.6rem; border-radius: 8px; }

/* 列数按钮统一尺寸 */
div.gallery-controls .stButton>button {
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  line-height: 1;
  min-width: 2.2rem;
}

/* 画廊卡片文本可滚动（固定高度，内部滚动） */
.card-text {
  font-size: 0.85rem; color: #666;
  height: 100px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 2px;
}

/* 分段参考：更小、更灰、更不抢眼 */
.segment-ref-box {
  border: 1px solid #e9e9ee;
  border-radius: 10px;
  background: #f8f9fb;
  padding: 8px 10px;
  font-size: 0.82rem;
  line-height: 1.35;
  color: #8a8f99;
  max-height: 120px;
  overflow-y: auto;
  margin-top: 4px;
  margin-bottom: 8px;
  white-space: normal;
}

/* 画廊阶段标注区域整体更紧凑 */
div[data-testid="stSlider"] {
  margin-bottom: 0.15rem;
}
div.intent-actions .stButton > button {
  padding: 0.28rem 0.55rem;
  border-radius: 8px;
}
div.gallery-controls .stButton > button {
  padding: 0.18rem 0.45rem;
  border-radius: 6px;
  line-height: 1;
  min-width: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ===== I/O =====
def load_json(p: Path) -> Dict[str, Any]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(d: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def get_annotated_book_stems(out_dir: Path) -> set:
    """
    从 books_annotated 中收集已经标注过的书名 stem。
    例如：book_34_annotated.json -> book_34
    """
    stems = set()
    if not out_dir.exists():
        return stems
    for p in out_dir.glob("*_annotated.json"):
        stem = p.stem
        if stem.endswith("_annotated"):
            stems.add(stem[:-10])  # 去掉 "_annotated"
    return stems

def format_book_option(book_name: str, annotated_stems: set) -> str:
    """
    仅用于下拉框显示：
    - 未标注：book_34.json
    - 已标注：book_34.json  ✅已标注
    """
    stem = Path(book_name).stem
    if stem in annotated_stems:
        return f"{book_name}  ✅已标注"
    return book_name

def natural_sort_key(s: str):
    """
    自然排序：
    book_2.json < book_10.json < book_34.json
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", s)]

def load_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def format_segment_reference_text(raw_text: str) -> str:
    """
    将 STEPB_raw.txt 整理为更紧凑的显示格式：
    - 去掉 ==SEGMENTS== / ==END==
    - 支持 "#段1" / "# 段1" 两种段落标题格式
    - 去掉“起止页：”“意图：”字样，仅保留值
    - 将 “#段X｜页码范围｜意图” 压到同一行
    - 段与段之间不额外空行
    """
    if not raw_text:
        return ""

    lines = [ln.strip() for ln in raw_text.splitlines()]
    lines = [ln for ln in lines if ln and ln not in {"==SEGMENTS==", "==END=="}]

    blocks = []
    cur = {"seg": "", "range": "", "intent": "", "reason": ""}

    def flush():
        if cur["seg"] or cur["range"] or cur["intent"] or cur["reason"]:
            title_parts = []
            if cur["seg"]:
                title_parts.append(cur["seg"])
            if cur["range"]:
                title_parts.append(cur["range"])
            if cur["intent"]:
                title_parts.append(cur["intent"])
            title = "｜".join(title_parts)
            if cur["reason"]:
                blocks.append(f"{title}\n依据：{cur['reason']}")
            else:
                blocks.append(title)

    for ln in lines:
        # 兼容 "#段1"、"# 段1"、"段1"
        if re.match(r"^#\s*段\d+\s*$", ln) or re.match(r"^段\d+\s*$", ln):
            flush()
            seg_num = re.sub(r"^#\s*", "", ln)   # 去掉开头 #
            seg_num = seg_num.replace(" ", "")   # 去掉中间空格，统一成“段1”
            cur = {"seg": f"#{seg_num}", "range": "", "intent": "", "reason": ""}
        elif ln.startswith("起止页"):
            cur["range"] = re.sub(r"^起止页[:：]\s*", "", ln)
        elif ln.startswith("意图"):
            cur["intent"] = re.sub(r"^意图[:：]\s*", "", ln)
        elif ln.startswith("依据"):
            cur["reason"] = re.sub(r"^依据[:：]\s*", "", ln)
        else:
            # 若依据跨多行，则并到依据后面；其他杂项行忽略
            if cur["reason"]:
                cur["reason"] += " " + ln

    flush()
    return "\n".join(blocks)
def get_reference_paths(base: Path, book_name: str):
    """
    参考文件命名约定：
    - 原书：book_6.json
    - 情感参考：emotional reference/book_6_MOSAIC_merged.json
    - 分段参考：segment reference/book_6_STEPB_raw.txt
    """
    stem = Path(book_name).stem
    emotion_ref_path = base / "emotional reference" / f"{stem}_MOSAIC_merged.json"
    segment_ref_path = base / "segment reference" / f"{stem}_STEPB_raw.txt"
    return emotion_ref_path, segment_ref_path

def extract_primary_secondary_from_ref_intensity(emotion_intensity: Dict[str, Any]):
    """
    从 emotion_intensity 提取：
    - 主情感：强度最高者（若并列，按 EMOTIONS 顺序取第一个）
    - 副情感：除主情感外，所有强度 > 0 的情感
    """
    vals = {}
    for emo in EMOTIONS:
        try:
            vals[emo] = int((emotion_intensity or {}).get(emo, 0) or 0)
        except Exception:
            vals[emo] = 0

    max_v = max(vals.values()) if vals else 0
    if max_v <= 0:
        return None, []

    primary = next((emo for emo in EMOTIONS if vals.get(emo, 0) == max_v), None)
    secondary = [emo for emo in EMOTIONS if emo != primary and vals.get(emo, 0) > 0]
    return primary, secondary

def build_emotion_reference_map(ref_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    输出：
    {
      1: {"primary_emotion": "...", "secondary_emotions": [...]},
      2: {...}
    }
    """
    out: Dict[int, Dict[str, Any]] = {}
    for p in (ref_data.get("pages") or []):
        try:
            page_no = int(p.get("page"))
        except Exception:
            continue
        intensity = p.get("emotion_intensity") or {}
        primary, secondary = extract_primary_secondary_from_ref_intensity(intensity)
        out[page_no] = {
            "primary_emotion": primary,
            "secondary_emotions": secondary,
        }
    return out

# ===== labels 保底 =====
def ensure_labels_fields(page: Dict[str, Any]) -> None:
    page.setdefault("labels", {})
    page["labels"].setdefault("primary_emotion", None)
    page["labels"].setdefault("secondary_emotions", [])
    page["labels"].setdefault("prev_trend_marks", [])
    page["labels"].setdefault("intent_tag", None)

# ===== 图片解析 =====
IMG_KEYS = ["image_resolved", "image", "image_path", "img", "pic"]

def resolve_image_path(page: Dict[str, Any], base_dir: Path, picture_dir: Path) -> Path:
    ir = page.get("image_resolved")
    if ir:
        p = Path(ir)
        if p.is_absolute() and p.exists():
            return p
        for c in [picture_dir / p, base_dir / "data" / "images" / p, base_dir / p]:
            if c.exists(): return c
    raw = None
    for k in IMG_KEYS[1:]:
        v = page.get(k)
        if v: raw = v; break
    if not raw:
        return Path()
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for c in [picture_dir / p, base_dir / "data" / "images" / p, base_dir / p]:
        if c.exists(): return c
    return Path()

def show_image_fixed_height(img_path: Path, height: int = 420) -> None:
    try:
        img = Image.open(img_path)
        w, h = img.size
        scale = height / max(1, h)
        st.image(img, width=int(w * scale))
    except Exception:
        st.caption(f"（图片无法加载：{img_path}）")

def thumb_base64(ipath: Path, target_h: int = 170) -> str:
    try:
        img = Image.open(ipath)
        w, h = img.size
        scale = target_h / max(1, h)
        img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""

# ===== 段落/阶段工具 =====
def segs_overlap(a1: int, a2: int, b1: int, b2: int) -> bool:
    return not (a2 < b1 or b2 < a1)

def check_no_overlap(segments: List[Dict[str, Any]], start: int, end: int) -> bool:
    for s in segments:
        if segs_overlap(start, end, int(s["start_page"]), int(s["end_page"])): return False
    return True

def normalize_segments(segments: List[Dict[str, Any]], total_pages: int) -> List[Dict[str, Any]]:
    clean = []
    for s in segments:
        try:
            a, b = int(s.get("start_page", 0)), int(s.get("end_page", 0))
            if b < a: a, b = b, a
            a, b = max(1, min(total_pages, a)), max(1, min(total_pages, b))
            intent = s.get("intent", None)
            if intent in INTENTS and a <= b:
                clean.append({"start_page": a, "end_page": b, "intent": intent})
        except Exception:
            continue
    clean.sort(key=lambda x: (x["start_page"], x["end_page"]))
    return clean

def apply_intent_segments_to_pages(data: Dict[str, Any]) -> None:
    """仅在内存里同步 intent_tag，便于 UI 呈现；此函数不写盘。"""
    segs = data.get("intent_segments") or []
    by_page = {}
    for s in segs:
        for pno in range(int(s["start_page"]), int(s["end_page"]) + 1):
            by_page[pno] = s["intent"]
    for pg in data.get("pages", []):
        ensure_labels_fields(pg)
        pg["labels"]["intent_tag"] = by_page.get(int(pg.get("page", 0)))

def merge_intent_tag_into_pages(existing_pages: List[Dict[str, Any]],
                                memory_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """返回更新后的 pages：
       - 以 existing_pages 为基准，逐页更新 labels.intent_tag 为 memory_pages 中对应页的 intent_tag
       - 若 existing 缺页，则补上 memory 的整页
    """
    mem_map = {int(p.get("page", i+1)): p for i, p in enumerate(memory_pages)}
    out = []
    seen = set()
    for ep in existing_pages or []:
        page_no = int(ep.get("page", 0))
        mp = mem_map.get(page_no)
        if mp:
            ep.setdefault("labels", {})
            mp.setdefault("labels", {})
            ep["labels"]["intent_tag"] = mp["labels"].get("intent_tag")
            out.append(ep)
            seen.add(page_no)
        else:
            out.append(ep)
    for page_no, mp in mem_map.items():
        if page_no not in seen:
            out.append(mp)
    out.sort(key=lambda x: int(x.get("page", 0)))
    return out

# ===== 情感重合 & 暂存保存 =====
def compute_overlap_emotions(prev_page: Dict[str, Any], curr_main: str, curr_subs: List[str]) -> List[str]:
    """
    趋势标注规则（primary_involved）：
    相邻页有同一情感，且该情感在两页中至少有一页是主情感，才要求标注趋势。
    也即：保留 主-主 / 主-副 / 副-主；排除纯 副-副 重合。
    """
    ensure_labels_fields(prev_page)

    prev_main = prev_page["labels"].get("primary_emotion")
    prev_subs = [e for e in (prev_page["labels"].get("secondary_emotions") or []) if e in EMOTIONS and e != prev_main]
    curr_subs = [e for e in (curr_subs or []) if e in EMOTIONS and e != curr_main]

    prev_all = set(prev_subs)
    curr_all = set(curr_subs)
    if prev_main in EMOTIONS:
        prev_all.add(prev_main)
    if curr_main in EMOTIONS:
        curr_all.add(curr_main)

    overlap = prev_all & curr_all
    primary_involved = {
        emo for emo in overlap
        if emo == prev_main or emo == curr_main
    }
    return sorted(primary_involved, key=lambda x: EMOTIONS.index(x) if x in EMOTIONS else 999)

def autosave_current_page(idx: int, data: Dict[str, Any], out_dir: Path,
                          enforce_trend: bool = True, write_to_disk: bool = False) -> bool:
    pages = data.get("pages") or []
    if not (0 <= idx < len(pages)):
        return True
    page = pages[idx]
    ensure_labels_fields(page)

    main_key = f"main_{page.get('page')}"
    main_val = st.session_state.get(main_key, None)
    main_label = main_val if main_val in EMOTIONS else None
    if not main_label:
        st.warning("请先为本页选择主情感。")
        return False

    sub_key = f"sub_{page.get('page')}_{main_label}"
    chosen = st.session_state.get(sub_key, page["labels"].get("secondary_emotions") or [])
    sub_labels = [e for e in chosen if e in EMOTIONS and e != main_label]

    trend_marks: List[str] = []
    if idx > 0:
        prev = pages[idx-1]
        overlap = compute_overlap_emotions(prev, main_label, sub_labels)
        if overlap and enforce_trend:
            for emo in overlap:
                tkey = f"trend_{page.get('page')}_{emo}"
                val = st.session_state.get(tkey, None)
                if val not in TREND_CHOICES:
                    st.warning(f"请先为情感「{emo}」选择趋势（+ / − / =）")
                    return False
            for emo in overlap:
                tkey = f"trend_{page.get('page')}_{emo}"
                label = st.session_state.get(tkey, None)
                sign = TREND_CHOICES.get(label, None)
                if sign: trend_marks.append(f"{emo}{sign}")
        elif overlap:
            for emo in overlap:
                tkey = f"trend_{page.get('page')}_{emo}"
                label = st.session_state.get(tkey, None)
                sign = TREND_CHOICES.get(label, None)
                if sign: trend_marks.append(f"{emo}{sign}")

    page["labels"]["primary_emotion"] = main_label
    page["labels"]["secondary_emotions"] = list(sub_labels)
    page["labels"]["prev_trend_marks"]  = trend_marks

    if write_to_disk:
        out_path = out_dir / f"{data.get('book_id','book')}_annotated.json"
        save_json(data, out_path)
    return True

# ===== 全量校验（最后一页保存时使用） =====
def validate_all_pages_before_save(data: Dict[str, Any]) -> Dict[str, List[int]]:
    pages = data.get("pages") or []
    missing_main, missing_trend = [], []

    for i, pg in enumerate(pages):
        ensure_labels_fields(pg)
        main = pg["labels"].get("primary_emotion")
        subs = pg["labels"].get("secondary_emotions") or []
        page_no = int(pg.get("page") or i+1)

        if main not in EMOTIONS:
            missing_main.append(page_no)
            continue

        if i > 0:
            prev = pages[i-1]
            overlap = compute_overlap_emotions(prev, main, subs)
            if overlap:
                marks = pg["labels"].get("prev_trend_marks") or []
                done = set(m[:-1] for m in marks if len(m) >= 2)
                if not set(overlap).issubset(done):
                    missing_trend.append(page_no)

    return {"missing_main": sorted(missing_main), "missing_trend": sorted(missing_trend)}

# ===== 侧边栏 =====
with st.sidebar:
    st.header("项目目录")
    default_base = st.session_state.get("base_dir", str(Path(__file__).resolve().parents[2]))
    base_dir_str = st.text_input("仓库根目录（读取 data/books 和 data/images；写入 outputs/annotations）", value=default_base, key="base_dir")
    base = Path(base_dir_str).expanduser()
    books_dir = base / "data" / "books"
    out_dir   = base / "outputs" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    st.caption(f"📁 书库：{books_dir}")
    st.caption(f"💾 输出：{out_dir}")

    try:
        raw_options = [p.name for p in books_dir.glob("*.json")] if books_dir.exists() else []
        annotated_stems = get_annotated_book_stems(out_dir)

        # 未标注在前，已标注在后；各组内部继续按自然排序
        options = sorted(
            raw_options,
            key=lambda name: (
                Path(name).stem in annotated_stems,
                natural_sort_key(name)
            )
        )
    except Exception:
        options = []
        annotated_stems = set()

    selected_json = st.selectbox(
        "选择一本书",
        options=options,
        key="book_sel",
        format_func=lambda name: format_book_option(name, annotated_stems)
    )

    col_a, col_b = st.columns(2)
    with col_a:
        btn_load = st.button("载入/重载", type="primary", key="btn_load")
    with col_b:
        btn_fill = st.button("补齐空 labels", key="btn_fill")

    st.markdown("---")
    st.caption("显示参数")
    st.number_input("图片高度(px)", min_value=160, max_value=800, value=420, step=20, key="img_h")
    st.number_input("文本高度(px)", min_value=120, max_value=800, value=180, step=20, key="txt_h")

# ===== 加载数据 =====
if btn_load or "data" not in st.session_state:
    if st.session_state.get("book_sel"):
        data_path = books_dir / st.session_state["book_sel"]
        data = load_json(data_path)
        if data:
            for p in data.get("pages", []):
                ensure_labels_fields(p)
            if "intent_segments" not in data:
                data["intent_segments"] = []
            pd = data.get("picture_dir", "")
            if pd:
                pd_path = Path(pd)
                if not pd_path.is_absolute():
                    pd_path = (data_path.parent / pd_path).resolve()
            else:
                pd_path = (base / "data" / "images" / data_path.stem).resolve()
            st.session_state.picture_dir = pd_path
            st.session_state.data = data
            st.session_state.idx = 0

            emotion_ref_path, segment_ref_path = get_reference_paths(base, st.session_state["book_sel"])
            emotion_ref_data = load_json(emotion_ref_path) if emotion_ref_path.exists() else {}
            segment_ref_text = format_segment_reference_text(load_text(segment_ref_path)) if segment_ref_path.exists() else ""

            st.session_state.emotion_reference_map = build_emotion_reference_map(emotion_ref_data)
            st.session_state.segment_reference_text = segment_ref_text
            st.session_state.emotion_reference_path = str(emotion_ref_path)
            st.session_state.segment_reference_path = str(segment_ref_path)
        else:
            st.session_state.emotion_reference_map = {}
            st.session_state.segment_reference_text = ""
            st.session_state.emotion_reference_path = ""
            st.session_state.segment_reference_path = ""
            st.error(f"JSON 为空或无法解析：{data_path}")

if btn_fill and "data" in st.session_state:
    for p in st.session_state.data.get("pages", []):
        ensure_labels_fields(p)
    st.success("已为所有页面补齐 labels 字段。")

# ===== 主体区域 =====
if "data" not in st.session_state:
    st.stop()

data: Dict[str, Any] = st.session_state["data"]
picture_dir: Path = Path(st.session_state.get("picture_dir"))
pages: List[Dict[str, Any]] = data.get("pages", [])
if not pages:
    st.stop()

N = len(pages)
idx = int(st.session_state.get("idx", 0))
idx = max(0, min(N-1, idx))
page = pages[idx]
ensure_labels_fields(page)

emotion_reference_map: Dict[int, Dict[str, Any]] = st.session_state.get("emotion_reference_map") or {}
page_no = int(page.get("page") or idx + 1)
emotion_ref = emotion_reference_map.get(page_no, {})

segment_reference_text = st.session_state.get("segment_reference_text", "") or ""
segment_reference_path = st.session_state.get("segment_reference_path", "") or ""

st.title(f"绘本情感标注器 · {data.get('book_id', '未命名')}")

colL, colR = st.columns([1, 1])

with colL:
    st.subheader(f"第 {idx+1}/{N} 页")
    ipath = resolve_image_path(page, base, picture_dir)
    if ipath.exists():
        show_image_fixed_height(ipath, st.session_state.get("img_h", 420))
    else:
        hint = None
        for k in IMG_KEYS:
            if page.get(k): hint = page.get(k); break
        if hint:
            st.warning(f"未找到图片：{hint}（已尝试 {picture_dir}、{base/'data'/'images'}、{base}）")
        else:
            st.info("本页 JSON 未提供 image 路径。")

    st.write("**文本：**")
    st.text_area(
        "文本", value=page.get("text", ""),
        height=st.session_state.get("txt_h", 180),
        disabled=True, key=f"txt_display_{page.get('page')}"
    )

with colR:
    st.subheader("当前页情感标注")

    st.markdown("**模型情感参考（当前页）**")
    if emotion_ref:
        ref_main = emotion_ref.get("primary_emotion") or "—"
        ref_subs = "、".join(emotion_ref.get("secondary_emotions") or []) or "—"
        st.caption(f"主情感：{ref_main}｜副情感：{ref_subs}")
    else:
        st.caption("未找到当前页对应的情感参考。")

    main_key = f"main_{page.get('page')}"
    if (page["labels"].get("primary_emotion") in EMOTIONS) and (main_key not in st.session_state):
        st.session_state[main_key] = page["labels"]["primary_emotion"]
    if main_key in st.session_state and st.session_state[main_key] in EMOTIONS:
        _main_index = EMOTIONS.index(st.session_state[main_key])
    else:
        _main_index = None
    main_label = st.radio("主情感（单选）", EMOTIONS, index=_main_index, key=main_key)

    st.caption("副情感（可多选）")
    if main_label in EMOTIONS:
        sub_candidates = [e for e in EMOTIONS if e != main_label]
        agg_key = f"sub_{page.get('page')}_{main_label}"
        if agg_key not in st.session_state:
            st.session_state[agg_key] = [e for e in (page["labels"].get("secondary_emotions") or []) if
                                         e in sub_candidates]
        sub_cols = st.columns(4)
        for i, emo in enumerate(sub_candidates):
            chk = f"subchk_{page.get('page')}_{main_label}_{emo}"
            if chk not in st.session_state:
                st.session_state[chk] = (emo in st.session_state[agg_key])
            with sub_cols[i % 4]:
                st.checkbox(emo, key=chk)
        chosen_subs = [emo for emo in sub_candidates if
                       st.session_state.get(f"subchk_{page.get('page')}_{main_label}_{emo}", False)]
        st.session_state[agg_key] = chosen_subs
        page["labels"]["primary_emotion"] = main_label
        page["labels"]["secondary_emotions"] = chosen_subs
    else:
        page["labels"]["primary_emotion"] = None
        page["labels"]["secondary_emotions"] = []
        chosen_subs = []
        st.info("请先选择主情感。")

    trend_marks: List[str] = []
    current_main = main_label if main_label in EMOTIONS else None
    current_subs = chosen_subs if current_main else []

    if idx > 0 and current_main:
        prev = pages[idx - 1]
        overlap = compute_overlap_emotions(prev, current_main, current_subs)
        if overlap:
            st.markdown("**与上一页相同的情感：请选择趋势（+ / − / =）**")
            existing = {(m[:-1]): m[-1] for m in (page["labels"].get("prev_trend_marks") or []) if len(m) >= 2}

            for emo in overlap:
                tkey = f"trend_{page.get('page')}_{emo}"
                if tkey not in st.session_state and emo in existing:
                    sign = existing[emo]
                    for k, v in TREND_CHOICES.items():
                        if v == sign:
                            st.session_state[tkey] = k
                            break

                opts = list(TREND_CHOICES.keys())
                if tkey in st.session_state and st.session_state[tkey] in opts:
                    t_idx = opts.index(st.session_state[tkey])
                else:
                    t_idx = None

                choice = st.radio(
                    f"「{emo}」趋势",
                    options=opts,
                    index=t_idx,
                    key=tkey,
                    horizontal=True
                )
                if choice in TREND_CHOICES:
                    trend_marks.append(f"{emo}{TREND_CHOICES[choice]}")
        else:
            st.info("与上一页无相同情感，无需趋势标注。")
    elif idx == 0:
        st.info("第1页无上一页，无需趋势标注。")

    st.write("")
    nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([1, 1, 1, 1])

    with nav_c1:
        if st.button("⬅️ 上一页", disabled=(idx == 0), key="btn_prev_top"):
            # 回看上一页时不做强校验，方便比较后再决定当前页的主情感/趋势
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    with nav_c2:
        if st.button("➡️ 下一页", disabled=(idx >= N - 1), key="btn_next_top"):
            # 向后翻页时维持强校验，避免漏标主情感或趋势
            ok = autosave_current_page(idx, data, out_dir, enforce_trend=True, write_to_disk=False)
            if ok:
                st.session_state.idx = min(N - 1, idx + 1)
                st.rerun()

    with nav_c3:
        jump_val = st.number_input(
            "页码",
            min_value=1,
            max_value=N,
            value=idx + 1,
            step=1,
            key=f"jump_{data.get('book_id', 'book')}",
            label_visibility="collapsed"
        )

    with nav_c4:
        if st.button("跳转", key="btn_jump"):
            target_idx = int(jump_val) - 1

            if target_idx < idx:
                # 向前跳转：允许直接跳，便于回看比较
                st.session_state.idx = target_idx
                st.rerun()
            elif target_idx > idx:
                # 向后跳转：仍然强校验
                ok = autosave_current_page(idx, data, out_dir, enforce_trend=True, write_to_disk=False)
                if ok:
                    st.session_state.idx = target_idx
                    st.rerun()
            else:
                # 跳到当前页：不做任何事
                pass

    # 当前页实时同步到内存，避免趋势显示与校验不一致
    page["labels"]["primary_emotion"] = main_label if main_label in EMOTIONS else None
    page["labels"]["secondary_emotions"] = chosen_subs if main_label in EMOTIONS else []
    page["labels"]["prev_trend_marks"] = trend_marks if idx > 0 else []

    if idx == N - 1:
        if st.button("💾 保存本书页级标注", key="btn_save_lastpage"):
            ok = autosave_current_page(idx, data, out_dir, enforce_trend=True, write_to_disk=False)
            if ok:
                report = validate_all_pages_before_save(data)
                if report["missing_main"] or report["missing_trend"]:
                    if report["missing_main"]:
                        st.error("保存失败：以下页**缺少主情感** → " + ", ".join(map(str, report["missing_main"])))
                    if report["missing_trend"]:
                        st.error("保存失败：以下页**存在重合情感但趋势未标注** → " + ", ".join(
                            map(str, report["missing_trend"])))
                else:
                    out_path = out_dir / f"{data.get('book_id', 'book')}_annotated.json"

                    # --- 将 picture_dir 统一写为相对 base 的 POSIX 字符串 ---
                    pd0 = data.get("picture_dir", "")
                    if pd0:
                        p0 = Path(pd0)
                        if not p0.is_absolute():
                            data["picture_dir"] = p0.as_posix()
                        else:
                            try:
                                data["picture_dir"] = p0.relative_to(base).as_posix()
                            except ValueError:
                                data["picture_dir"] = p0.as_posix()
                    else:
                        data["picture_dir"] = f"../images/{data.get('book_id', 'book')}"

                    save_json(data, out_path)
                    st.success(f"已保存至：{out_path}\n\n👉 请下滑至页面底部，完成 **阶段标注**。")

    st.divider()

# ===== 画廊式阶段标注 =====
st.subheader("画廊式阶段标注")

# 先显示模型分段参考（整本书）——放在操作区上方，方便标注者先看再操作
st.markdown("<div style='font-size:0.92rem;font-weight:600;margin-bottom:4px;'>模型分段参考（整本书）</div>", unsafe_allow_html=True)
if segment_reference_text:
    seg_ref_html = html.escape(segment_reference_text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            border:1px solid #ececf1;
            border-radius:8px;
            background:#f8f9fb;
            padding:8px 10px;
            font-size:0.80rem;
            line-height:1.30;
            color:#8b9099;
            max-height:115px;
            overflow-y:auto;
            margin-top:2px;
            margin-bottom:8px;
            white-space:normal;
        ">{seg_ref_html}</div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("未找到 segment reference 文件。")

# 画廊列数：默认值（可改为你想要的默认列数）
if "gallery_cols" not in st.session_state:
    st.session_state.gallery_cols = 4

# 选择连续页范围
rng = st.slider("选择连续页范围", 1, N, (max(1, idx+1), min(N, idx+5)), key="gallery_range")

# 顶部一行：意图标签 + 删除/添加/保存 + 右侧列数控制
row = st.columns([13, 1, 1, 1, 2])

with row[0]:
    st.markdown("<div style='font-size:0.84rem;color:#777;margin-bottom:2px;'>选择意图标签</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="intent-row">', unsafe_allow_html=True)
        gallery_intent = st.radio(
            "选择意图", INTENTS, index=0, key="gallery_intent_radio",
            horizontal=True, label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

with row[1]:
    st.markdown('<div class="intent-actions">', unsafe_allow_html=True)
    if st.button("删除", key="btn_seg_del_one"):
        segs = data.get("intent_segments") or []
        if segs:
            segs = normalize_segments(segs, N)
            removed = segs.pop()
            data["intent_segments"] = segs
            # 改为右上角弹窗提示
            st.toast(f"已删除：{removed['start_page']}–{removed['end_page']}：{removed['intent']}", icon="🗑️")
            st.rerun()
        else:
            # 无可删阶段时也用弹窗
            st.toast("当前无阶段可删。", icon="ℹ️")

with row[2]:
    st.markdown('<div class="intent-actions">', unsafe_allow_html=True)
    if st.button("添加", key="btn_gallery_add"):
        a, b = int(rng[0]), int(rng[1])
        if b < a: a, b = b, a
        segs = data.get("intent_segments") or []
        if not check_no_overlap(segs, a, b):
            st.toast(f"添加失败：选择的页面范围 {a}–{b} 与已有阶段发生重叠。", icon="❌")
        else:
            segs.append({"start_page": a, "end_page": b, "intent": gallery_intent})
            data["intent_segments"] = normalize_segments(segs, N)
            st.toast("已添加阶段。", icon="✅")
            st.rerun()

with row[3]:
    st.markdown('<div class="intent-actions">', unsafe_allow_html=True)
    if st.button("保存", key="btn_seg_save"):
        # 1) 标准化并在内存同步 intent_tag
        segs = normalize_segments(data.get("intent_segments") or [], N)
        data["intent_segments"] = segs
        apply_intent_segments_to_pages(data)   # 内存同步：每页 labels.intent_tag

        # 2) 写盘：intent_segments + pages(仅 intent_tag 变更/或首次写入)
        out_path = out_dir / f"{data.get('book_id','book')}_annotated.json"
        existing = load_json(out_path) if out_path.exists() else {}

        if isinstance(existing, dict) and isinstance(existing.get("pages"), list):
            merged_pages = merge_intent_tag_into_pages(existing.get("pages"), data.get("pages", []))
        else:
            merged_pages = data.get("pages", [])

        merged = dict(existing) if isinstance(existing, dict) else {}
        merged["book_id"] = data.get("book_id")
        merged["intent_segments"] = segs
        merged["pages"] = merged_pages

        try:
            save_json(merged, out_path)
            st.toast("阶段已保存，并已将意图同步到每一页（已写入 JSON）。", icon="✅")
        except Exception as e:
            st.toast(f"保存失败：{e}", icon="❌")

# 右侧：当前列数 + ± 控制
with row[4]:
    cnt_col, minus_col, plus_col = st.columns([1.4, 0.6, 0.6])
    with cnt_col:
        st.caption("列数")
        st.markdown(
            f"<div style='border:1px solid #e6e6e6;border-radius:8px;padding:8px 10px;text-align:center;'>"
            f"{st.session_state.gallery_cols}</div>",
            unsafe_allow_html=True
        )
    with minus_col:
        st.markdown('<div class="gallery-controls">', unsafe_allow_html=True)
        if st.button("−", key="cols_minus"):
            st.session_state.gallery_cols = max(3, st.session_state.gallery_cols - 1)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with plus_col:
        st.markdown('<div class="gallery-controls">', unsafe_allow_html=True)
        if st.button("＋", key="cols_plus"):
            st.session_state.gallery_cols = min(8, st.session_state.gallery_cols + 1)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# 设定列数
cols = int(st.session_state.gallery_cols)

# 渲染“当前阶段”
data["intent_segments"] = normalize_segments(data.get("intent_segments") or [], N)
segments = data["intent_segments"]
if segments:
    st.success("当前阶段：\n- " + "\n- ".join([f"{s['start_page']}–{s['end_page']}：{s['intent']}" for s in segments]))
else:
    st.info("当前暂无阶段。")

# ===== 画廊显示（固定高度 + 整卡描边 + 文本可滚动 + 显示“意图”）=====
selected_set = set(range(int(rng[0]), int(rng[1]) + 1))
rows = (N + cols - 1) // cols

for r in range(rows):
    col_objs = st.columns(cols)
    for c in range(cols):
        i = r * cols + c
        if i >= N: continue
        pg = pages[i]; ensure_labels_fields(pg)
        in_sel = (pg["page"] in selected_set)

        ipp = resolve_image_path(pg, base, picture_dir)
        b64 = thumb_base64(ipp) if ipp.exists() else ""
        if b64:
            img_html = (
                "<img src='data:image/jpeg;base64," + b64 +
                "' style='height:170px; width:auto; display:block; margin:0 auto;' />"
            )
        else:
            img_html = (
                "<div style='height:170px;display:flex;align-items:center;justify-content:center;'>（无图）</div>"
            )

        raw_text = (pg.get("text", "") or "")
        flat_text = raw_text.replace("\n", " ")
        intent_badge = pg["labels"].get("intent_tag") or "—"
        border_css = "3px solid #10a37f" if in_sel else "1px solid #e6e6e6"

        box_html = (
            f"<div style=\"border:{border_css}; border-radius:10px; padding:8px;"
            f"height:340px; display:flex; flex-direction:column; gap:6px;\">"
              f"<div><strong>P{pg.get('page')}</strong></div>"
              f"<div>{img_html}</div>"
              f"<div style='font-size:0.85rem;color:#666;'>主：{pg['labels'].get('primary_emotion') or '—'}"
              f"｜副：{','.join(pg['labels'].get('secondary_emotions') or []) or '—'}"
              f"｜意图：{intent_badge}</div>"
              f"<div class='card-text'>{flat_text}</div>"
            f"</div>"
        )
        with col_objs[c]:
            st.markdown(box_html, unsafe_allow_html=True)
