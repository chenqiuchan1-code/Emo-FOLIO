# -*- coding: utf-8 -*-


import io
import os
import re
import json
import time
import base64
from urllib.parse import quote
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from PIL import Image
from mosaic.runtime import guarded_request, RunStatus, save_run_status, half_open_probe_ok

# ============== 日志 ==============
def log(msg: str) -> None:
    print(msg, flush=True)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

HF_ENDPOINT_API_KEY_ENV = "HF_ENDPOINT_API_KEY"
HF_INTERNVL35_8B_BASE_URL_ENV = "HF_INTERNVL35_8B_BASE_URL"
HF_INTERNVL35_38B_BASE_URL_ENV = "HF_INTERNVL35_38B_BASE_URL"

HF_IMAGE_REPO_ID_ENV = "HF_IMAGE_REPO_ID"
HF_IMAGE_REVISION_ENV = "HF_IMAGE_REVISION"
HF_IMAGE_ROOT_ENV = "HF_IMAGE_ROOT"
HF_LOCAL_IMAGE_ROOT_ENV = "HF_LOCAL_IMAGE_ROOT"
INTERNVL_IMAGE_SOURCE_ENV = "INTERNVL_IMAGE_SOURCE"

def is_internvl35_8b_model(model: str) -> bool:
    return str(model or "").lower() == "internvl35-8b"

def is_internvl35_38b_model(model: str) -> bool:
    return str(model or "").lower() == "internvl35-38b"

def is_internvl_model(model: str) -> bool:
    return is_internvl35_8b_model(model) or is_internvl35_38b_model(model)

def use_hf_dataset_image_url(model: str) -> bool:
    if not is_internvl_model(model):
        return False
    source = os.getenv(INTERNVL_IMAGE_SOURCE_ENV, "local").strip().lower() or "local"
    if source not in {"local", "remote"}:
        raise ValueError(f"{INTERNVL_IMAGE_SOURCE_ENV} must be 'local' or 'remote', got: {source}")
    return source == "remote"

def _get_hf_endpoint_base_url(model: str) -> str:
    if is_internvl35_8b_model(model):
        return os.getenv(HF_INTERNVL35_8B_BASE_URL_ENV, "").strip()
    if is_internvl35_38b_model(model):
        return os.getenv(HF_INTERNVL35_38B_BASE_URL_ENV, "").strip()
    return ""

def is_qwen_model(model: str) -> bool:
    return str(model or "").lower().startswith("qwen")

def is_gemini_model(model: str) -> bool:
    return str(model or "").startswith("gemini-")

def resolve_api_model_name(model: str) -> str:
    m = str(model or "").strip()

    if m == "internvl35-8b":
        return "OpenGVLab/InternVL3_5-8B-Instruct"
    if m == "internvl35-38b":
        return "OpenGVLab/InternVL3_5-38B-Instruct"

    return m

def make_client_for_model(model: str) -> OpenAI:
    if is_gemini_model(model):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("未设置 GEMINI_API_KEY，请先在本地环境变量中配置 Gemini API Key。")
        return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    if is_qwen_model(model):
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("当前模型为 Qwen，但未设置 DASHSCOPE_API_KEY。")
        return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)

    if is_internvl_model(model):
        api_key = os.getenv(HF_ENDPOINT_API_KEY_ENV, "").strip()
        base_url = _get_hf_endpoint_base_url(model)
        if not api_key:
            raise RuntimeError("当前模型走 Hugging Face Endpoint，但未设置 HF_ENDPOINT_API_KEY。")
        if not base_url:
            raise RuntimeError(f"当前模型为 {model}，但未设置对应的 Hugging Face Endpoint BASE_URL。")
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI()

# ============== 与 baseline 对齐的底座函数 ==============
def downscale_to_data_url(img_path: Path, max_side: int = 512) -> str:
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            scale = min(1.0, max_side / max(im.size))
            if scale < 1.0:
                im = im.resize((int(im.width * scale), int(im.height * scale)))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

def normalize_pages(data: Any) -> List[Dict[str, Any]]:
    pages = data["pages"] if isinstance(data, dict) else data
    result = []
    for i, p in enumerate(pages):
        page_num = p.get("page", i + 1)
        text = p.get("text", "")
        img = p.get("image") or p.get("image_path") or p.get("img") or p.get("image_url", "")
        result.append({"page": int(page_num), "text": str(text), "image_path": str(img)})
    result.sort(key=lambda x: x["page"])
    return result

def resolve_picture_dir(json_path: Path, data: Any) -> Optional[Path]:
    json_dir = json_path.parent
    picture_dir_val = data.get("picture_dir") if isinstance(data, dict) else None
    if not picture_dir_val:
        return None
    pd = Path(picture_dir_val)
    if pd.is_absolute():
        return pd if pd.exists() else None
    guess1 = (json_dir / pd).resolve()
    if guess1.exists():
        return guess1
    guess2 = (json_dir.parent / pd).resolve()
    if guess2.exists():
        return guess2
    log(f"⚠️ 指定的 picture_dir 未找到：尝试 {guess1} 与 {guess2} 均不存在。")
    return None

def resolve_image_path(raw_img: str, json_path: Path, picture_dir_abs: Optional[Path]) -> Path:
    p = Path(raw_img)
    if p.is_absolute():
        return p
    json_dir = json_path.parent
    candidates = []
    if picture_dir_abs:
        candidates.append(picture_dir_abs / p)
    candidates.append(json_dir / p)
    candidates.append(json_dir.parent / p)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else p

def _guess_local_picture_root(img_path: Path, picture_dir_abs: Optional[Path], remote_root_name: str) -> Optional[Path]:
    img_abs = img_path.resolve()
    candidates: List[Path] = []

    if picture_dir_abs:
        pda = picture_dir_abs.resolve()
        if pda.name == remote_root_name:
            candidates.append(pda)
        if pda.parent.name == remote_root_name:
            candidates.append(pda.parent)

    for anc in img_abs.parents:
        if anc.name == remote_root_name:
            candidates.append(anc)
            break

    for root in candidates:
        try:
            img_abs.relative_to(root)
            return root
        except Exception:
            continue

    return None


def build_hf_dataset_image_url(img_path: Path, picture_dir_abs: Optional[Path]) -> str:
    repo_id = os.getenv(HF_IMAGE_REPO_ID_ENV, "").strip()
    revision = os.getenv(HF_IMAGE_REVISION_ENV, "main").strip() or "main"
    remote_root = os.getenv(HF_IMAGE_ROOT_ENV, "images").strip().strip("/") or "images"
    local_root_name = os.getenv(HF_LOCAL_IMAGE_ROOT_ENV, "images").strip().strip("/") or "images"

    if not repo_id:
        raise RuntimeError(
            f"{INTERNVL_IMAGE_SOURCE_ENV}=remote, but {HF_IMAGE_REPO_ID_ENV} is unset. "
            "Upload the released images to a repository you control and configure its repository ID."
        )

    img_abs = img_path.resolve()
    local_root = _guess_local_picture_root(img_abs, picture_dir_abs, local_root_name)
    if local_root is None:
        raise RuntimeError(
            f"无法从本地路径推断 HF 图片相对路径：{img_abs}。"
            f"请确认图片目录结构中包含名为 '{local_root_name}' 的本地根目录。"
        )

    rel_path = img_abs.relative_to(local_root).as_posix()
    remote_path = f"{remote_root}/{rel_path}"

    return (
        f"https://huggingface.co/datasets/"
        f"{quote(repo_id, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/"
        f"{quote(remote_path, safe='/')}"
    )


def build_image_ref_for_model(
    img_path: Path,
    picture_dir_abs: Optional[Path],
    model: str,
    *,
    max_side: int,
    data_url_builder,
) -> str:
    if use_hf_dataset_image_url(model):
        return build_hf_dataset_image_url(img_path, picture_dir_abs)
    return data_url_builder(img_path, max_side=max_side)

# ================== 提示词（强化“图像线索”客观性） ==================
INSTRUCTIONS = """你是儿童绘本情感理解专家。请严格遵守以下原则：
1. 候选情感仅限 {快乐, 悲伤, 恐惧, 愤怒, 惊讶, 平静, 好奇}。
2. 文本线索仅来自文字内容；图像线索必须来自图像本体要素，禁止复述文本或主观评价。
3. 输出必须严格遵守固定结构（BOOK_SUMMARY、PAGE_SUMMARIES），不得输出额外解释或 Markdown。
"""

PROMPT_HEAD = """你将看到一本儿童绘本的原始图文数据（按页）。请完成以下任务：

① 全书分析
- BOOK_SUMMARY：用80–150字概括全书的主要事件、关系变化与情节推进。

② 分页分析
对每一页输出以下四项：
- 语义摘要：概括该页最关键的事件或状态变化，尽量体现其在全书推进中的作用。
- 候选情感：只能从{快乐, 悲伤, 恐惧, 愤怒, 惊讶, 平静, 好奇}中选Top-3，不得输出其他标签。
- 文本线索：只写页面文字中能支持情绪或意图判断的短语。
- 图像线索：将该页图像压缩为可供后续任务直接使用的视觉替代表示，具体写法遵循下方 VISUAL_CUES_STYLE_GUIDE。

请按固定格式输出，不要添加解释，不要使用Markdown。
"""

# 视觉线索的风格与词表引导（防止主观/文本化）
VISUAL_CUES_STYLE_GUIDE = """==VISUAL_CUES_STYLE_GUIDE==
“图像线索”将作为后续模型看不到原图时的替代输入，因此不能只列零散名词，必须尽量保留当前页的视觉状态与关系结构。

请遵守：
1. 每页写成3–5个信息密度较高的分句，总体尽量控制在45–90字。
2. 优先覆盖以下信息：
- 主要角色的表情、动作、姿态
- 角色之间的相对关系，如靠近、分离、追逐、对抗、陪伴、回避等
- 关键道具、动作结果或事件痕迹
- 场景是否延续前页或切换到新场景
- 文字未直说、但图中清楚可见的重要细节
3. 每个分句尽量包含“对象 + 动作/状态 + 关系/结果”，不要只罗列名词。
4. 少写低信息背景词，如“有树木、花草、房子、天空、色彩鲜艳”；除非这些变化本身推动了情节或场景切换。
5. 保持客观，尽量写视觉事实，不直接下情绪结论。
==END_GUIDE==
"""

PROMPT_TAIL = """输出格式为纯文本，严格使用以下结构：

==BOOK_SUMMARY==
<全书摘要，80–150字>

==PAGE_SUMMARIES==
# 第1页
语义摘要：<≤40字>
候选情感：<七类中的Top-3，顿号分隔>
文本线索：<短语，顿号分隔>
图像线索：<3-5个分句，用分号分隔，形成对该页图像的压缩替代表示>

# 第2页
...

（直到最后一页）

==END==
"""

# 允许的七类 + 常见同义词映射
ALLOWED = {"快乐", "悲伤", "恐惧", "愤怒", "惊讶", "平静", "好奇"}

SYN_MAP = {
    # —— 快乐（积极愉悦/轻松玩耍/正反馈）——
    "兴奋": "快乐", "满意": "快乐", "得意": "快乐", "轻松": "快乐", "开心": "快乐", "高兴": "快乐", "愉快": "快乐",
    "喜悦": "快乐", "欢喜": "快乐", "欢快": "快乐", "愉悦": "快乐", "幸福": "快乐", "快乐极了": "快乐",
    "有趣": "快乐", "好玩": "快乐", "玩得开心": "快乐", "玩耍": "快乐", "调皮": "快乐", "淘气": "快乐",
    "满足": "快乐", "成就感": "快乐", "被表扬": "快乐", "被夸奖": "快乐",

    # —— 好奇（探索/注意力集中/求知欲/不解）——
    "期待": "好奇", "渴望": "好奇", "希望": "好奇", "欲望": "好奇", "疑惑": "好奇",
    "好奇心": "好奇", "困惑": "好奇", "不解": "好奇", "纳闷": "好奇", "奇怪": "好奇",
    "专注": "好奇", "专心": "好奇", "投入": "好奇", "观察": "好奇", "打量": "好奇", "探究": "好奇",
    "探索": "好奇", "思考": "好奇", "思索": "好奇", "研究": "好奇", "打听": "好奇", "兴致勃勃": "好奇",
    "迫不及待": "好奇", "想知道": "好奇",

    # —— 恐惧（担忧/紧张/害羞/惊恐）——
    "担心": "恐惧", "焦虑": "恐惧", "无助": "恐惧", "害怕": "恐惧",
    "担忧": "恐惧", "忧虑": "恐惧", "顾虑": "恐惧", "紧张": "恐惧", "不安": "恐惧", "惴惴不安": "恐惧",
    "畏惧": "恐惧", "心虚": "恐惧", "胆怯": "恐惧", "怯生": "恐惧", "害羞": "恐惧",
    "惊恐": "恐惧", "恐慌": "恐惧", "惊险": "恐惧", "受惊": "恐惧", "被吓到": "恐惧",

    # —— 悲伤（低落/失落/孤独/委屈/疲惫/哭泣）——
    "难过": "悲伤", "失望": "悲伤", "悲痛": "悲伤", "疲惫": "悲伤", "伤心": "悲伤",
    "羞愧": "悲伤", "愧疚": "悲伤", "内疚": "悲伤", "沮丧": "悲伤",
    "伤感": "悲伤", "忧伤": "悲伤", "忧郁": "悲伤", "失落": "悲伤", "落寞": "悲伤", "孤单": "悲伤", "孤独": "悲伤",
    "难受": "悲伤", "痛苦": "悲伤", "委屈": "悲伤",
    "想哭": "悲伤", "哭泣": "悲伤", "流泪": "悲伤", "含泪": "悲伤",

    # —— 愤怒（生气/恼怒/不满/反感/厌恶/发火）——
    "生气": "愤怒", "气愤": "愤怒", "恼怒": "愤怒", "恼火": "愤怒", "愠怒": "愤怒", "气恼": "愤怒",
    "愤怒": "愤怒", "震怒": "愤怒", "怒火": "愤怒", "暴怒": "愤怒", "发火": "愤怒", "生闷气": "愤怒",
    "愤慨": "愤怒", "不满": "愤怒", "抵触": "愤怒", "反感": "愤怒", "厌恶": "愤怒", "嫌弃": "愤怒",

    # —— 惊讶（出乎意料/突然/惊喜/震惊/诧异）——
    "惊讶": "惊讶", "惊喜": "惊讶", "惊奇": "惊讶", "意外": "惊讶", "意想不到": "惊讶",
    "吃惊": "惊讶", "吓一跳": "惊讶", "诧异": "惊讶", "震惊": "惊讶", "惊呆": "惊讶",

    # —— 平静（镇定/安心/舒适/温暖/安宁/放松/冷静/宁静）——
    "理解": "平静", "同情": "平静", "帮助": "平静", "信任": "平静", "镇定": "平静", "安心": "平静",
    "冷静": "平静", "淡定": "平静", "平和": "平静", "宁静": "平静", "安宁": "平静",
    "放松": "平静", "舒适": "平静", "安静": "平静", "温暖": "平静", "温馨": "平静", "沉着": "平静",
    "踏实": "平静", "从容": "平静", "悠闲": "平静", "困倦": "平静", "想睡": "平静", "打哈欠": "平静",
}

# ============== 从 txt 解析为结构化 JSON（供 Step C 使用） ==============
def parse_stepA_txt(txt: str, book_id: str, total_pages: int, enforce_visual_filter: bool=False) -> Dict[str, Any]:
    if not txt:
        raise ValueError("空响应文本，无法解析。")

    m = re.search(r"==BOOK_SUMMARY==\s*(.+?)\s*==PAGE_SUMMARIES==", txt, flags=re.S)
    if not m:
        raise ValueError("未找到 BOOK_SUMMARY 块。")
    book_summary = m.group(1).strip()

    m2 = re.search(r"==PAGE_SUMMARIES==\s*(.+?)\s*(?:==END==|$)", txt, flags=re.S)


    if not m2:
        raise ValueError("未找到 PAGE_SUMMARIES 块。")
    body = m2.group(1)

    blocks = re.split(r"^\s*#\s*第(\d+)页\s*$", body, flags=re.M)
    page_summaries: List[Dict[str, Any]] = []

    for i in range(1, len(blocks), 2):
        try:
            pg = int(blocks[i])
        except Exception:
            continue
        content = blocks[i + 1]

        sem = re.search(r"语义摘要：(.+)", content)
        cand = re.search(r"候选情感：(.+)", content)
        ev = re.search(r"文本线索：(.+)", content)
        vis = re.search(r"图像线索：(.+)", content)

        semantic = sem.group(1).strip() if sem else ""

        # 候选情感规范化
        cand_text = cand.group(1).strip() if cand else ""
        sep_norm = cand_text.replace("、", ",").replace("，", ",").replace("；", ",").replace(";", ",")
        tokens = [t.strip() for t in sep_norm.split(",") if t.strip()]
        norm_cands: List[str] = []
        for w in tokens:
            w0 = re.sub(r"\(.*?\)", "", w).strip()
            base = SYN_MAP.get(w0, w0)
            if base not in ALLOWED and w0 in ALLOWED:
                base = w0
            if base not in ALLOWED:
                mparen2 = re.match(r"^([快乐悲伤恐惧愤怒惊讶平静好奇]+)\s*$", w0)
                if mparen2:
                    base = mparen2.group(1)
            if base in ALLOWED:
                norm_cands.append(base)
        # 去重限3
        seen = set(); cands=[]
        for e in norm_cands:
            if e not in seen:
                seen.add(e); cands.append(e)
        cands = cands[:3]

        ev_list = [x.strip() for x in (ev.group(1).strip() if ev else "").replace("、", ",").replace("，", ",").split(",")
                   if x.strip()]
        vis_list = [x.strip() for x in
                    (vis.group(1).strip() if vis else "").replace("、", ",").replace("，", ",").split(",") if x.strip()]

        # （可选）轻量过滤：删除明显“文本化/主观化”的图像线索
        if enforce_visual_filter and vis_list:
            # 规则：含有以下词根之一则去掉（你可根据数据再补充）
            banned = ["神秘", "兴趣盎然", "可爱", "开心", "难过", "害怕", "愤怒", "惊讶", "平静", "好奇", "想要", "希望", "决定", "打算"]
            def looks_textual(cue: str) -> bool:
                # 1) 出现常见主观/情绪词；2) 明显剧情/意图动词
                return any(b in cue for b in banned)
            vis_list = [c for c in vis_list if not looks_textual(c)]

        page_summaries.append({
            "page": pg,
            "semantic": semantic,
            "emotion_candidates": [{"label": e} for e in cands],
            "text_cues": ev_list,
            "visual_cues": vis_list
        })

    # 补齐缺页
    have = {x["page"] for x in page_summaries}
    for p in range(1, total_pages + 1):
        if p not in have:
            page_summaries.append({"page": p, "semantic": "", "emotion_candidates": [], "keywords": [], "visual_cues": []})
    page_summaries.sort(key=lambda x: x["page"])


    return {
        "book_id": book_id,
        "book_summary": book_summary,
        "page_summaries": page_summaries,
    }

# ============== 构建输入（注入视觉线索规范块） ==============
def build_inputs(
    json_path: Path,
    data: Any,
    pages: List[Dict[str, Any]],
    prompt_head: str,
    prompt_tail: str,
    allow_missing: bool,
    missing_list: List[str],
    model: str,
    include_visual_style_guide: bool = True,
    image_detail_stepA: str = "auto",
    max_image_side_stepA: int = 512,
):
    parts = []
    if prompt_head:
        parts.append({"type": "input_text", "text": prompt_head})
        # 受控注入：视觉线索风格引导
        if include_visual_style_guide:
            parts.append({"type": "input_text", "text": VISUAL_CUES_STYLE_GUIDE})

    picture_dir_abs = resolve_picture_dir(json_path, data)

    for p in pages:
        parts.append({"type": "input_text", "text": f"【第{p['page']}页 - 文字】\n{p['text'].strip()}\n"})
        raw = p["image_path"]
        if raw:
            ip = resolve_image_path(raw, json_path, picture_dir_abs)
            if ip.exists():
                image_ref = build_image_ref_for_model(
                    ip,
                    picture_dir_abs,
                    model,
                    max_side=max_image_side_stepA,
                    data_url_builder=downscale_to_data_url,
                )
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": image_ref,
                        "detail": image_detail_stepA,
                    }
                )
            else:
                if not allow_missing:
                    raise FileNotFoundError(f"图片不存在: {ip}（原始字段：{raw}）")
                parts.append({"type": "input_text", "text": f"(提示：第{p['page']}页图片缺失：{ip}；原始：{raw})"})
                missing_list.append(f"- 第{p['page']}页: {ip}（原始：{raw}）")
        else:
            if allow_missing:
                parts.append({"type": "input_text", "text": f"(提示：第{p['page']}页未提供图片路径)"})
                missing_list.append(f"- 第{p['page']}页: （未提供图片字段）")

    if prompt_tail:
        parts.append({"type": "input_text", "text": prompt_tail})
    return [{"role": "user", "content": parts}]

# ============== 调用与重试 ==============
def call_openai_compat(client, *, model: str, messages, temperature: float = 0.2,
                       reasoning_effort: str | None = None,
                       text_verbosity: str | None = None):
    """
    统一适配：
    - gemini-*      -> Chat Completions（OpenAI compatibility）
    - gpt-5*        -> Responses API
    - 其它 GPT 模型 -> Chat Completions
    """

    def _is_chat_msgs(x):
        return isinstance(x, list) and x and isinstance(x[0], dict) and ("role" in x[0])

    def _is_responses_items(x):
        return isinstance(x, list) and x and isinstance(x[0], dict) and \
               isinstance(x[0].get("type"), str) and x[0]["type"].startswith("input_")

    def _extract_url(url_like):
        if isinstance(url_like, dict):
            return (
                url_like.get("url")
                or url_like.get("image_url")
                or url_like.get("data")
                or url_like.get("src")
            )
        return url_like

    def _normalize_to_chat_messages(msgs):
        # 已经是 chat messages
        if _is_chat_msgs(msgs):
            out = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")

                if isinstance(content, str):
                    out.append({"role": role, "content": content})
                    continue

                parts = []
                for p in (content or []):
                    t = p.get("type")
                    if t in ("text", "input_text"):
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif t in ("image_url", "input_image"):
                        url = _extract_url(p.get("image_url") or p.get("url"))
                        if url:
                            parts.append({"type": "image_url", "image_url": {"url": url}})
                out.append({"role": role, "content": parts if parts else ""})
            return out

        # Responses items
        if _is_responses_items(msgs):
            parts = []
            for it in msgs:
                t = it.get("type")
                if t == "input_text":
                    parts.append({"type": "text", "text": it.get("text", "")})
                elif t == "input_image":
                    url = _extract_url(it.get("image_url") or it.get("url"))
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
            return [{"role": "user", "content": parts if parts else ""}]

        # 兜底
        return [{"role": "user", "content": [{"type": "text", "text": str(msgs)}]}]

    def _chat_content_to_responses_items(content):
        items = []
        for p in (content or []):
            t = p.get("type")
            if t in ("text", "input_text"):
                items.append({"type": "input_text", "text": p.get("text", "")})
            elif t in ("image_url", "input_image"):
                url = _extract_url(p.get("image_url") or p.get("url"))
                if url:
                    items.append({"type": "input_image", "image_url": url})
        return items

    api_model = resolve_api_model_name(model)

    # ===== Gemini：统一走 Chat Completions =====
    if is_gemini_model(model):
        chat_messages = _normalize_to_chat_messages(messages)
        kwargs = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    # ===== GPT-5：走 Responses =====
    if str(model).startswith("gpt-5"):
        input_items = []

        if _is_chat_msgs(messages):
            for m in messages:
                role = m.get("role", "user")
                content_items = _chat_content_to_responses_items(m.get("content", []))
                if content_items:
                    input_items.append({"role": role, "content": content_items})

        elif _is_responses_items(messages):
            input_items = [{"role": "user", "content": messages}]

        else:
            input_items = [{"role": "user", "content": [{"type": "input_text", "text": str(messages)}]}]

        if not input_items:
            input_items = [{"role": "user", "content": [{"type": "input_text", "text": ""}]}]

        kwargs = {"model": api_model, "input": input_items, "temperature": temperature}
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity:
            kwargs["text"] = {"verbosity": text_verbosity}

        resp = client.responses.create(**kwargs)
        return getattr(resp, "output_text", None) or resp.output[0].content[0].text

    # ===== 其它 GPT：走 Chat Completions =====
    chat_messages = _normalize_to_chat_messages(messages)
    resp = client.chat.completions.create(
        model=api_model,
        messages=chat_messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

def call_with_retries(client=None, payload=None, fn=None, max_retries=5, base_delay=2.0, timeout=600):
    """
    通用重试器：
    Accepts either a request callable or a legacy provider payload.
    """
    import random

    last_err = None
    for i in range(max_retries):
        try:
            log(f"📡 发起请求（第{i + 1}/{max_retries}次）…")

            if fn is not None:
                resp = fn()
                return resp

            if not isinstance(payload, dict):
                raise ValueError("call_with_retries: 需要提供 fn= 或 payload=dict(...)")

            if client is None:
                client = OpenAI()

            if "input" in payload or "instructions" in payload:
                resp = client.responses.create(timeout=timeout, **payload)
            elif "messages" in payload:
                resp = client.chat.completions.create(timeout=timeout, **payload)
            else:
                resp = client.responses.create(timeout=timeout, **payload)

            return resp

        except Exception as e:
            last_err = e
            log(f"⚠️ 第{i + 1}次调用失败：{e}")
            if i == max_retries - 1:
                break
            sleep_s = base_delay * (2 ** i) * (0.8 + 0.4 * random.random())
            time.sleep(sleep_s)

    raise last_err or RuntimeError("调用失败次数过多")

# ============== 对外主函数 ==============
def run_step_a(
    book_json: Path,
    model: str = "gpt-4o",
    allow_missing: bool = True,
    dump_payload: bool = False,
    enforce_visual_filter: bool = False,
    reasoning_effort: Optional[str] = None,
    text_verbosity: Optional[str] = None,
    include_visual_style_guide: bool = True,
    dry_run: bool = False,
    image_detail_stepA: str = "auto",
    max_image_side_stepA: int = 512,
    trace_tokens: bool = False,
    out_dir: str | Path = "results/mosaic/standalone",
) -> Dict[str, Any]:



    """
    enforce_visual_filter: 解析后是否启用“轻量图像线索去主观/去文本化过滤”（默认 False）
    """
    data = json.loads(book_json.read_text(encoding="utf-8"))
    pages = normalize_pages(data)
    missing: List[str] = []

    inputs = build_inputs(
        json_path=book_json,
        data=data,
        pages=pages,
        prompt_head=PROMPT_HEAD,
        prompt_tail=PROMPT_TAIL,
        allow_missing=allow_missing,
        missing_list=missing,
        model=model,
        include_visual_style_guide=include_visual_style_guide,
        image_detail_stepA=image_detail_stepA,
        max_image_side_stepA=max_image_side_stepA,
    )

    # Report text-size and image-payload statistics without modifying the request.
    if trace_tokens:
        try:
            items = inputs[0]["content"]
        except Exception:
            items = []

        # 仅统计文本 parts（input_text）
        stepA_chars = sum(
            len(str(p.get("text", "")))
            for p in items
            if isinstance(p, dict) and p.get("type") == "input_text"
        )
        stepA_tokens_est = max(1, stepA_chars // 4)

        # 统计图片 dataURL 长度（input_image）
        img_urls = []
        for p in items:
            if not isinstance(p, dict) or p.get("type") != "input_image":
                continue
            iu = p.get("image_url", "")
            if isinstance(iu, dict):
                img_urls.append(str(iu.get("url", "")))
            else:
                img_urls.append(str(iu))

        img_n = len(img_urls)
        total_url_len = sum(len(u) for u in img_urls)
        max_url_len = max([len(u) for u in img_urls], default=0)
        avg_url_len = (total_url_len // img_n) if img_n else 0

        log(f"🧾 StepA tokens (estimated, text-only): ~{stepA_tokens_est} | chars={stepA_chars} | images={img_n}")
        log(f"🖼️ StepA images: n={img_n} | total_image_url_len={total_url_len} | avg={avg_url_len} | max={max_url_len} | detail={image_detail_stepA}")

    # 可选：写 payload 预览（脱敏/截断版）
    if dump_payload or dry_run:
        out_dir_preview = Path(out_dir)
        out_dir_preview.mkdir(parents=True, exist_ok=True)

        preview = {"model": model, "instructions_head": (INSTRUCTIONS or "")[:400], "input_preview": []}
        img_cnt = 0
        try:
            items = inputs[0]["content"]
        except Exception:
            items = []

        for part in items:
            if isinstance(part, dict) and part.get("type") == "input_text":
                txt = str(part.get("text", ""))
                preview["input_preview"].append({"type": "text", "text": txt[:400] + ("..." if len(txt) > 400 else "")})
            elif isinstance(part, dict) and part.get("type") == "input_image":
                img_cnt += 1
                # 这里的 image_url 在 StepA 里是 str(dataURL) 形态，截断即可
                url_head = str(part.get("image_url", ""))[:120] + "..."
                preview["input_preview"].append({"type": "image", "image_url_head": url_head})

        preview["meta"] = {"num_items": len(items), "num_images": img_cnt, "missing_images": list(missing)}

        preview_path = out_dir_preview / f"{book_json.stem}_STEPA_payload_preview.json"
        preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

        if dry_run:
            log(f"🧪 Dry-run: payload preview saved (no request sent) → {preview_path}")
            return {
                "book_id": book_json.stem,
                "book_summary": "",
                "page_summaries": [],
            }

    client = make_client_for_model(model)

    log(f"▶️ Step A 开始：{book_json.name}（共 {len(pages)} 页）")

    # 固定的重试/超时（Step A 无 CLI 参数）
    _MAX_RETRIES_STEPA = 5
    _TIMEOUT_S = 600

    # Create the output directory selected by --out-dir.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = call_with_retries(
            fn=lambda: call_openai_compat(
                client,
                model=model,
                messages=inputs,
                temperature=0.2,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
            ),
            max_retries=_MAX_RETRIES_STEPA,
            base_delay=2,
            timeout=_TIMEOUT_S
        )

    except Exception as e:
        # —— 请求阶段彻底失败：写状态并早停 ——
        err_path = out_dir / f"{book_json.stem}_STEPA_request_error.txt"
        err_path.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
        log(f"⛔ Step A 请求阶段彻底失败（已写入 {err_path}），触发 Fail-Fast：本书终止。")
        status = RunStatus(
            success=False,
            hard_fail=True,
            reason="stepA_request_fail",
            detail={"error": f"{type(e).__name__}: {e}"}
        )
        save_run_status(book_json.stem, "STEPA", status)
        return {"status": status.__dict__}

    raw = str(raw or "")

    if missing:
        missing_info = "\n\n⚠️ 以下图片在处理过程中未找到/未提供，已跳过：\n" + "\n".join(missing)
        missing_info += "\n请确认路径是否正确或补充图片后重新运行。"
        raw = (raw or "") + missing_info

    raw_path = out_dir / f"{book_json.stem}_STEPA_raw.txt"
    raw_path.write_text(raw or "", encoding="utf-8")
    log(f"📝 已保存原始返回：{raw_path}")

    # —— 请求阶段彻底失败：raw 为空 → 早停 ——
    if not (raw and raw.strip()):
        err_path = out_dir / f"{book_json.stem}_STEPA_parse_error.txt"
        err_path.write_text("empty response text\n\n====RAW====\n", encoding="utf-8")
        log(f"⛔ Step A 无任何返回文本，触发 Fail-Fast：本书后续步骤终止。")
        status = RunStatus(success=False, hard_fail=True, reason="stepA_request_fail")
        save_run_status(book_json.stem, "STEPA", status)
        return {"status": status.__dict__}

    # 先解析
    try:
        parsed = parse_stepA_txt(
            raw or "",
            book_id=book_json.stem,
            total_pages=len(pages),
            enforce_visual_filter=enforce_visual_filter
        )
    except Exception as e:
        err_path = out_dir / f"{book_json.stem}_STEPA_parse_error.txt"
        err_path.write_text(f"{type(e).__name__}: {e}\n\n====RAW====\n{raw or ''}", encoding="utf-8")
        log(f"⛔ Step A 解析失败（已写入 {err_path}），触发 Fail-Fast：本书后续步骤终止。")
        status = RunStatus(success=False, hard_fail=True, reason="stepA_parse_fail")
        save_run_status(book_json.stem, "STEPA", status)
        return {"status": status.__dict__}

    # 再做结构校验（这里 parsed 已经有值）
    required_keys = ("book_id", "book_summary", "page_summaries")
    invalid = (
            not isinstance(parsed, dict)
            or any(k not in parsed for k in required_keys)
            or not parsed.get("page_summaries")
    )
    if invalid:
        err_path = out_dir / f"{book_json.stem}_STEPA_parse_error.txt"
        err_path.write_text(
            "invalid structure after parse\n\n====RAW====\n" + (raw or ""),
            encoding="utf-8"
        )
        log("⛔ Step A 解析结果结构无效，触发 Fail-Fast：本书后续步骤终止。")
        status = RunStatus(success=False, hard_fail=True, reason="stepA_parse_invalid")
        save_run_status(book_json.stem, "STEPA", status)
        return {"status": status.__dict__}

    json_path = out_dir / f"{book_json.stem}_STEPA_information.json"
    json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"✅ StepA 完成：{json_path}")
    # Record successful completion for the current run.
    status = RunStatus(success=True, hard_fail=False, reason="ok")
    save_run_status(book_json.stem, "STEPA", status)
    return parsed

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser("MOSAIC Step A runner")
    parser.add_argument("--book-json", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results/mosaic/standalone", help="Output directory for Step A artifacts.")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--allow-missing", action="store_true",
                        help="允许缺图时继续（以文字代替提示）")
    parser.add_argument("--dump-payload", action="store_true",
                        help="保存请求前的payload预览文件")
    parser.add_argument("--enforce-visual-filter", action="store_true",
                        help="解析后对视觉线索做轻量去主观/去文本化过滤")
    # Optional input controls.
    parser.add_argument("--dry-run", action="store_true",
                        help="只生成payload预览，不发送请求")
    parser.add_argument("--no-visual-style-guide", action="store_true",
                        help="不注入 VISUAL_CUES_STYLE_GUIDE")
    parser.add_argument("--image-detail-stepA", choices=["auto", "low", "high"], default="auto",
                        help="Step A 图片 detail 预算（auto/low/high）")
    parser.add_argument("--max-image-side-stepA", type=int, default=512,
                        help="Step A 图片压缩最大边长（像素），影响 dataURL 大小")
    parser.add_argument("--trace-tokens", action="store_true",
                        help="打印 Step A 的 tokens/图片体积统计（与主控 --trace-tokens 共用同名开关）")

    # Optional provider-specific generation controls.
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"])
    parser.add_argument("--text-verbosity", type=str, default=None,
                        choices=["low", "medium", "high"])

    args = parser.parse_args()

    run_step_a(
        book_json=Path(args.book_json),
        model=args.model,
        allow_missing=args.allow_missing,
        dump_payload=args.dump_payload,
        enforce_visual_filter=args.enforce_visual_filter,
        reasoning_effort=args.reasoning_effort,
        text_verbosity=args.text_verbosity,
        include_visual_style_guide=(not args.no_visual_style_guide),
        dry_run=args.dry_run,
        image_detail_stepA=args.image_detail_stepA,
        max_image_side_stepA=args.max_image_side_stepA,
        trace_tokens=args.trace_tokens,
        out_dir=args.out_dir,
    )
