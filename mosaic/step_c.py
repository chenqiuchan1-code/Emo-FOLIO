# -*- coding: utf-8 -*-
"""MOSAIC Step C: stage-aware local page-emotion estimation.

Step B stages define ordered local windows. Each window is evaluated with its
page content and selected Step A cues; a continuity anchor carries intensity
references from the preceding window. The final output is the concatenated
page-level emotion-intensity sequence in the original page order.
"""

import os
import re
import io
import json
import time
import base64
from urllib.parse import quote
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from mosaic.runtime import RunStatus, save_run_status


try:
    from PIL import Image
except Exception:
    Image = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

EMOS = ["快乐", "悲伤", "恐惧", "愤怒", "惊讶", "平静", "好奇"]

def _is_effective_model_output(text: str) -> bool:
    """返回 True 表示看起来像有效输出；否则视为无效，需要重试。"""
    if not text:
        return False
    s = text.strip()
    if not s:
        return False
    # 只含控制字符/空白
    if s.strip("\r\n\t ") == "":
        return False
    # 只要出现任一情绪词或 0–5 数字，就算“有内容”
    has_emo = any(e in s for e in EMOS)
    has_digit = any(ch in s for ch in "012345")
    return has_emo or has_digit

def _is_model_refusal(text: str) -> bool:
    """
    检测典型拒答（中英文都有）。拒答通常很短，且不包含任何我们期望的输出锚点。
    命中后应立即停止重试，避免额外费用。
    """
    if not text:
        return False
    s = text.strip()
    if not s:
        return False

    # 有明确输出锚点则不判拒答（避免误伤“格式不标准但有分数”的情况）
    anchor_tokens = ["==PAGE_SCORES==", "第1页", "第2页", "快乐", "悲伤", "恐惧", "愤怒", "惊讶", "平静", "好奇"]
    if any(tok in s for tok in anchor_tokens):
        return False

    # 拒答常见关键词（中英）
    refusal_patterns = [
        r"\b(i\s*(?:can\'t|cannot)\s*(?:help|assist|comply|provide))\b",
        r"\b(i\'m\s*sorry|sorry)\b",
        r"\b(not\s*able\s*to)\b",
        r"\b(refuse|refusal)\b",
        r"无法(协助|帮助|处理|提供|完成)",
        r"不能(协助|帮助|处理|提供|完成)",
        r"抱歉.*(无法|不能)",
        r"对不起.*(无法|不能)",
        r"我不能.*(提供|协助|帮助)",
    ]
    hit = any(re.search(pat, s, flags=re.IGNORECASE) for pat in refusal_patterns)

    # 再加一个“短文本”约束，降低误判
    return hit and (len(s) <= 500)



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

# ---------- 兼容 Step A 产物：把 list[str|dict] 规整为 list[str] ----------
def _to_str_list(val):
    """
    Normalize possibly mixed (str | dict) lists to List[str].
    Accepts dict items like {"label": "..."} or {"text": "..."} or {"cue":"..."} etc.
    """
    out = []
    if isinstance(val, list):
        for x in val:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
            elif isinstance(x, dict):
                for key in ("label", "text", "cue", "name", "value"):
                    if key in x and isinstance(x[key], (str, int, float)):
                        s = str(x[key]).strip()
                        if s:
                            out.append(s)
                        break
    return out

def _make_zero_raw_for_window(start: int, end: int) -> str:
    lines = ["==PAGE_SCORES=="]
    for i_pg in range(start, end + 1):
        lines.append(f"第{i_pg}页：快乐=0，悲伤=0，恐惧=0，愤怒=0，惊讶=0，平静=0，好奇=0")
    lines.append("==END==")
    return "\n".join(lines)



def _build_prompt_head_for_step2(
    include_book_summary: bool,
    include_anchor_hint: bool,
    include_s1_summary: bool,
    include_s1_candidates: bool,
    include_s1_text_cues: bool,
    include_s1_visual_cues: bool,
    include_page_images: bool,
    include_anchor_images: bool,
) -> str:
    # 仅说明“输入块结构”，不重复任务/规则/输出格式（这些由 INSTRUCTIONS + PROMPT_TAIL 统一负责）
    lines = []
    lines.append("==INPUT_BLOCKS==")
    idx = 1

    if include_book_summary:
        lines.append(f"{idx}) ==BOOK_SUMMARY==：整书剧情摘要（仅作背景）")
        idx += 1

    lines.append(f"{idx}) ==WINDOW_META==：本窗口页码信息")
    idx += 1

    desc = ["逐页文字"]
    if include_s1_summary:
        desc.append("StepA摘要")
    if include_s1_candidates:
        desc.append("候选情感")
    if include_s1_text_cues:
        desc.append("文本线索")
    if include_s1_visual_cues:
        desc.append("图像线索")
    if include_page_images:
        desc.append("原始图片")

    lines.append(f"{idx}) ==WINDOW_CONTENT==：逐页材料（" + "、".join(desc) + "）")
    idx += 1

    if include_anchor_hint:
        if include_anchor_images:
            lines.append(f"{idx}) ==ANCHOR_HINT==：上一窗口末尾两页的文字/线索/图片参考与已确定强度（仅作连续性比对，不可改写）")
        else:
            lines.append(f"{idx}) ==ANCHOR_HINT==：上一窗口末尾两页的文字/线索参考与已确定强度（仅作连续性比对，不可改写）")

    return "\n".join(lines)

def _merge_short_stepb_chunks(
    chunks: List[Tuple[int, int, str]],
    short_len: int = 2,
) -> List[Tuple[int, int, str]]:
    """
    仅供 StepC 评分使用：把页数 <= short_len 的 StepB 短块并入相邻块。
    规则：
    - 首块短：向下并
    - 尾块短：向上并
    - 中间短块：优先并到“合并后总页数更少”的一侧；若相等，优先向下并
    说明：
    - 这里只改变 StepC 的 scoring chunks，不改 StepB 原始 segments
    - 合并后 intent 保留“被并入的一侧”的 intent；该字段当前仅随块携带，StepC 不依赖它判分
    """
    out = list(chunks or [])
    if len(out) <= 1:
        return out

    i = 0
    while i < len(out):
        s, e, intent = out[i]
        cur_len = e - s + 1

        if cur_len > short_len:
            i += 1
            continue

        if len(out) == 1:
            break

        # 首块短：只能向下并
        if i == 0:
            rs, re, r_intent = out[1]
            out[1] = (min(s, rs), max(e, re), r_intent)
            del out[0]
            i = 0
            continue

        # 尾块短：只能向上并
        if i == len(out) - 1:
            ls, le, l_intent = out[i - 1]
            out[i - 1] = (min(ls, s), max(le, e), l_intent)
            del out[i]
            i = max(i - 1, 0)
            continue

        # 中间短块：比较左右两侧合并后的总页数；相等时优先向下并
        ls, le, l_intent = out[i - 1]
        rs, re, r_intent = out[i + 1]

        left_total = (le - ls + 1) + cur_len
        right_total = (re - rs + 1) + cur_len

        if right_total <= left_total:
            out[i + 1] = (min(s, rs), max(e, re), r_intent)
            del out[i]
            i = max(i - 1, 0)
        else:
            out[i - 1] = (min(ls, s), max(le, e), l_intent)
            del out[i]
            i = max(i - 1, 0)

    return out

def _make_fixed_chunks(N: int, window_size: int) -> List[Tuple[int, int, str]]:
    """
    固定窗口切分：
    - 先按不重叠窗口切
    - 若最后余数 <= floor(window_size / 2)，并入最后一个完整窗口
    - 否则余数单独形成最后一个窗口
    """
    if N <= 0:
        return []

    k = max(1, int(window_size))
    full = N // k
    rem = N % k

    if full == 0:
        return [(1, N, "")]

    chunks: List[Tuple[int, int, str]] = []
    for i in range(full):
        s = i * k + 1
        e = (i + 1) * k
        chunks.append((s, e, ""))

    if rem == 0:
        return chunks

    if rem <= max(1, k // 2):
        s, _, intent = chunks[-1]
        chunks[-1] = (s, N, intent)
    else:
        chunks.append((full * k + 1, N, ""))

    return chunks

# ============== 工具：图片下采样为 data URL ==============
def downscale_to_data_url(img_path: Path, max_side: int = 512) -> str:
    if (Image is None) or (not img_path.exists()):
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
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


# ============== 解析图片目录 ==============
def resolve_picture_dir(json_path: Path, data: Dict[str, Any]) -> Optional[Path]:
    """
    优先使用 book JSON 里的 picture_dir（相对/绝对均可）。
    若缺失，则兜底尝试 data/images/<book_stem>/ 目录。
    找不到则返回 None（下游再用其它候选路径去找单张图片）。
    """
    json_dir = json_path.parent
    # 1) 优先 picture_dir 字段
    picture_dir_val = data.get("picture_dir") or data.get("pictures_dir")
    if picture_dir_val:
        pd = Path(picture_dir_val)
        if not pd.is_absolute():
            pd = (json_dir / pd).resolve()
        if pd.exists():
            return pd
        # 再试一层上级
        pd2 = (json_dir.parent / picture_dir_val).resolve()
        if pd2.exists():
            return pd2
        log(f"⚠️ 指定的 picture_dir 不存在：{pd} / {pd2}")

    # 2) 兜底：data/images/<book_stem>/
    fallback = (json_dir.parent / "images" / json_path.stem).resolve()
    if fallback.exists():
        return fallback

    return None


# ============== 解析图片路径 ==============
def resolve_image_path(raw_img: str, json_path: Path, picture_dir_abs: Optional[Path]) -> Path:
    """
    在以下候选中按顺序查找，返回第一个存在的路径：
    - 若 raw_img 为绝对路径：直接返回
    - 若给出了 picture_dir_abs：<picture_dir_abs>/<raw_img>
    - <json_dir>/<raw_img>
    - <json_dir.parent>/<raw_img>
    - 若能从 raw_img 名字末尾解析出页码 n：尝试在 picture_dir_abs 下的
      <book_stem>_<n>.(jpg|jpeg|png|webp) 这些常见命名
    找不到则返回最靠前的候选（便于错误日志定位）
    """
    p = Path(raw_img)
    if p.is_absolute():
        return p

    json_dir = json_path.parent
    candidates = []

    # 1) 直接拼接相对路径
    if picture_dir_abs:
        candidates.append((picture_dir_abs / p).resolve())
    candidates.append((json_dir / p).resolve())
    candidates.append((json_dir.parent / p).resolve())

    # 2) 若能从文件名推断出页码，补充常见命名猜测
    page_num = None
    try:
        # 例如 book_2_13.jpg / 13.png / page_07.jpeg -> 取末尾数字
        page_num = int(Path(raw_img).stem.split("_")[-1])
    except Exception:
        page_num = None

    if picture_dir_abs and page_num:
        stem = json_path.stem
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidates.append((picture_dir_abs / f"{stem}_{page_num}{ext}").resolve())

    # 3) 返回第一个存在的候选
    for c in candidates:
        if c.exists():
            return c

    # 4) 都没有就返回第一个候选（便于报错定位）
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


# ============== 解析窗口返回文本（放宽到“快乐=2 / 快乐2”都可） ==============
def parse_window_txt(txt: str) -> Dict[str, Any]:
    EMOS = ["快乐", "悲伤", "恐惧", "愤怒", "惊讶", "平静", "好奇"]
    scores: List[Dict[str, Any]] = []

    s = txt.strip()

    # --- 先尝试解析标准格式（向后兼容你原来的正则） ---
    std_found = False
    for m in re.finditer(r"第\s*(\d+)\s*页[:：]?\s*(.+)", s):
        p = int(m.group(1))
        rest = m.group(2)
        emo_map = {}
        for e in EMOS:
            # 兼容 "快乐=2" / "快乐：2" / "快乐 2"
            m2 = re.search(fr"{e}\s*(?:=|:|：)?\s*([0-5])", rest)
            if m2:
                emo_map[e] = int(m2.group(1))
        if emo_map:
            scores.append({"page": p, "emotions": emo_map})
            std_found = True
    if std_found and scores:
        return {"scores": scores}

    # --- 若未匹配到标准格式，再尝试“分块格式” ---
    # 将连续非空行归为一块；块首行是页码（如 "1" / "第1页"）
    lines = [ln.strip() for ln in s.splitlines()]
    blocks: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if ln == "":
            if cur:
                blocks.append(cur); cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)

    def parse_emoline(ln: str) -> Optional[Tuple[str, int]]:
        # 兼容 "快乐 4" / "快乐:4" / "快乐：4" / "快乐=4"
        m = re.match(r"^(快乐|悲伤|恐惧|愤怒|惊讶|平静|好奇)\s*(?:=|:|：)?\s*([0-5])\s*$", ln)
        if m:
            return m.group(1), int(m.group(2))
        return None

    parsed_any = False
    for block in blocks:
        if not block:
            continue
        # 头部识别页码
        head = block[0]
        m_head = re.match(r"^(?:第\s*)?(\d+)\s*(?:页)?\s*$", head)
        if not m_head:
            # 不是页块，跳过
            continue
        p = int(m_head.group(1))

        emo_map: Dict[str, int] = {}
        for ln in block[1:]:
            r = parse_emoline(ln)
            if r:
                emo_map[r[0]] = r[1]

        if emo_map:
            # 补全缺失情绪为0（可选；若不想补全，删除这段）
            for e in EMOS:
                emo_map.setdefault(e, 0)
            scores.append({"page": p, "emotions": emo_map})
            parsed_any = True

    if parsed_any and scores:
        # 保序
        scores.sort(key=lambda x: x["page"])
        return {"scores": scores}

    # 两路都失败
    raise ValueError("未能解析出任何页的情感强度（不符合标准行或分块行格式）")


# ============== Prompt（与原风格一致，并追加尾部提示确保格式） ==============
def _build_instructions_for_step2(
    include_anchor_hint: bool,
    include_page_images: bool,
    include_anchor_images: bool,
) -> str:
    parts = [
        "【INSTRUCTIONS】",
        "你将进行**上下文感知**的页级情绪强度评估：对“当前窗口”内每一页标注七类情绪（快乐/悲伤/恐惧/愤怒/惊讶/平静/好奇）的 0–5 整数强度，不存在即为 0。"
    ]

    if include_anchor_hint:
        if include_anchor_images:
            parts.append("评分需结合**上一窗口锚点**与**相邻页内容**，使强度能**反映自然的变化趋势**（无依据勿剧烈跳变）。")
        else:
            parts.append("评分需结合**上一窗口锚点提供的文字/线索**与**相邻页内容**，使强度能**反映自然的变化趋势**（无依据勿剧烈跳变）。")
    else:
        parts.append("评分需结合**相邻页内容**，使强度能**反映自然的变化趋势**（无依据勿剧烈跳变）。")

    if include_page_images:
        parts.append("以当前页提供的文字、StepA线索和原始图片为主要证据。")
    else:
        parts.append("以当前页提供的文字与StepA线索为主要证据；若未提供原始图片，不要臆测图片细节。")

    parts.append("只按输出格式作答，不要额外说明或 Markdown。")
    return "\n".join(parts)


PROMPT_HEAD = (
    "【TASK】\n"
    "根据输入材料，为本窗口内每一页打分。\n\n"
    "【输入块顺序（下方依次出现）】\n"
    "1) ==BOOK_SUMMARY==：整书摘要（来自 Step A）。随后会有 ==WINDOW_META==（start/end/k，仅作参考）。\n"
    "2) ==WINDOW_CONTENT==：本窗口每页的文字、（若有）图片，以及 Step A 提供的摘要/线索/候选情感。\n"
    "3) ==ANCHOR_HINT==…==END_ANCHOR==：上一窗口最后两页的参考（文字/图片/已确定强度），只用于连续性比对，强度不可改写。\n\n"
    "【RULES】\n"
    "A. 连续性：若无明显转折，本窗口第一页应与锚点最后一页自然衔接；有显著转折时可合理跳变。\n"
    "B. 证据优先：以“当前页图文”为主，摘要/锚点仅作校准，避免机械沿用锚点强度。\n"
    "C. 锚点既定：锚点页强度是既定事实，不得修改或写入结果。\n"
    "D. 逐页演化：窗口内后续页与上一页保持可解释的变化，避免无依据的大幅波动。\n"
    "E. 标签语义要准确：“平静”只在页面明确呈现安抚、稳定、放松、被安定下来、平和叙述或情绪落定时给分；不要把“平静”当作其他情绪不够强时的默认安全标签。\n"
    "F. 区分“好奇/惊讶”：“好奇”偏向对未知事物的探索、提问、期待与试探；“惊讶”偏向对已经发生且超出预期之事的即时反应。两者可以共存，但只有同页同时存在这两类直接证据时才同时给分，不要机械捆绑。\n"
)

PROMPT_TAIL = (
    "【输出格式】仅输出以下内容：\n"
    "==PAGE_SCORES==\n"
    "第{start}页：快乐=0，悲伤=0，恐惧=0，愤怒=0，惊讶=0，平静=0，好奇=0\n"
    "...（直到本窗口最后一页，每页一行）\n"
    "==END=="
)

def call_openai_compat(client, *, model: str, messages,
                       temperature: float = 0.2,
                       reasoning_effort: Optional[str] = None,
                       text_verbosity: Optional[str] = None):
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

        return [{"role": "user", "content": [{"type": "text", "text": str(msgs)}]}]

    def _chat_content_to_responses_items(content):
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]

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
        kwargs = {"model": api_model, "messages": chat_messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
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
                content_items = _chat_content_to_responses_items(m.get("content", ""))
                if content_items:
                    input_items.append({"role": role, "content": content_items})

        elif _is_responses_items(messages):
            input_items = [{"role": "user", "content": messages}]

        else:
            input_items = [{"role": "user", "content": [{"type": "input_text", "text": str(messages)}]}]

        if not input_items:
            input_items = [{"role": "user", "content": [{"type": "input_text", "text": ""}]}]

        kwargs = {"model": api_model, "input": input_items}
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity:
            kwargs["text"] = {"verbosity": text_verbosity}

        resp = client.responses.create(**kwargs)
        return getattr(resp, "output_text", None) or resp.output[0].content[0].text

    # ===== 其它 GPT：走 Chat Completions =====
    chat_messages = _normalize_to_chat_messages(messages)
    kwargs = {"model": api_model, "messages": chat_messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

# ============== OpenAI Responses 兼容封装（按块重置重试计数） ==============
def call_with_retries(client=None, payload=None, fn=None, max_retries: int = 5, base_delay: int = 2, timeout: int = 600):
    last_err = None
    for i in range(max_retries):
        try:
            log(f"📡 发起请求（第{i + 1}/{max_retries}次）…")

            if fn is not None:
                return fn()

            if not isinstance(payload, dict):
                raise ValueError("call_with_retries: 需要提供 fn= 或 payload=dict(...)")

            if client is None:
                client = OpenAI()

            if "input" in payload or "instructions" in payload:
                return client.responses.create(timeout=timeout, **payload)
            elif "messages" in payload:
                return client.chat.completions.create(timeout=timeout, **payload)
            else:
                return client.responses.create(timeout=timeout, **payload)

        except Exception as e:
            last_err = e
            log(f"⚠️ 第{i + 1}次调用失败：{e}")
            if i == max_retries - 1:
                break
            time.sleep(base_delay * (2 ** i))

    raise last_err or RuntimeError("调用失败次数过多")


# ============== 主函数（对外签名保持不变） ==============
def run_step_c(
    book_json: Path,
    stepA_json: Optional[Path],
    stepB_json: Optional[Path],
    model: str = "gpt-4o",
    dump_payload: bool = False,
    out_dir: str | Path = "results/mosaic/standalone",
    out_dir_debug: str | Path | None = None,
    reuse_existing_windows: bool = False,
    reasoning_effort: Optional[str] = None,
    text_verbosity: Optional[str] = None,
    max_retries_stepC: int = 3,
    image_detail_stepC: str = "auto",
    max_image_side_stepC: int = 512,
    trace_tokens: bool = False,
    *,
    inject_page_text: bool = True,
    inject_page_images: bool = True,
    allow_missing_images: bool = True,
    emit_missing_image_placeholders: bool = False,
    include_window_intent: bool = False,
    include_anchor_hint: bool = True,
    include_anchor_images: bool = False,
    include_anchor_s1_cues: bool = False,
    inject_book_summary: bool = False,
    inject_s1_summary: bool = False,
    inject_s1_candidates: bool = False,
    inject_s1_text_cues: bool = True,
    inject_s1_visual_cues: bool = True,
    include_prompt_head: bool = False,
    chunk_mode: str = "stepb_windows",
    window_size_stepC: int = 8,
    dry_run: bool = False,
) -> Dict[str, Any]:


    data = json.loads(book_json.read_text(encoding="utf-8"))
    pages: List[Dict[str, Any]] = []
    for i, pg in enumerate(data.get("pages", []), 1):
        pages.append({
            "page": pg.get("page", i),
            "text": (pg.get("text") or "").strip(),
            "image_path": pg.get("image_path") or pg.get("image") or pg.get("img") or "",
        })
    N = len(pages)

    # StepA / StepB 读取（A2/A3/A4 允许为空）
    stepA_obj: Dict[str, Any] = {}
    if stepA_json and Path(stepA_json).exists():
        stepA_obj = json.loads(Path(stepA_json).read_text(encoding="utf-8"))

    stepB_obj: Dict[str, Any] = {}
    if stepB_json and Path(stepB_json).exists():
        stepB_obj = json.loads(Path(stepB_json).read_text(encoding="utf-8"))

    # StepA 摘要映射（页→{semantic, emotion_candidates, text_cues, visual_cues}）
    s1_map: Dict[int, Dict[str, Any]] = {}
    for it in stepA_obj.get("page_summaries", []) or []:
        try:
            p = int(it.get("page", 0))
            if p > 0:
                s1_map[p] = it
        except Exception:
            pass
    book_summary = (stepA_obj.get("book_summary") or "").strip()

    # StepC chunk mode
    chunk_mode = str(chunk_mode or "stepb_windows").strip().lower()
    if chunk_mode not in {"stepb_windows", "whole_book", "fixed_windows"}:
        raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")

    stepb_chunks: List[Tuple[int, int, str]] = []

    if chunk_mode == "whole_book":
        if N <= 0:
            raise RuntimeError("book_json 中 pages 为空：无法进行 StepC whole-book 评分。")
        stepb_chunks = [(1, N, "")]
        chunks = list(stepb_chunks)

    elif chunk_mode == "fixed_windows":
        if N <= 0:
            raise RuntimeError("book_json 中 pages 为空：无法进行 StepC fixed-window 评分。")
        stepb_chunks = []
        chunks = _make_fixed_chunks(N, window_size_stepC)

    else:
        for seg in (stepB_obj.get("segments") or []):
            try:
                a = int(seg.get("start"))
                b = int(seg.get("end"))
                if a > b:
                    a, b = b, a
                a = max(1, min(N, a))
                b = max(1, min(N, b))
                intent = str(seg.get("intent") or "").strip()
                stepb_chunks.append((a, b, intent))
            except Exception:
                continue
        stepb_chunks.sort(key=lambda x: (x[0], x[1]))

        if not stepb_chunks:
            raise RuntimeError("StepB segments 为空：无法进行 StepC 按段落切窗。请先确认 StepB 输出是否正常。")

        chunks = _merge_short_stepb_chunks(stepb_chunks, short_len=2)

    # Shared chunk representation used throughout Step C.
    max_retries_step2 = int(max_retries_stepC)
    image_detail_step2 = str(image_detail_stepC)
    max_image_side_step2 = int(max_image_side_stepC)

    picture_dir_abs = resolve_picture_dir(book_json, data)

    if (OpenAI is None) and (not dry_run):
        raise RuntimeError("OpenAI SDK 未安装或不可用")

    client = make_client_for_model(model) if not dry_run else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(out_dir_debug) if out_dir_debug is not None else out_dir
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ✅ trace：Step C 全局统计累加器（不写入任何 JSON，仅用于控制台输出）
    _trace_step2_global_chars = 0
    _trace_step2_global_img_n = 0
    _trace_step2_global_total_url_len = 0
    _trace_step2_global_max_url_len = 0
    _trace_step2_global_windows = 0

    # Construct chunks according to the selected Step C mode.
    if chunk_mode == "whole_book" and (include_anchor_hint or include_anchor_images or include_anchor_s1_cues):
        log("ℹ️ StepC 当前为 whole_book 模式：anchor 相关设置将自动失效。")

    # 开始日志（与 Step A 风格对齐；请求次数在 call_with_retries 内按块单独计数）
    mode_cn = "整书单块" if chunk_mode == "whole_book" else "按 StepB 分块"
    if chunk_mode == "whole_book":
        log(f"▶️ Step C 开始：{book_json.name}（共 {N} 页，{mode_cn}，分块 {len(chunks)}）")
    else:
        log(
            f"▶️ Step C 开始：{book_json.name}（共 {N} 页，{mode_cn}，"
            f"StepB段 {len(stepb_chunks)} -> 评分块 {len(chunks)}）"
        )

    raw_scores: Dict[int, List[Dict[str, Any]]] = {}
    anchor_pages: Optional[List[int]] = None  # 上一块末尾两页
    # Window-level resume: 一旦遇到失败窗口（parse_error 或 raw 不可解析），从该窗口起以及后续窗口强制重跑
    rerun_from_here = False

    # ====== 按块调用（或回退后的窗口列表）======
    for start, end, seg_intent in chunks:
        # ---------- Window-level reuse (no extra cost) ----------
        raw_path = debug_dir / f"{book_json.stem}_STEPC_win_{start}_{end}_raw.txt"
        errp = debug_dir / f"{book_json.stem}_STEPC_win_{start}_{end}_parse_error.txt"

        if reuse_existing_windows and (not dry_run):
            # 若该窗口存在 parse_error：认为这是上次失败点 -> 从这里起强制重跑
            if errp.exists():
                rerun_from_here = True

            # 如果还没进入“强制重跑区间”且 raw 存在：直接解析 raw，复用窗口结果
            if (not rerun_from_here) and raw_path.exists():
                try:
                    raw_cached = raw_path.read_text(encoding="utf-8")
                    parsed_cached = parse_window_txt(raw_cached)
                    rows = parsed_cached.get("scores") or []
                    rows = [
                        r for r in rows
                        if isinstance(r, dict) and "page" in r and start <= int(r["page"]) <= end
                    ]
                    if rows:
                        for row in rows:
                            raw_scores.setdefault(int(row["page"]), []).append(row)
                        log(f"⏭️ 复用窗口结果：{start}-{end}（{len(rows)} 页） -> {raw_path.name}")

                        # 复用也要更新 anchor_pages，保证后续窗口锚点一致
                        last2 = [max(end - 1, start), end] if (end - start + 1) >= 2 else [end]
                        anchor_pages = last2
                        continue
                    else:
                        # raw 存在但解析不到有效行：从该窗口起重跑
                        rerun_from_here = True
                except Exception:
                    # raw 存在但解析异常：从该窗口起重跑
                    rerun_from_here = True
        # -------------------------------------------------------

        # 组输入：Prompt + 全书总结 + 本块每页（文字 + stepA摘要/候选/关键词/视觉线索 + 图片）
        parts = [
            {
                "type": "input_text",
                "text": _build_instructions_for_step2(
                    include_anchor_hint=(include_anchor_hint and chunk_mode != "whole_book"),
                    include_page_images=inject_page_images,
                    include_anchor_images=(include_anchor_images and chunk_mode != "whole_book"),
                ),
            },
        ]

        if include_prompt_head:
            parts.append({
                "type": "input_text",
                "text": PROMPT_HEAD,
            })

        # StepC：输入块结构说明
        parts.append({
            "type": "input_text",
            "text": _build_prompt_head_for_step2(
                include_book_summary=inject_book_summary,
                include_anchor_hint=(include_anchor_hint and chunk_mode != "whole_book"),
                include_s1_summary=inject_s1_summary,
                include_s1_candidates=inject_s1_candidates,
                include_s1_text_cues=inject_s1_text_cues,
                include_s1_visual_cues=inject_s1_visual_cues,
                include_page_images=inject_page_images,
                include_anchor_images=(include_anchor_images and chunk_mode != "whole_book"),
            )
        })

        # Book summary（全局背景，按开关决定是否注入）
        if inject_book_summary and book_summary:
            parts.append({"type": "input_text", "text": f"==BOOK_SUMMARY==\n{book_summary}\n"})

        parts.append({"type": "input_text",
                      "text": f"==WINDOW_META==\nstart_page={start}, end_page={end}, n_pages={end - start + 1}\n"})
        parts.append({"type": "input_text", "text": "==WINDOW_CONTENT=="})

        img_cnt = 0
        for i_pg in range(start, end + 1):
            pg = next(p for p in pages if p["page"] == i_pg)
            s1 = s1_map.get(i_pg, {})

            # StepA cues（归一化为字符串）
            semantic = (s1.get("semantic") or s1.get("summary") or "").strip()
            cand = "、".join(_to_str_list(s1.get("emotion_candidates") or s1.get("candidates") or []))
            tc = "、".join(_to_str_list(s1.get("text_cues") or s1.get("text_clues") or s1.get("keywords") or []))
            vc = "、".join(_to_str_list(s1.get("visual_cues") or s1.get("visual_clues") or []))

            block_lines: List[str] = []
            block_lines.append(f"【第{i_pg}页】")

            if inject_page_text:
                t = (pg.get("text") or "").strip()
                block_lines.append("页面文本：")
                block_lines.append(t if t else "(空)")

            # StepA 线索（按开关注入；全部关闭时不再输出“页面线索”区块）
            cue_lines: List[str] = []
            if inject_s1_summary and semantic:
                cue_lines.append(f"- 语义摘要：{semantic}")
            if inject_s1_candidates and cand:
                cue_lines.append(f"- 候选情感：{cand}")
            if inject_s1_text_cues and tc:
                cue_lines.append(f"- 文本线索：{tc}")
            if inject_s1_visual_cues and vc:
                cue_lines.append(f"- 图像线索：{vc}")
            if cue_lines:
                block_lines.append("")
                block_lines.append("页面线索：")
                block_lines.extend(cue_lines)
            # 图片标签放在文本块最后一行：这样紧跟的 input_image part 就“接”在这行后面
            if inject_page_images:
                block_lines.append("")
                block_lines.append("页面图片：")

            parts.append({"type": "input_text", "text": "\n".join(block_lines).strip()})

            # 若图片存在，紧跟着追加真正的 input_image part
            if inject_page_images:
                raw_img = (pg.get("image_path") or "").strip()
                if raw_img:
                    ip = resolve_image_path(raw_img, book_json, picture_dir_abs)
                    if ip.exists():
                        image_ref = build_image_ref_for_model(
                            ip,
                            picture_dir_abs,
                            model,
                            max_side=max_image_side_step2,
                            data_url_builder=downscale_to_data_url,
                        )
                        parts.append({
                            "type": "input_image",
                            "image_url": image_ref,
                            "detail": image_detail_step2,
                        })

                        img_cnt += 1
                    else:
                        # Missing images are silent unless diagnostic placeholders are enabled.
                        if (allow_missing_images and emit_missing_image_placeholders):
                            parts.append({"type": "input_text", "text": "(图片缺失)"})
                else:
                    if (allow_missing_images and emit_missing_image_placeholders):
                        parts.append({"type": "input_text", "text": "(图片字段为空)"})

        if chunk_mode != "whole_book" and include_anchor_hint and anchor_pages:
            # 在原有 ANCHOR_HINT 的位置，扩充为“参考页内容 + 已确定强度”
            anchor_intro = (
                "==ANCHOR_HINT==\n（以下为上一块末尾两页的文字/线索/图片参考与已确定强度，用于连续性比对）"
                if include_anchor_images
                else
                "==ANCHOR_HINT==\n（以下为上一块末尾两页的文字/线索参考与已确定强度，用于连续性比对）"
            )
            parts.append({"type": "input_text", "text": anchor_intro})

            for ap in anchor_pages:
                # 已确定强度（来自上一块解析结果 raw_scores）
                known_emotions = {}
                if ap in raw_scores and raw_scores[ap]:
                    known_emotions = raw_scores[ap][-1].get("emotions", {})

                # 文字内容（来自原 book JSON）
                pg = next((p for p in pages if p["page"] == ap), None)
                page_text = (pg.get("text") or "").strip() if pg else ""

                # 组装锚点文字卡片
                anchor_card_lines = [
                    f"# 锚点-第{ap}页",
                    f"文字：{page_text if page_text else '(无文字)'}",
                ]

                # —— 锚点页：按开关注入 StepA 摘要 + 线索 ——
                if include_anchor_s1_cues:
                    s1a = s1_map.get(ap, {})  # 复用窗口页同一套 s1_map
                    semantic_a = (s1a.get("semantic") or s1a.get("summary") or "").strip()
                    cand_a = "、".join(_to_str_list(s1a.get("emotion_candidates") or s1a.get("candidates") or []))
                    tc_a = "、".join(
                        _to_str_list(s1a.get("text_cues") or s1a.get("text_clues") or s1a.get("keywords") or []))
                    vc_a = "、".join(_to_str_list(s1a.get("visual_cues") or s1a.get("visual_clues") or []))

                    anchor_cue_lines: List[str] = []
                    if inject_s1_summary and semantic_a:
                        anchor_cue_lines.append(f"- 语义摘要：{semantic_a}")
                    if inject_s1_candidates and cand_a:
                        anchor_cue_lines.append(f"- 候选情感：{cand_a}")
                    if inject_s1_text_cues and tc_a:
                        anchor_cue_lines.append(f"- 文本线索：{tc_a}")
                    if inject_s1_visual_cues and vc_a:
                        anchor_cue_lines.append(f"- 图像线索：{vc_a}")

                    if anchor_cue_lines:
                        anchor_card_lines.append("")
                        anchor_card_lines.append("锚点线索（来自 StepA，仅作连续性校准）：")
                        anchor_card_lines.extend(anchor_cue_lines)

                if known_emotions:
                    emo_line = "已确定强度：" + "，".join([f"{e}={int(known_emotions.get(e, 0))}" for e in EMOS])
                    anchor_card_lines.append(emo_line)

                parts.append({"type": "input_text", "text": "\n".join(anchor_card_lines)})

                # 注入锚点图片（若存在；可通过 include_anchor_images 关闭以降低拒答概率）
                if include_anchor_images and pg and pg.get("image_path"):
                    ip = resolve_image_path(pg["image_path"], book_json, picture_dir_abs)
                    if ip.exists():
                        image_ref = build_image_ref_for_model(
                            ip,
                            picture_dir_abs,
                            model,
                            max_side=max_image_side_step2,
                            data_url_builder=downscale_to_data_url,
                        )
                        parts.append({
                            "type": "input_image",
                            "image_url": image_ref,
                            "detail": image_detail_step2,
                        })
                        img_cnt += 1
                    else:
                        # 与过去 allow_missing=True 的默认行为一致：静默跳过
                        pass

            parts.append({"type": "input_text", "text": "==END_ANCHOR=="})

        # 无论是否包含锚点，必须追加输出格式约束，保证 parse 稳定
        parts.append({"type": "input_text", "text": PROMPT_TAIL.format(start=start)})

        # 可选：dump 本块的 payload 预览（含图片计数与 dataURL 头部）
        if dump_payload:
            preview = {
                "model": model,
                "chunk": {"start": start, "end": end, "pages": end - start + 1, "images": img_cnt},
                "input_preview": []
            }
            for part in parts:
                if part.get("type") == "input_text":
                    t = str(part.get("text", ""))
                    preview["input_preview"].append({
                        "type": "text",
                        "text": t[:300] + ("..." if len(t) > 300 else "")
                    })
                elif part.get("type") == "input_image":
                    iu = part.get("image_url")
                    # 兼容两种形态：dict 或 str
                    if isinstance(iu, dict):
                        url = str(iu.get("url", ""))
                        detail = iu.get("detail", None)
                    else:
                        url = str(iu or "")
                        detail = part.get("detail", None)

                    preview["input_preview"].append({
                        "type": "image",
                        "image_url_head": url[:60],
                        "detail": detail,
                    })

            (debug_dir / f"{book_json.stem}_STEPC_win_{start}_{end}_payload_preview.json").write_text(
                json.dumps(preview, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        # 统一请求消息：后续由 call_openai_compat 按模型分流
        request_messages = [{"role": "user", "content": parts}]

        # ✅ trace：窗口级仅累计，不打印（避免刷屏）
        if trace_tokens:
            # 仅统计文本 parts（input_text）
            chunk_chars = sum(
                len(str(p.get("text", "")))
                for p in parts
                if p.get("type") == "input_text"
            )

            # 统计图片 dataURL 长度（input_image）
            img_urls = [
                str(p.get("image_url", ""))
                for p in parts
                if p.get("type") == "input_image"
            ]
            chunk_img_n = len(img_urls)
            chunk_total_url_len = sum(len(u) for u in img_urls)
            chunk_max_url_len = max([len(u) for u in img_urls], default=0)

            # 全局累计（用于最终汇总）
            _trace_step2_global_chars += chunk_chars
            _trace_step2_global_img_n += chunk_img_n
            _trace_step2_global_total_url_len += chunk_total_url_len
            _trace_step2_global_max_url_len = max(_trace_step2_global_max_url_len, chunk_max_url_len)
            _trace_step2_global_windows += 1

        log(f"🖼️ 本块图像数：{img_cnt}")
        # raw_path 已在窗口开头定义（支持 reuse_existing_windows）

        # ====== 重试到成功或达上限 ======
        success = False
        raw_final = ""
        parsed = None

        # Dry run: construct a zero-score response and pass it through the parser.
        if dry_run:
            raw_final = _make_zero_raw_for_window(start, end)
            # 写出与真实流程一致的 raw 快照，便于对齐
            raw_path.write_text(raw_final, encoding="utf-8")
            try:
                parsed = parse_window_txt(raw_final)
                success = True
            except Exception:
                parsed = {"scores": []}  # 兜底

        # <<<

        if not dry_run:
            raw = ""
            parsed = None
            got_raw = False  # 本窗口内是否至少有一次成功拿到文本

            for attempt in range(1, max_retries_step2 + 1):
                log(f"📡 发起请求（第{attempt}/{max_retries_step2}次）…")
                try:
                    # —— 发请求（业务重试由这个 for 控制）——
                    raw = call_openai_compat(
                        client=client,
                        model=model,
                        messages=request_messages,
                        temperature=0.2,
                        reasoning_effort=reasoning_effort,
                        text_verbosity=text_verbosity,
                    )

                    raw = str(raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
                    got_raw = bool(raw)

                    # —— 拒答检测：命中则立刻 Fail-Fast（方案A），不做盲重试，避免额外费用 ——
                    if _is_model_refusal(raw or ""):
                        raw_path.write_text(raw or "", encoding="utf-8")
                        log(f"📝 已保存原始返回：{raw_path}")
                        errp.write_text(
                            f"model_refusal (no-retry)\n\n====RAW====\n{raw or ''}",
                            encoding="utf-8"
                        )
                        log(f"⛔ 窗口 {start}-{end} 模型拒答：立即终止本书 StepC（不再重试），避免额外费用。")

                        status = RunStatus(
                            success=False,
                            hard_fail=True,
                            reason=f"stepC_model_refusal_{start}_{end}",
                            detail={"start": start, "end": end, "attempt": attempt}
                        )
                        save_run_status(book_json.stem, "STEPC", status)
                        return {"status": status.__dict__}


                    # —— 解析（注意：你的 parse_window_txt 只接收一个参数）——
                    try:
                        parsed = parse_window_txt(raw or "")
                    except Exception as pe:
                        parsed = None
                        log(f"⚠️ 解析异常（第{attempt}次）：{type(pe).__name__}: {pe}")

                    # —— 成功判定：有分数 + 页码在本窗口范围内 ——
                    if parsed and isinstance(parsed, dict) and parsed.get("scores"):
                        rows = parsed["scores"]
                        # 可选：把越界页过滤掉，避免模型把其他页写进来
                        rows = [r for r in rows if
                                isinstance(r, dict) and "page" in r and start <= int(r["page"]) <= end]
                        if rows:
                            raw_path.write_text(raw or "", encoding="utf-8")
                            log(f"📝 已保存原始返回：{raw_path}")
                            for row in rows:
                                raw_scores.setdefault(int(row["page"]), []).append(row)
                            log(f"✅ 解析成功：{start}-{end} 共 {len(rows)} 页")
                            # Clear stale parse errors after a successful retry.
                            try:
                                if errp.exists():
                                    errp.unlink()
                            except Exception:
                                pass
                            break  # 成功 -> 退出 for attempt

                except Exception as e:
                    log(f"⚠️ 调用失败（第{attempt}/{max_retries_step2}次）：{e}")

            # —— 用尽重试后仍未成功：Fail-Fast ——
            if not (parsed and isinstance(parsed, dict) and parsed.get("scores")):
                raw_path.write_text(raw or "", encoding="utf-8")
                log(f"📝 已保存原始返回：{raw_path}")
                errp = debug_dir / f"{book_json.stem}_STEPC_win_{start}_{end}_parse_error.txt"
                if not got_raw:
                    errp.write_text(
                        f"request failed after {max_retries_step2} attempts\n\n====RAW====\n{raw or ''}",
                        encoding="utf-8"
                    )
                    log(f"⛔ 窗口 {start}-{end} 请求阶段彻底失败，触发 Fail-Fast：本书终止。")
                else:
                    errp.write_text(
                        f"empty/invalid or parse failed after {max_retries_step2} attempts\n\n====RAW====\n{raw or ''}",
                        encoding="utf-8"
                    )
                    log(f"⛔ 窗口 {start}-{end} 解析阶段彻底失败，触发 Fail-Fast：本书终止。")

                status = RunStatus(
                    success=False,
                    hard_fail=True,
                    reason=f"step2_window_fail_{start}_{end}",
                    detail={"start": start, "end": end, "got_raw": got_raw}
                )
                save_run_status(book_json.stem, "STEPC", status)
                return {"status": status.__dict__}

        # 更新下一块的锚点（取本块的最后两页，若只有1页就取1页）
        last2 = [max(end - 1, start), end] if (end - start + 1) >= 2 else [end]
        anchor_pages = last2

    # ====== 融合（保持你原有口径；示例：取每页“最新一次”） ======
    final_scores: List[Dict[str, Any]] = []
    for i in range(1, N + 1):
        if i in raw_scores and raw_scores[i]:
            final_scores.append(raw_scores[i][-1])
        else:
            final_scores.append({"page": i, "emotions": {e: 0 for e in EMOS}})

    # ✅ trace：打印 Step C 全书级汇总（窗口全部跑完之后）
    if trace_tokens:
        total_chars = _trace_step2_global_chars
        total_tokens_est = max(1, total_chars // 4)

        n_img = _trace_step2_global_img_n
        total_len = _trace_step2_global_total_url_len
        avg_len = (total_len // n_img) if n_img else 0
        max_len = _trace_step2_global_max_url_len

        log(f"🧾 Step C tokens (estimated, text-only): ~{total_tokens_est} | chars={total_chars} | windows={_trace_step2_global_windows}")
        log(f"🖼️ Step C images: n={n_img} | total_image_url_len={total_len} | avg={avg_len} | max={max_len} | detail={image_detail_step2}")

    out_json = {
        "book_id": book_json.stem,
        "scores": final_scores,
        "model": model,
        "meta": {
            "step": "STEP_C_FEEL_LOCAL",
            "inject_page_text": bool(inject_page_text),
            "inject_page_images": bool(inject_page_images),
            "include_anchor_hint": bool(include_anchor_hint),
            "include_anchor_images": bool(include_anchor_images),
            "include_anchor_s1_cues": bool(include_anchor_s1_cues),
            "inject_book_summary": bool(inject_book_summary),
            "inject_s1_summary": bool(inject_s1_summary),
            "inject_s1_candidates": bool(inject_s1_candidates),
            "inject_s1_text_cues": bool(inject_s1_text_cues),
            "inject_s1_visual_cues": bool(inject_s1_visual_cues),
            "chunk_mode": chunk_mode,
            "image_detail": image_detail_step2,
            "max_image_side": max_image_side_step2,
            "segments_from_stepB": len(stepb_chunks),
            "chunks_used": len(chunks),
        },
    }

    (out_dir / f"{book_json.stem}_STEPC_emotions.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Record successful completion so the merge step can proceed.
    status = RunStatus(success=True, hard_fail=False, reason="ok", detail={"windows": len(chunks)})
    save_run_status(book_json.stem, "STEPC", status)
    log(f"✅ StepC 完成：{out_dir / f'{book_json.stem}_STEPC_emotions.json'}")
    return out_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MOSAIC Step C runner")
    parser.add_argument("--book-json", type=str, required=True)
    parser.add_argument("--stepA-json", type=str, default="", help="可选；A2 可留空。")
    parser.add_argument("--stepB-json", type=str, default="", help="可选；A3/A4 可留空。")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--dump-payload", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/mosaic/standalone", help="Output directory for Step C artifacts.")
    parser.add_argument("--reuse-existing-windows", action="store_true",
                        help="Reuse existing per-window raw results; rerun only failed window and subsequent windows.")

    parser.add_argument("--inject-page-text", action=argparse.BooleanOptionalAction, default=True,
                        help="Inject the current window's page text.")
    parser.add_argument("--inject-page-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Inject the current window's page images.")
    parser.add_argument("--allow-missing-images", action="store_true", help="Allow missing images (skip silently by default).")
    parser.add_argument("--emit-missing-image-placeholders", action="store_true", help="Emit '(missing)' placeholders for debug.")

    parser.add_argument("--include-window-intent", action="store_true",
                        help="(debug) ask model to output window intent.")
    parser.add_argument("--include-anchor-hint", action=argparse.BooleanOptionalAction, default=True,
                        help="Include the cross-window continuity anchor.")
    parser.add_argument("--anchor-images", action=argparse.BooleanOptionalAction,
                        dest="include_anchor_images", default=False,
                        help="Inject images for anchor pages.")
    parser.add_argument("--anchor-s1-cues", action=argparse.BooleanOptionalAction,
                        dest="include_anchor_s1_cues", default=False,
                        help="Inject Step A cues for anchor pages.")
    parser.add_argument("--inject-book-summary", action=argparse.BooleanOptionalAction, default=False,
                        help="Inject the Step A book summary.")
    parser.add_argument("--inject-s1-summary", action=argparse.BooleanOptionalAction, default=False,
                        help="Inject Step A page summaries.")
    parser.add_argument("--inject-s1-candidates", action=argparse.BooleanOptionalAction, default=False,
                        help="Inject Step A emotion candidates.")
    parser.add_argument("--inject-s1-text-cues", action=argparse.BooleanOptionalAction, default=True,
                        help="Inject Step A text cues.")
    parser.add_argument("--inject-s1-visual-cues", action=argparse.BooleanOptionalAction, default=True,
                        help="Inject Step A visual cues.")
    parser.add_argument("--include-prompt-head", action=argparse.BooleanOptionalAction, default=False,
                        help="Inject the optional prompt head.")
    parser.add_argument(
        "--chunk-mode",
        type=str,
        default="stepb_windows",
        choices=["stepb_windows", "whole_book", "fixed_windows"],
        help="Chunking mode for StepC: use StepB windows / score the whole book / fixed windows.",
    )
    parser.add_argument("--window-size-stepC", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--max-retries-stepC", type=int, default=3)
    parser.add_argument("--image-detail-stepC", type=str, default="auto", choices=["auto", "low", "high"])
    parser.add_argument("--max-image-side-stepC", type=int, default=512)

    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--text-verbosity", type=str, default=None)


    args = parser.parse_args()

    run_step_c(
        book_json=Path(args.book_json),
        stepA_json=(Path(args.stepA_json) if args.stepA_json.strip() else None),
        stepB_json=(Path(args.stepB_json) if args.stepB_json.strip() else None),
        model=args.model,
        dump_payload=args.dump_payload,
        reasoning_effort=args.reasoning_effort,
        text_verbosity=args.text_verbosity,
        max_retries_stepC=args.max_retries_stepC,
        image_detail_stepC=args.image_detail_stepC,
        max_image_side_stepC=args.max_image_side_stepC,
        inject_page_text=args.inject_page_text,
        inject_page_images=args.inject_page_images,
        allow_missing_images=args.allow_missing_images,
        emit_missing_image_placeholders=args.emit_missing_image_placeholders,
        include_window_intent=args.include_window_intent,
        include_anchor_hint=args.include_anchor_hint,
        include_anchor_images=args.include_anchor_images,
        include_anchor_s1_cues=args.include_anchor_s1_cues,
        dry_run=args.dry_run,
        out_dir=args.out_dir,
        inject_book_summary=args.inject_book_summary,
        inject_s1_summary=args.inject_s1_summary,
        inject_s1_candidates=args.inject_s1_candidates,
        inject_s1_text_cues=args.inject_s1_text_cues,
        inject_s1_visual_cues=args.inject_s1_visual_cues,
        include_prompt_head=args.include_prompt_head,
        chunk_mode=args.chunk_mode,
        window_size_stepC=args.window_size_stepC,
        reuse_existing_windows=args.reuse_existing_windows,
    )
