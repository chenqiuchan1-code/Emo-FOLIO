# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import time
import base64
from urllib.parse import quote
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import OpenAI
from mosaic.runtime import RunStatus, save_run_status

# ---------------------------------- 常量 ---------------------------------- #
INTENTS = {"安抚", "激励", "逗趣", "引发好奇", "制造紧张", "表达悲悯", "信息传递", "教育启发"}

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

def _make_stepB_instructions(require_rationale_output: bool = True) -> str:
    if require_rationale_output:
        return f"""
你是一名儿童绘本专家。你将看到一本儿童绘本的全书级材料（包含：全书摘要、逐页摘要线索；可能还包含可选的原始图文页）。
你的任务是：在全书尺度上输出连续覆盖全书的若干段落（segments），并为每段标注一个“阶段级创作意图”。

要求：
1）分段应反映故事的主要阶段，而非每页的细小变化。
2）整本书通常划分为 3~6 个连续阶段；除非出现明确情节转折，否则不要频繁切段。
3）意图是“阶段级创作意图”，不应因个别页面内容而频繁改变。
4）必须输出完整覆盖全书、连续不重叠的 segments；段落按页码递增。
5）候选意图仅限：{", ".join(sorted(INTENTS))}
6）“意图”字段不是情绪标签，不能写“快乐、悲伤、平静、愤怒、好奇”等情绪词。
7）若你想表达“快乐/开心/有趣/轻松”，统一写为“逗趣”；若你想表达“平静/温和/安慰/安心”，统一写为“安抚”。
8）若拿不准，也必须从候选集合中选择最接近的一个，绝不能输出集合外标签。

输出格式（严格遵守，不要输出其他内容）：
==SEGMENTS==
# 段1
起止页：a-b
意图：<仅从候选意图中选择1个>
依据：<用1-2句说明该段为何构成一个阶段；若有切分点，请说明为何在 b 与 b+1 之间切分>
# 段2
起止页：c-d
意图：...
依据：...
==END==
""".strip()

    return f"""
你是一名儿童绘本专家。你将看到一本儿童绘本的全书级材料（包含：全书摘要、逐页摘要线索；可能还包含可选的原始图文页）。
你的任务是：在全书尺度上输出连续覆盖全书的若干段落（segments），并为每段标注一个“阶段级创作意图”。

要求：
1）分段应反映故事的主要阶段，而非每页的细小变化。
2）整本书通常划分为 3~6 个连续阶段；除非出现明确情节转折，否则不要频繁切段。
3）意图是“阶段级创作意图”，不应因个别页面内容而频繁改变。
4）必须输出完整覆盖全书、连续不重叠的 segments；段落按页码递增。
5）候选意图仅限：{", ".join(sorted(INTENTS))}
6）“意图”字段不是情绪标签，不能写“快乐、悲伤、平静、愤怒、好奇”等情绪词。
7）若你想表达“快乐/开心/有趣/轻松”，统一写为“逗趣”；若你想表达“平静/温和/安慰/安心”，统一写为“安抚”。
8）若拿不准，也必须从候选集合中选择最接近的一个，绝不能输出集合外标签。

输出格式（严格遵守，不要输出其他内容）：
==SEGMENTS==
# 段1
起止页：a-b
意图：<仅从候选意图中选择1个>
# 段2
起止页：c-d
意图：...
==END==
""".strip()


def log(msg: str) -> None:
    print(msg, flush=True)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

HF_ENDPOINT_API_KEY_ENV = "HF_ENDPOINT_API_KEY"
HF_INTERNVL35_8B_BASE_URL_ENV = "HF_INTERNVL35_8B_BASE_URL"
HF_INTERNVL35_38B_BASE_URL_ENV = "HF_INTERNVL35_38B_BASE_URL"

# LLaVA-OneVision-1.5 Hugging Face endpoints
HF_LLAVA_OV15_4B_BASE_URL_ENV = "HF_LLAVA_OV15_4B_BASE_URL"
HF_LLAVA_OV15_8B_BASE_URL_ENV = "HF_LLAVA_OV15_8B_BASE_URL"
HF_LLAVA_OV15_4B_MODEL_NAME_ENV = "HF_LLAVA_OV15_4B_MODEL_NAME"
HF_LLAVA_OV15_8B_MODEL_NAME_ENV = "HF_LLAVA_OV15_8B_MODEL_NAME"

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

# LLaVA model aliases
def is_llava_ov15_4b_model(model: str) -> bool:
    return str(model or "").lower() == "llava-ov15-4b"

def is_llava_ov15_8b_model(model: str) -> bool:
    return str(model or "").lower() == "llava-ov15-8b"

def is_llava_model(model: str) -> bool:
    return is_llava_ov15_4b_model(model) or is_llava_ov15_8b_model(model)

def use_hf_dataset_image_url(model: str) -> bool:
    if not (is_internvl_model(model) or is_llava_model(model)):
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
    if is_llava_ov15_4b_model(model):
        return os.getenv(HF_LLAVA_OV15_4B_BASE_URL_ENV, "").strip()
    if is_llava_ov15_8b_model(model):
        return os.getenv(HF_LLAVA_OV15_8B_BASE_URL_ENV, "").strip()
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

    if m == "llava-ov15-4b":
        return os.getenv(
            HF_LLAVA_OV15_4B_MODEL_NAME_ENV,
            "LLaVA-OneVision-1.5-4B-Instruct"
        ).strip()

    if m == "llava-ov15-8b":
        return os.getenv(
            HF_LLAVA_OV15_8B_MODEL_NAME_ENV,
            "LLaVA-OneVision-1.5-8B-Instruct"
        ).strip()

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

    if is_internvl_model(model) or is_llava_model(model):
        api_key = os.getenv(HF_ENDPOINT_API_KEY_ENV, "").strip()
        base_url = _get_hf_endpoint_base_url(model)
        if not api_key:
            raise RuntimeError("当前模型走 Hugging Face Endpoint，但未设置 HF_ENDPOINT_API_KEY。")
        if not base_url:
            raise RuntimeError(f"当前模型为 {model}，但未设置对应的 Hugging Face Endpoint BASE_URL。")
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI()

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_picture_dir(json_path: Path, data: Any) -> Optional[Path]:
    """与 Step A 对齐：只在 book 中显式给出 picture_dir 时解析；
    相对路径会尝试 json_dir/picture_dir 与 json_dir.parent/picture_dir。
    """
    json_dir = json_path.parent
    picture_dir_val = data.get("picture_dir") if isinstance(data, dict) else None
    if not picture_dir_val:
        return None

    pd = Path(str(picture_dir_val)).expanduser()
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



def _resolve_image_path(raw_img: str, json_path: Path, picture_dir_abs: Optional[Path]) -> Path:
    """与 Step A 对齐：相对路径依次尝试
    1) picture_dir_abs / img
    2) json_dir / img
    3) json_dir.parent / img
    """
    p = Path(str(raw_img)).expanduser()
    if p.is_absolute():
        return p

    json_dir = json_path.parent
    candidates: List[Path] = []
    if picture_dir_abs:
        candidates.append((picture_dir_abs / p).resolve())
    candidates.append((json_dir / p).resolve())
    candidates.append((json_dir.parent / p).resolve())

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

def _make_image_part(
    img_path: Path,
    max_side: Optional[int],
    detail: str,
    model: str,
    picture_dir_abs: Optional[Path],
) -> Dict[str, Any]:
    if use_hf_dataset_image_url(model):
        return {
            "type": "input_image",
            "image_url": {
                "url": build_hf_dataset_image_url(img_path, picture_dir_abs),
                "detail": detail,
            },
        }

    img = Image.open(img_path).convert("RGB")
    if max_side and max_side > 0:
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            s = max_side / float(m)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    import io
    b = io.BytesIO()
    img.save(b, format="PNG")
    data = base64.b64encode(b.getvalue()).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": {"url": f"data:image/png;base64,{data}", "detail": detail},
    }


def _format_stepA(
    stepA: Dict[str, Any],
    *,
    include_book_summary: bool = True,
    include_page_summaries: bool = True,
    include_emotion_candidates: bool = True,
    include_text_cues: bool = True,
    include_visual_cues: bool = True,
) -> str:
    bs = (stepA.get("book_summary") or "").strip()
    lines: List[str] = []

    if include_book_summary:
        lines.append("【BOOK_SUMMARY】")
        lines.append(bs if bs else "(空)")
        lines.append("")

    if include_page_summaries:
        lines.append("【PAGE_SUMMARIES】")
        pss = stepA.get("page_summaries") or []
        for row in pss:
            p = row.get("page")
            if p is None:
                continue

            semantic = (row.get("semantic") or row.get("summary") or "").strip()

            ec = row.get("emotion_candidates") or row.get("candidates") or []
            ecs: List[str] = []
            if isinstance(ec, list):
                for it in ec:
                    if isinstance(it, dict) and it.get("label"):
                        ecs.append(str(it["label"]))
                    elif isinstance(it, str):
                        ecs.append(it)
            cand_txt = "，".join(ecs[:3]) if ecs else ""

            text_cues = row.get("text_cues") or row.get("text_clues") or []
            visual_cues = row.get("visual_cues") or row.get("visual_clues") or []
            tc = "，".join([str(x) for x in text_cues]) if isinstance(text_cues, list) else str(text_cues)
            vc = "，".join([str(x) for x in visual_cues]) if isinstance(visual_cues, list) else str(visual_cues)

            lines.append(f"# 第{p}页")
            lines.append(f"语义摘要：{semantic}")
            if include_emotion_candidates and cand_txt:
                lines.append(f"候选情感：{cand_txt}")
            if include_text_cues and tc and tc != "None":
                lines.append(f"文本线索：{tc}")
            if include_visual_cues and vc and vc != "None":
                lines.append(f"图像线索：{vc}")
            lines.append("")

    return "\n".join(lines).strip()


def parse_segments(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    s = text.strip()
    m = re.search(r"==SEGMENTS==(.+)$", s, flags=re.S)
    if m:
        s = m.group(1)
    blocks = re.split(r"^\s*#\s*段\s*\d+\s*$", s, flags=re.M)

    out: List[Dict[str, Any]] = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        m1 = re.search(r"起止页[:：]\s*(\d+)\s*[-~—]\s*(\d+)", blk)
        m2 = re.search(r"意图[:：]\s*([\u4e00-\u9fa5A-Za-z0-9_\-]+)", blk)
        if m1:
            a, b = int(m1.group(1)), int(m1.group(2))
            intent_raw = (m2.group(1).strip() if m2 else "")
            intent = normalize_intent_label(intent_raw)
            if a > b:
                a, b = b, a
            out.append({"start": a, "end": b, "intent": intent})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def _redact_image_data_urls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """保存 payload 预览时脱敏 data url（避免文件爆炸）。
    兼容 Responses API 两种消息 content 形态：
      - str（纯文本）
      - list[dict]（multimodal parts）
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")

        # case 1: content 是纯字符串
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        # case 2: content 是 list parts
        mm: Dict[str, Any] = {"role": role, "content": []}
        parts = content if isinstance(content, list) else []
        for it in parts:
            # 容错：遇到非 dict 直接保留其 repr
            if not isinstance(it, dict):
                mm["content"].append({"type": "input_text", "text": str(it)})
                continue

            if it.get("type") == "input_image":
                u = (it.get("image_url") or {}).get("url", "")
                head = u[:80] + ("...(redacted)" if len(u) > 80 else "")
                mm["content"].append(
                    {
                        "type": "input_image",
                        "image_url": {
                            "url": head,
                            "detail": (it.get("image_url") or {}).get("detail", "auto"),
                        },
                    }
                )
            else:
                mm["content"].append(it)
        out.append(mm)
    return out

def _format_stepA_page(
    stepA: Dict[str, Any],
    page_idx: int,
    *,
    include_semantic_summary: bool = True,
    include_emotion_candidates: bool = True,
    include_text_cues: bool = True,
    include_visual_cues: bool = True,
) -> str:
    """返回指定页的 Step-A 摘要线索文本（用于逐页交错模式）。page_idx 从 1 开始。"""
    pss = stepA.get("page_summaries") or []
    row = None
    for r in pss:
        if isinstance(r, dict) and r.get("page") == page_idx:
            row = r
            break
    if not row:
        return f"【StepA-第{page_idx}页摘要线索】(缺失)"

    semantic = (row.get("semantic") or row.get("summary") or "").strip()

    ec = row.get("emotion_candidates") or row.get("candidates") or []
    ecs: List[str] = []
    if isinstance(ec, list):
        for it in ec:
            if isinstance(it, dict) and it.get("label"):
                ecs.append(str(it["label"]))
            elif isinstance(it, str):
                ecs.append(it)
    cand_txt = "，".join(ecs[:3]) if ecs else ""

    text_cues = row.get("text_cues") or row.get("text_clues") or []
    visual_cues = row.get("visual_cues") or row.get("visual_clues") or []

    tc = "，".join([str(x) for x in text_cues]) if isinstance(text_cues, list) else str(text_cues)
    vc = "，".join([str(x) for x in visual_cues]) if isinstance(visual_cues, list) else str(visual_cues)

    lines: List[str] = []
    if include_semantic_summary:
        lines.append(f"语义摘要：{semantic}")
    if include_emotion_candidates and cand_txt:
        lines.append(f"候选情感：{cand_txt}")
    if include_text_cues and tc and tc != "None":
        lines.append(f"文本线索：{tc}")
    if include_visual_cues and vc and vc != "None":
        lines.append(f"图像线索：{vc}")
    return "\n".join(lines).strip()


def build_messages(
    book_json: Path,
    stepA_json: Optional[Path],
    model: str,

    inject_page_text: bool,
    inject_page_images: bool,
    inline_stepA_per_page: bool,

    allow_missing_images: bool,
    emit_missing_image_placeholders: bool,

    image_detail: str,
    max_image_side: Optional[int],

    include_book_summary: bool = True,
    include_page_summaries: bool = True,
    include_emotion_candidates: bool = True,
    include_text_cues: bool = True,
    include_visual_cues: bool = True,
    require_rationale_output: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    book = load_json(book_json)
    stepA = load_json(stepA_json) if stepA_json else {}

    input_parts: List[Dict[str, Any]] = []
    preview: Dict[str, Any] = {
        "book_id": book_json.stem,
        "has_stepA": bool(stepA_json),
        "inject_page_text": bool(inject_page_text),
        "inject_page_images": bool(inject_page_images),
        "inline_stepA_per_page": bool(inline_stepA_per_page),
        "include_book_summary": bool(include_book_summary),
        "include_page_summaries": bool(include_page_summaries),
        "include_emotion_candidates": bool(include_emotion_candidates),
        "include_text_cues": bool(include_text_cues),
        "include_visual_cues": bool(include_visual_cues),
        "require_rationale_output": bool(require_rationale_output),
    }

    pages = book.get("pages") or []
    pic_dir = _resolve_picture_dir(book_json, book)
    preview["picture_dir_resolved"] = (str(pic_dir) if pic_dir else None)

    # ① 若有 StepA，先放全局线索块
    if stepA:
        stepA_txt = _format_stepA(
            stepA,
            include_book_summary=include_book_summary,
            include_page_summaries=(include_page_summaries and (not inline_stepA_per_page)),
            include_emotion_candidates=include_emotion_candidates,
            include_text_cues=include_text_cues,
            include_visual_cues=include_visual_cues,
        )
        if stepA_txt.strip():
            input_parts.append({"type": "input_text", "text": stepA_txt})

    # ② 原始页材料（可选）
    for idx, page in enumerate(pages, start=1):
        page_text = page.get("text") or page.get("caption") or page.get("content") or ""
        page_text = str(page_text).strip()

        if inline_stepA_per_page and stepA:
            page_cues = _format_stepA_page(
                stepA,
                idx,
                include_semantic_summary=include_page_summaries,
                include_emotion_candidates=include_emotion_candidates,
                include_text_cues=include_text_cues,
                include_visual_cues=include_visual_cues,
            )
            if page_cues.strip():
                input_parts.append({
                    "type": "input_text",
                    "text": f"【StepA-第{idx}页摘要线索】\n{page_cues}\n"
                })

        if inject_page_text:
            input_parts.append({
                "type": "input_text",
                "text": f"【第{idx}页原始文字】\n{page_text}\n"
            })

        if inject_page_images:
            raw_img = (
                page.get("image")
                or page.get("image_path")
                or page.get("img")
                or page.get("img_path")
                or page.get("image_url")
                or page.get("path")
                or page.get("filename")
                or page.get("file")
                or ""
            )
            raw_img = str(raw_img).strip()
            if raw_img:
                img_path = _resolve_image_path(raw_img, book_json, pic_dir)
                if img_path.exists():
                    input_parts.append(_make_image_part(
                        img_path,
                        max_image_side,
                        image_detail,
                        model,
                        pic_dir,
                    ))
                else:
                    if allow_missing_images and emit_missing_image_placeholders:
                        input_parts.append({"type": "input_text", "text": f"(提示：第{idx}页图片缺失：{img_path})"})
                    elif not allow_missing_images:
                        raise FileNotFoundError(f"Page {idx}: image not found (picture_dir={pic_dir}, raw={raw_img})")

    messages = [
        {"role": "system", "content": _make_stepB_instructions(require_rationale_output=require_rationale_output)},
        {"role": "user", "content": input_parts},
    ]
    return messages, preview


def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

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

    # ===== Gemini：统一走 Chat Completions =====
    if is_gemini_model(model):
        chat_messages = _normalize_to_chat_messages(messages)
        kwargs = {"model": model, "messages": chat_messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    api_model = resolve_api_model_name(model)

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


def _call_with_retries(client=None, payload=None, fn=None, max_retries: int = 5, base_delay: int = 2, timeout: int = 600):
    last_err = None
    for i in range(max_retries):
        try:
            print(f"📡 StepB 发起请求（第{i + 1}/{max_retries}次）…", flush=True)

            if fn is not None:
                return fn()

            if not isinstance(payload, dict):
                raise ValueError("_call_with_retries: 需要提供 fn= 或 payload=dict(...)")

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
            print(f"⚠️ StepB 第{i + 1}次调用失败：{e}", flush=True)
            if i == max_retries - 1:
                break
            time.sleep(base_delay * (2 ** i))

    raise last_err or RuntimeError("StepB 调用失败次数过多")


def _extract_text_from_rsp(rsp: Any) -> str:
    # 与 step3_segment.py 同风格：尽量稳健抽取文本
    txt = None
    try:
        if hasattr(rsp, "output_text"):
            txt = rsp.output_text
        if not txt and hasattr(rsp, "output") and rsp.output:
            # 兼容 response 对象结构
            parts = []
            for item in rsp.output:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            parts.append(getattr(c, "text", "") or "")
            txt = "\n".join([p for p in parts if p])
    except Exception:
        pass
    return (txt or "").strip()


def run_step_b(
    book_json: Path,
    stepA_json: Optional[Path],
    out_json: Path,
    out_txt: Optional[Path],
    model: str,

    inject_page_text: bool,
    inject_page_images: bool,
    inline_stepA_per_page: bool,

    allow_missing_images: bool,
    emit_missing_image_placeholders: bool,

    image_detail: str,
    max_image_side: Optional[int],
    dump_payload: bool,
    max_retries: int,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    text_verbosity: Optional[str] = None,

    include_book_summary: bool = True,
    include_page_summaries: bool = True,
    include_emotion_candidates: bool = True,
    include_text_cues: bool = True,
    include_visual_cues: bool = True,
    require_rationale_output: bool = True,
) -> Dict[str, Any]:



    book_id = Path(book_json).stem
    log(f"▶️ Step B 开始：{Path(book_json).name}")

    messages, preview = build_messages(
        book_json=Path(book_json),
        stepA_json=(Path(stepA_json) if stepA_json else None),
        model=model,

        inject_page_text=inject_page_text,
        inject_page_images=inject_page_images,
        inline_stepA_per_page=inline_stepA_per_page,

        allow_missing_images=allow_missing_images,
        emit_missing_image_placeholders=emit_missing_image_placeholders,

        image_detail=image_detail,
        max_image_side=max_image_side,

        include_book_summary=include_book_summary,
        include_page_summaries=include_page_summaries,
        include_emotion_candidates=include_emotion_candidates,
        include_text_cues=include_text_cues,
        include_visual_cues=include_visual_cues,
        require_rationale_output=require_rationale_output,
    )

    # payload 预览落盘（脱敏）
    if dump_payload:
        req_preview_path = out_json.parent / f"{book_id}_STEPB_request_preview.json"
        _save_json(req_preview_path,
                   {"model": model, "messages": _redact_image_data_urls(messages), "preview": preview})
        log(f"🧾 StepB request preview saved: {req_preview_path}")

    client = make_client_for_model(model)

    # 请求阶段错误：写 RunStatus（Fail-Fast 友好）
    try:
        rsp = _call_with_retries(
            fn=lambda: call_openai_compat(
                client=client,
                model=model,
                messages=messages,
                temperature=(0.2 if temperature is None else temperature),
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
            ),
            max_retries=max_retries,
            base_delay=2,
            timeout=600,
        )
    except Exception as e:
        status = RunStatus(success=False, hard_fail=True, reason="stepB_request_fail", detail={"error": f"{type(e).__name__}: {e}"})
        save_run_status(book_id, "STEPB", status)
        log("⛔ StepB 请求阶段彻底失败，触发 Fail-Fast：本书终止。")
        return {"status": status.__dict__}

    if isinstance(rsp, str):
        txt = rsp.strip()
    else:
        txt = _extract_text_from_rsp(rsp)

    txt = (txt or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    segs = parse_segments(txt)

    out_obj = {
        "book_id": book_id,
        "segments": segs,
        "raw_text": txt,
        "meta": {
            "model": model,
            "inject_page_text": inject_page_text,
            "inject_page_images": inject_page_images,
            "inline_stepA_per_page": inline_stepA_per_page,
            "allow_missing_images": allow_missing_images,
            "emit_missing_image_placeholders": emit_missing_image_placeholders,
            "image_detail": image_detail,
            "max_image_side": max_image_side,
        }
    }
    _save_json(out_json, out_obj)
    log(f"✅ StepB saved: {out_json}")

    if out_txt:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(txt, encoding="utf-8")
        log(f"✅ StepB raw txt saved: {out_txt}")

    status = RunStatus(success=True, hard_fail=False, reason="ok", detail={"n_segments": len(segs)})
    save_run_status(book_id, "STEPB", status)
    return out_obj


def main() -> None:
    ap = argparse.ArgumentParser(description="MOSAIC Step B: predict narrative stages and communicative intents.")
    ap.add_argument("--book-json", type=str, required=True)
    ap.add_argument("--stepA-json", type=str, default="", help="可选；A2 可留空。")
    ap.add_argument("--out-dir", type=str, default="results/mosaic/standalone", help="Output directory for Step B artifacts.")
    ap.add_argument("--out-json", type=str, default="", help="Optional. If empty, auto-name under --out-dir.")
    ap.add_argument("--out-txt", type=str, default="", help="Optional. If empty, do not write raw txt.")
    ap.add_argument("--model", type=str, default="gpt-4o")

    # --- Raw book injection switches ---
    ap.add_argument("--inject-page-text", action="store_true", help="Inject original page TEXT blocks.")
    ap.add_argument("--inject-page-images", action="store_true", help="Inject original page IMAGES.")
    ap.add_argument(
        "--inline-stepA-per-page",
        action="store_true",
        help="Interleave Step-A per-page summaries with each injected page's raw material.",
    )

    # --- StepA 内容开关（B2/B3/B4/B5） ---
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

    ap.add_argument(
        "--strict-images",
        action="store_true",
        help="If set, missing images will raise FileNotFoundError. Default: skip missing images silently.",
    )

    ap.add_argument(
        "--emit-missing-image-placeholders",
        action="store_true",
        help="If an image is missing and allow-missing-images is on, emit a '(missing)' text placeholder (debug only).",
    )
    ap.add_argument("--image-detail", type=str, default="auto", choices=["auto", "low", "high"])
    ap.add_argument("--max-image-side", type=int, default=512)
    ap.add_argument("--dump-payload", action="store_true")
    ap.add_argument("--max-retries", type=int, default=5)

    ap.add_argument("--reasoning-effort", type=str, default=None)
    ap.add_argument("--text-verbosity", type=str, default=None)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_json.strip():
        out_json = Path(args.out_json)
    else:
        out_json = out_dir / f"{Path(args.book_json).stem}_STEPB_segments.json"

    if args.out_txt.strip():
        out_txt = Path(args.out_txt)
    else:
        out_txt = out_dir / f"{Path(args.book_json).stem}_STEPB_raw.txt"

    run_step_b(
        book_json=Path(args.book_json),
        stepA_json=(Path(args.stepA_json) if args.stepA_json.strip() else None),
        out_json=out_json,
        out_txt=out_txt,
        model=args.model,

        inject_page_text=args.inject_page_text,
        inject_page_images=args.inject_page_images,
        inline_stepA_per_page=args.inline_stepA_per_page,

        allow_missing_images=(not args.strict_images),
        emit_missing_image_placeholders=args.emit_missing_image_placeholders,

        image_detail=args.image_detail,
        max_image_side=args.max_image_side,
        dump_payload=args.dump_payload,
        max_retries=args.max_retries,
        reasoning_effort=args.reasoning_effort,
        text_verbosity=args.text_verbosity,

        include_book_summary=args.include_book_summary_stepB,
        include_page_summaries=args.include_page_summaries_stepB,
        include_emotion_candidates=args.include_emotion_candidates_stepB,
        include_text_cues=args.include_text_cues_stepB,
        include_visual_cues=args.include_visual_cues_stepB,
        require_rationale_output=args.require_rationale_output_stepB,
    )


if __name__ == "__main__":
    main()
