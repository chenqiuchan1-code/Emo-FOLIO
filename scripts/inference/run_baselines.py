# -*- coding: utf-8 -*-
"""Run Full-book E2E or CoT inference over one or more book JSON files.

Use ``python -m scripts.inference.run_baselines --help`` for the command-line
interface or edit the configuration block in ``run_baselines.sh``.
"""

import os
import io
import json
import base64
from urllib.parse import quote
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from PIL import Image

# ==== 你可以自定义 Prompt 开头 / 结尾 ====
PROMPT_HEAD = """你将看到一本绘本的“原始图文数据”（按页）。请完整读取全书后再统一输出结果。"""

COT_HINT = (
    "请先在内部按“全书叙事脉络→阶段划分→页级情绪判断”的顺序逐步思考，"
    "再输出最终结果；不要输出推理过程，只输出最终要求的txt对象。"
)

# 注：这里只要求 GPT 返回“最小必需信息”：每页7类强度 + 段落意图（不直接返回派生字段）
PROMPT_TAIL = """任务如下（仅中文输出）：
（1）页面级情感强度：输出每页×7类情感的强度（{快乐、悲伤、恐惧、愤怒、惊讶、平静、好奇}；0-5整数）。
（2）情感阶段划分：输出若干连续页面分段（按创作意图稳定段），并给出每段的意图标签（{安抚、激励、逗趣、引发好奇、制造紧张、表达悲悯、信息传递、教育启发}）。
请仅输出以下结构的txt对象（字段含义见注释，不要任何解释或多余文字）：
{
  "book_id": "<可留空或复用文件名>",
  "pages": [
    {
      "page": <int>,
      "emotions": {
        "快乐": <0-5 int>, "悲伤": <0-5 int>, "恐惧": <0-5 int>,
        "愤怒": <0-5 int>, "惊讶": <0-5 int>, "平静": <0-5 int>, "好奇": <0-5 int>
      }
    },
    ...
  ],
  "intent_segments":[
    {"start_page":<int>, "end_page":<int>, "intent":"<从 {安抚, 激励, 逗趣, 引发好奇, 制造紧张, 表达悲悯, 信息传递, 教育启发} 中选择>"},
    ...
  ]
}

要求：
- 只返回一个严格可解析的txt；禁止任何注释、额外文本、Markdown。
- 情感强度为 0~5 的整数。
- 页码必须连续覆盖整本书的页数（缺页请补 0 强度）。
"""

DEFAULT_INSTRUCTIONS = "你是评测助手：严格遵循 PROMPT_HEAD 与 PROMPT_TAIL 的要求，输出结构化、可复制的文本结果。"

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

def log(msg: str):
    print(msg, flush=True)

def _fmt_books(paths):
    """把 Path 列表变成 'book_1, book_2, ...' 的字符串"""
    return ", ".join(p.stem for p in paths)

def downscale_to_data_url(img_path: Path, max_side=512) -> str:
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
    """把 JSON 顶层对象或数组统一成 pages 列表"""
    pages = data["pages"] if isinstance(data, dict) else data
    result = []
    for i, p in enumerate(pages):
        if not isinstance(p, dict):
            raise ValueError(f"第 {i+1} 个条目不是对象：{p!r}")
        page_num = p.get("page", i + 1)
        text = p.get("text", "")
        img = p.get("image") or p.get("image_path") or p.get("img") or p.get("image_url", "")
        result.append({"page": int(page_num), "text": str(text), "image_path": str(img)})
    result.sort(key=lambda x: x["page"])
    return result

def resolve_picture_dir(json_path: Path, data: Any) -> Optional[Path]:
    """
    解析 picture_dir 的绝对路径：
    1) 若 picture_dir 是绝对路径，直接使用；
    2) 若是相对路径，优先 json_dir / picture_dir；不存在则尝试 json_dir.parent / picture_dir；
    3) 都不存在则返回 None（后续还会用其他组合去找图片）。
    """
    json_dir = json_path.parent
    picture_dir_val = None
    if isinstance(data, dict):
        picture_dir_val = data.get("picture_dir")
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
    log(f"⚠️ 指定的 picture_dir 未找到：尝试 {guess1} 与 {guess2} 均不存在。将继续使用其他相对路径组合。")
    return None

def resolve_image_path(raw_img: str, json_path: Path, picture_dir_abs: Optional[Path]) -> Path:
    """
    多策略解析相对路径：
    1) 绝对路径：直接返回；
    2) picture_dir_abs / raw_img；
    3) json_dir / raw_img；
    4) json_dir.parent / raw_img（处理“book 与 picture 同级”的情形）。
    """
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

    # 全部不存在，就返回首选（用于错误提示展示）
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

def build_inputs(json_path: Path, data: Any, pages: List[Dict[str, Any]],
                 prompt_head: str, prompt_tail: str,
                 allow_missing: bool, missing_list: List[str],
                 *,
                 model: str,
                 inject_images: bool = True,
                 image_detail: str = "auto",
                 max_image_side: int = 512):

    parts = []
    if prompt_head:
        parts.append({"type": "input_text", "text": prompt_head})

    picture_dir_abs = resolve_picture_dir(json_path, data)

    for p in pages:
        parts.append({"type": "input_text", "text": f"【第{p['page']}页 - 文字】\n{p['text'].strip()}\n"})

        if not inject_images:
            continue

        raw = p["image_path"]
        if raw:
            ip = resolve_image_path(raw, json_path, picture_dir_abs)
            if ip.exists():
                image_ref = build_image_ref_for_model(
                    ip,
                    picture_dir_abs,
                    model,
                    max_side=max_image_side,
                    data_url_builder=downscale_to_data_url,
                )
                parts.append({
                    "type": "input_image",
                    "image_url": image_ref,
                    "detail": image_detail,
                })
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

def _apply_cot_prompt(prompt_head: str, cot_enabled: bool, cot_hint: str) -> str:
    if not cot_enabled:
        return prompt_head
    base = (prompt_head or "").rstrip()
    hint = (cot_hint or "").strip()
    if not hint:
        return base
    if hint in base:
        return base
    return f"{base}\n{hint}" if base else hint


def _resolve_out_dir(out_arg: str, model: str, cot_enabled: bool, base_dir: Path) -> Path:
    """Resolve baseline and CoT output directories.

    The default roots are ``results/baselines`` and ``results/cot``. Explicit
    custom output paths are preserved.
    """
    out = Path(out_arg).expanduser()

    results_root = (base_dir / "results" / "baselines").resolve()
    cot_root = (base_dir / "results" / "cot").resolve()

    try:
        out_resolved = out.resolve()
    except Exception:
        out_resolved = out

    # 情况 A：用户传的是 ./results
    if cot_enabled and out_resolved == results_root:
        return cot_root / model

    # 情况 B：用户传的是 ./results/<model>
    if cot_enabled and out_resolved == (results_root / model):
        return cot_root / model

    # 情况 C：普通 baseline，且只传了 ./results
    if (not cot_enabled) and out_resolved == results_root:
        return results_root / model

    # 情况 D：普通 baseline，且已传 ./results/<model>
    if (not cot_enabled) and out_resolved == (results_root / model):
        return results_root / model

    # 其他自定义路径：尊重用户输入；若最后一级不是 model，则自动补 model
    if out.name != model:
        out = out / model

    return out


def _dump_payload_preview(
    request_messages,
    out_dir: Path,
    book_stem: str,
    model: str,
    cot_enabled: bool,
    cot_hint: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        items = request_messages[0]["content"]
    except Exception:
        items = []

    preview = {
        "model": model,
        "cot_enabled": bool(cot_enabled),
        "cot_hint": cot_hint if cot_enabled else "",
        "input_preview": [],
    }

    img_cnt = 0
    for part in items:
        ptype = part.get("type")
        if ptype == "input_text":
            text = str(part.get("text", ""))
            preview["input_preview"].append({
                "type": "text",
                "text": text[:400] + ("..." if len(text) > 400 else "")
            })
        elif ptype == "input_image":
            iu = part.get("image_url")
            if isinstance(iu, dict):
                url = str(iu.get("url", ""))
                detail = iu.get("detail", None)
            else:
                url = str(iu or "")
                detail = part.get("detail", None)

            img_cnt += 1
            preview["input_preview"].append({
                "type": "image",
                "image_url_head": url[:80],
                "detail": detail,
            })

    preview["meta"] = {
        "num_items": len(items),
        "num_images": img_cnt,
    }

    preview_path = out_dir / f"{book_stem}_WHOLE_payload_preview.json"
    preview_path.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log(f"🧾 Baseline payload preview saved: {preview_path}")
    return preview_path

# ---------- Provider API compatibility ----------
def call_openai_compat(client, *, model: str, messages,
                       temperature: float = 0.2,
                       reasoning_effort: str | None = None,
                       text_verbosity: str | None = None):
    """
    messages 可为：
      A) Chat 风格：
         [{"role":"user","content":[
            {"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"..."}},
            {"type":"input_text","text":"..."}, {"type":"input_image","image_url":"..."}
         ]}]
      B) Responses 风格：
         [{"type":"input_text","text":"..."}, {"type":"input_image","image_url":"..."}]
      C) 纯文本/其他对象：自动兜底为一条 input_text
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

    # —— 将 Chat / Responses 输入统一转成 Chat Completions 可接受的 messages —— #
    def _normalize_to_chat_messages(msgs):
        # 情况1：已经是 chat messages
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

        # 情况2：responses items
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

        # 情况3：其他对象兜底
        return [{"role": "user", "content": [{"type": "text", "text": str(msgs)}]}]

    # —— 将 Chat content 统一规范为 Responses content —— #
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

    def _normalize_responses_items(items):
        norm = []
        for it in (items or []):
            t = it.get("type")
            if t == "input_text":
                norm.append({"type": "input_text", "text": it.get("text", "")})
            elif t == "input_image":
                url = _extract_url(it.get("image_url") or it.get("url"))
                if url:
                    norm.append({"type": "input_image", "image_url": url})
        return norm

    api_model = resolve_api_model_name(model)

    # ==================== Gemini：统一走 Chat Completions ====================
    if is_gemini_model(model):
        chat_messages = _normalize_to_chat_messages(messages)

        kwargs = {
            "model": api_model,
            "messages": chat_messages,
            "temperature": temperature,
        }
        # Gemini OpenAI compat 可接受 reasoning_effort；text_verbosity 这里不透传
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    # ==================== GPT-5：走 Responses API ====================
    if str(model).startswith("gpt-5"):
        input_items = []

        if _is_chat_msgs(messages):
            for m in messages:
                role = m.get("role", "user")
                content_items = _chat_content_to_responses_items(m.get("content", []))
                content_items = _normalize_responses_items(content_items)
                if content_items:
                    input_items.append({"role": role, "content": content_items})

        elif _is_responses_items(messages):
            input_items = [{"role": "user", "content": _normalize_responses_items(messages)}]

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

    # ==================== 其他 GPT 多模态模型：走 Chat Completions ====================
    chat_messages = _normalize_to_chat_messages(messages)
    resp = client.chat.completions.create(
        model=api_model,
        messages=chat_messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def call_with_retries(client=None, payload=None, fn=None, max_retries=5, base_delay=2.0, timeout=600):
    """
    Retry a provider request supplied as a callable or legacy payload.
    """
    import time, random
    last_err = None
    for i in range(max_retries):
        try:
            log(f"📡 发起请求（第{i+1}/{max_retries}次）…")
            # Execute the provider call through the shared timeout wrapper.
            if fn is not None:
                return fn()

            # Retain payload dispatch for backward compatibility.
            if not isinstance(payload, dict):
                raise ValueError("call_with_retries: 需要提供 fn= 或 payload=dict(...)")
            if client is None:
                from openai import OpenAI
                client = OpenAI()

            # 优先 Responses（有 input / instructions / input_text 这种字段）
            if "input" in payload or "instructions" in payload:
                return client.responses.create(timeout=timeout, **payload)

            # 否则看看是不是 Chat 风格
            if "messages" in payload:
                return client.chat.completions.create(timeout=timeout, **payload)

            # 兜底：直接丢给 Responses，避免参数缺失
            return client.responses.create(timeout=timeout, **payload)

        except Exception as e:
            last_err = e
            log(f"⚠️ 第{i+1}次调用失败：{e}")
            if i == max_retries - 1:
                break
            # 指数退避 + 抖动
            sleep_s = base_delay * (2 ** i) * (0.8 + 0.4 * random.random())
            time.sleep(sleep_s)
    raise last_err or RuntimeError("调用失败次数过多")


def run_one(json_path: Path, out_dir: Path, model, prompt_head, prompt_tail,
            instructions, allow_missing,
            reasoning_effort=None, text_verbosity=None,
            trace_tokens: bool = False,
            inject_images: bool = True,
            image_detail: str = "auto",
            max_image_side: int = 512,
            dump_payload: bool = False,
            dry_run: bool = False,
            cot_enabled: bool = False,
            cot_hint: str = ""):

    data = json.loads(json_path.read_text(encoding="utf-8"))
    pages = normalize_pages(data)
    missing = []

    log("—— Baseline / Whole-Book ——")
    log(f"▶️ Baseline(Whole) 开始：{json_path.name}（共 {len(pages)} 页）")

    inputs = build_inputs(
        json_path, data, pages,
        prompt_head, prompt_tail,
        allow_missing, missing,
        model=model,
        inject_images=inject_images,
        image_detail=image_detail,
        max_image_side=max_image_side,
    )

    preview_path = None
    if dump_payload or dry_run:
        preview_path = _dump_payload_preview(
            request_messages=inputs,
            out_dir=out_dir,
            book_stem=json_path.stem,
            model=model,
            cot_enabled=cot_enabled,
            cot_hint=cot_hint,
        )

    if dry_run:
        log(f"🧪 Dry-run: payload preview saved (no request sent) → {preview_path}")
        return

    client = make_client_for_model(model)

    # trace：Baseline 全局 tokens + images 统计
    if trace_tokens:
        try:
            items = inputs[0]["content"]
        except Exception:
            items = []

        n_chars = 0
        img_lens = []

        for it in (items or []):
            t = it.get("type")
            if t in ("input_text", "text"):
                n_chars += len(it.get("text") or "")
            elif t in ("input_image", "image"):
                u = it.get("image_url") or ""
                img_lens.append(len(u))

        est_tokens = None
        try:
            import tiktoken
            enc = tiktoken.get_encoding("o200k_base")
            joined = []
            for it in (items or []):
                if it.get("type") in ("input_text", "text"):
                    joined.append(it.get("text") or "")
            est_tokens = len(enc.encode("\n".join(joined)))
        except Exception:
            est_tokens = int(n_chars / 2)

        print(f"🧾 Baseline tokens (estimated, text-only): ~{est_tokens} | chars={n_chars}", flush=True)

        if img_lens:
            total_len = sum(img_lens)
            avg_len = total_len // len(img_lens)
            max_len = max(img_lens)
            print(
                f"🖼️ Baseline images: n={len(img_lens)} | total_image_url_len={total_len} | "
                f"avg={avg_len} | max={max_len} | detail={image_detail}",
                flush=True,
            )

    result = call_with_retries(
        fn=lambda: call_openai_compat(
            client=client,
            model=model,
            messages=inputs,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity
        ),
        max_retries=5,
        base_delay=2.0
    )

    if missing:
        missing_info = "\n\n⚠️ 以下图片在处理过程中未找到/未提供，已跳过：\n" + "\n".join(missing)
        missing_info += "\n请确认路径是否正确或补充图片后重新运行。"
        print(missing_info)
        result += missing_info

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{json_path.stem}_WHOLE.txt"
    out_path.write_text(result, encoding="utf-8")
    log(f"✅ 已保存：{out_path}")

# ======== Book selection ========

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

        # 提取左右端的“尾部数字串”
        mL = re.search(r"(\d+)$", left)
        mR = re.search(r"(\d+)$", right)
        if mL and mR:
            s_num, e_num = mL.group(1), mR.group(1)
            a, b = int(s_num), int(e_num)
            if a > b:
                a, b = b, a

            # 左端“前缀”（把尾部数字去掉）与右端“前缀”
            prefixL = left[:mL.start(1)]
            prefixR = right[:mR.start(1)]

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
            for p in list(books_dir.glob("*.json")) + list(books_dir.glob("*.JSON")):
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


# ================================

def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_in = repo_root / "data" / "books"
    default_out = repo_root / "results" / "baselines"

    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(default_in), help="JSON 文件或目录（默认：./data/books）")
    ap.add_argument("--out", default=str(default_out), help="输出根目录（默认 baseline=./results/baselines；CoT 自动切到 ./results/cot）")
    ap.add_argument("--model", default="gpt-4o", help="模型名称（默认 gpt-4o）")
    ap.add_argument("--prompt-head-file", default="", help="可选：从文件加载 PROMPT_HEAD")
    ap.add_argument("--prompt-tail-file", default="", help="可选：从文件加载 PROMPT_TAIL")
    ap.add_argument("--instructions-file", default="", help="可选：从文件加载 instructions")
    ap.add_argument(
        "--strict-images", action="store_true",
        help="If set, missing images will raise FileNotFoundError. Default: skip missing images silently.",
    )
    ap.add_argument("--book-glob", default="", help="筛选书目：通配/单本/区间/点名多本，示例见文件头注释")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--text-verbosity", choices=["low", "medium", "high"], default=None)

    # ===== 便于与 MOSAIC / CoT 做对照实验 =====
    ap.add_argument("--trace-tokens", action="store_true",
                    help="打印输入规模统计（文本tokens估算 + 图片dataURL长度），不影响模型调用")
    ap.add_argument("--image-detail", type=str, default="auto",
                    choices=["auto", "low", "high"],
                    help="图片细节级别：auto/low/high（对 4o/4.1 与 gpt-5 均保留在 payload 中）")
    ap.add_argument("--max-image-side", type=int, default=512,
                    help="下采样最大边长（默认 512；与论文实验设置一致）")
    ap.add_argument("--no-inject-images", action="store_false", dest="inject_images",
                    help="禁用 baseline 中的图片注入，只保留逐页文字输入。")

    # CoT and payload-preview options.
    ap.add_argument("--cot", action="store_true",
                    help="启用 CoT 引导；默认输出目录自动切到 ./results/cot/<model>")
    ap.add_argument("--cot-hint", default=COT_HINT,
                    help="自定义 CoT 引导语")
    ap.add_argument("--dump-payload", action="store_true",
                    help="保存 baseline 请求预览（含文本截断预览与图片 URL 头部）")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅保存 payload preview，不实际发起模型请求")

    ap.set_defaults(inject_images=True)
    args = ap.parse_args()
    allow_missing_images = (not args.strict_images)

    path = Path(args.path)
    out = _resolve_out_dir(args.out, args.model, args.cot, repo_root)

    def read_file(path_str, default):
        return Path(path_str).read_text(encoding="utf-8") if path_str and Path(path_str).exists() else default

    raw_prompt_head = read_file(args.prompt_head_file, PROMPT_HEAD)
    prompt_head = _apply_cot_prompt(raw_prompt_head, args.cot, args.cot_hint)
    prompt_tail = read_file(args.prompt_tail_file, PROMPT_TAIL)
    instructions = read_file(args.instructions_file, DEFAULT_INSTRUCTIONS)

    log(f"🧠 CoT 模式：{'ON' if args.cot else 'OFF'}")
    if args.cot:
        log(f"📝 CoT 提示：{args.cot_hint}")
    if args.dry_run:
        log("🧪 Dry-run 模式：仅保存 payload preview，不发起模型请求。")

    if path.is_file() and path.suffix.lower() == ".json":
        # Single-book mode.
        files = [path]
        log(f"📚 本次将处理 1 本：{_fmt_books(files)}")
        log(f"📁 输出目录：{out}")
        run_one(path, out, args.model, prompt_head, prompt_tail, instructions, allow_missing_images,
                reasoning_effort=args.reasoning_effort,
                text_verbosity=args.text_verbosity,
                trace_tokens=args.trace_tokens,
                inject_images=args.inject_images,
                image_detail=args.image_detail,
                max_image_side=args.max_image_side,
                dump_payload=args.dump_payload,
                dry_run=args.dry_run,
                cot_enabled=args.cot,
                cot_hint=args.cot_hint)

    else:
        # Directory mode supports wildcard, range, and explicit selections.
        if not path.exists():
            raise FileNotFoundError(f"目录不存在：{path}")

        if args.book_glob:
            specs = _split_specs(args.book_glob)
            selected: List[Path] = []
            missing_specs: List[str] = []

            if len(specs) == 1 and any(ch in specs[0] for ch in "*?[]"):
                # 单一通配：glob
                selected = sorted(path.glob(specs[0]))
            else:
                # 范围 / 多选 / 单名
                seen = set()
                for sp in specs:
                    paths = _expand_specs(path, sp)
                    if paths:
                        for p in paths:
                            if p not in seen:
                                selected.append(p)
                                seen.add(p)
                    else:
                        missing_specs.append(sp)

            files = selected
            if not files:
                raise FileNotFoundError(f"未匹配到任何 .json：目录={path}，book_glob='{args.book_glob}'")
        else:
            # 方式1：通配全部（默认 *.json）
            files = list(path.glob("*.json"))
            if not files:
                raise FileNotFoundError(f"未在目录中发现 .json 文件：{path}")

        files = sorted(files, key=lambda x: x.name)

        # Print the resolved scope before model calls begin.
        log(f"📚 本次将处理 {len(files)} 本：{_fmt_books(files)}")
        log(f"📁 输出目录：{out}")

        for f in files:
            # Per-book status is printed by the runner below.
            run_one(f, out, args.model, prompt_head, prompt_tail, instructions, allow_missing_images,
                    reasoning_effort=args.reasoning_effort,
                    text_verbosity=args.text_verbosity,
                    trace_tokens=args.trace_tokens,
                    inject_images=args.inject_images,
                    image_detail=args.image_detail,
                    max_image_side=args.max_image_side,
                    dump_payload=args.dump_payload,
                    dry_run=args.dry_run,
                    cot_enabled=args.cot,
                    cot_hint=args.cot_hint)


if __name__ == "__main__":
    main()
