import os, re, json, argparse, sys, traceback
from pathlib import Path
import cv2, numpy as np
from paddleocr import PaddleOCR

# ---------- 说明 ----------
# 功能：按类别/书本进行选择性处理；OCR+抹字；输出清洁图片与JSON
# Preserve the source directory structure and record fields.

# ---------- 工具 ----------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)

def list_images(dirpath: str):
    exts = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}
    try:
        files = [f for f in os.listdir(dirpath) if Path(f).suffix.lower() in exts]
    except FileNotFoundError:
        return []
    files.sort(key=natural_key)
    return [str(Path(dirpath)/f) for f in files]

def parse_book_folder(name: str):
    """
    将 'book_X_绘本名称' 解析为 ('book_X', '绘本名称'); 否则返回 (name, name)
    """
    m = re.match(r"^(book_\d+)[_\-\s]+(.+)$", name.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return name.strip(), name.strip()

def ocr_engine():
    # Chinese recognition with angle classification.
    return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

def upscale(img_bgr: np.ndarray, scale: float):
    if scale and scale > 1.0:
        h, w = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return img_bgr

def poly_scale(poly, s: float):
    return [[p[0]*s, p[1]*s] for p in poly]

def polys_for_mask_from_lines(lines, img_h, img_w, mask_min_prob=0.90, max_area_frac=0.08, max_h_factor=2.5):
    """
    用于抹字的poly过滤：更严格、更保守，避免插画纹理误检导致糊脸。
    """
    if not lines:
        return []

    # 先收集所有候选框高度，用于求中位数
    hs = []
    tmp = []
    for poly, (txt, prob) in lines:
        p = float(prob or 0.0)
        if p < mask_min_prob:
            continue
        xs = [pt[0] for pt in poly]; ys = [pt[1] for pt in poly]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        h = max(1.0, y1 - y0)
        w = max(1.0, x1 - x0)
        area = h * w
        tmp.append((poly, h, area))
        hs.append(h)

    if not tmp:
        return []

    med_h = float(np.median(hs)) if hs else 1.0
    area_limit = max_area_frac * float(img_h * img_w)
    h_limit = max_h_factor * med_h

    out = []
    for poly, h, area in tmp:
        if area > area_limit:
            continue
        if h > h_limit:
            continue
        out.append(poly)
    return out

def mask_from_polys(h, w, polys, dilate=3):
    mask = np.zeros((h,w), dtype=np.uint8)
    for pts in polys:
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    if dilate and dilate>0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate,dilate))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def bottom_textband_mask(img_bgr: np.ndarray, y0_frac: float=0.55):
    h, w = img_bgr.shape[:2]
    y0 = max(0, min(h-1, int(h*y0_frac)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    band = gray[y0:,:]
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
    blackhat = cv2.morphologyEx(band, cv2.MORPH_BLACKHAT, k1)
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k2, iterations=2)
    th = cv2.dilate(th, np.ones((3,3), np.uint8), iterations=1)
    mask = np.zeros((h,w), dtype=np.uint8)
    mask[y0:,:] = th
    return mask

def inpaint(img_bgr, mask, method="ns", radius=2):
    flag = cv2.INPAINT_TELEA if method=="telea" else cv2.INPAINT_NS
    return cv2.inpaint(img_bgr, mask, radius, flag)

# Text cleanup, paragraph merging, and alphanumeric-noise filtering.

def normalize_zh_inline_spaces(t: str) -> str:
    """
    清理中文场景下的多余空格：
    - 去掉中文/字母/数字 与 中文标点之间的空格
    - 去掉引号/括号内部或外部紧邻的空格
    - 合并多空格为无空格（中文默认不留空）
    """
    if not t: return t
    t = t.strip()
    punct = r"，。、；：？！“”‘’（）《》【】…—,.!?;:\"'()[]<>"
    t = re.sub(rf"\s+([{punct}])", r"\1", t)  # 标点前空格
    t = re.sub(rf"([{punct}])\s+", r"\1", t)  # 标点后空格
    t = re.sub(r"[ \t\u00A0]+", "", t)        # 合并空格为无
    return t

# —— 判定是否含中文（CJK）
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")

def has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))

# —— 过滤段内/段首/段尾的页码与英数噪声
def strip_pagecode_tokens(s: str) -> str:
    """
    去掉常见页码/英数噪声：
    - 段首：04128 / 09|28 / 12/28 / 1728 / ABC123 / 123ABC 等
    - 段尾：... ABC12345 / ... 1728 / ... 09|28
    只在去掉后“剩余部分含中文”时才生效，避免误删正常英文句子。
    """
    if not s: return s
    t = s.strip()

    # 段首：页码或英数串
    m = re.match(r"^\s*(?:\d{1,3}\s*[\|/:.\-]?\s*\d{1,3}|[A-Za-z]{1,4}\s*\d{1,5}|\d{1,5}[A-Za-z]{1,4})\s*", t)
    if m:
        rest = t[m.end():].lstrip()
        if has_cjk(rest):
            t = rest

    # 段尾：页码或英数串
    m2 = re.search(r"(?:[A-Za-z]{1,4}\s*\d{1,5}|\d{1,5}[A-Za-z]{1,4}|[\d\|/:.\-]{2,10})\s*$", t)
    if m2:
        head = t[:m2.start()].rstrip()
        if has_cjk(head):
            t = head

    return t.strip()

def clean_page_text_noise(page_text: str) -> str:
    """
    逐段清理：
    1) 去掉段首/段尾页码与英数噪声
    2) 纯英数符号段落（无中文）整体丢弃
    3) 内联中文空格再清理
    """
    if not page_text:
        return page_text
    out_paras = []
    for para in page_text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        p = strip_pagecode_tokens(p)

        # 如果整个段落无中文（例如 ABC12345 / 01|28），认为是噪声段落，直接跳过
        if not has_cjk(p):
            continue

        p = normalize_zh_inline_spaces(p)
        if p:
            out_paras.append(p)
    return "\n\n".join(out_paras)

def merge_ocr_lines_to_paragraphs(
    lines,
    min_prob=0.8,
    x_overlap=0.35,        # 仍保留，但仅作为辅因子
    vgap_factor=1.3,       # 段内允许的行距倍数
    line_merge_factor=0.6  # 行内聚类阈值
):
    """
    列检测 + 段内分行排序：
      1) 先按 x 中线做“列”切分（1~2列），避免跨列误合；
      2) 列内按“垂直间距”切段（更依赖 vgap，而非强依赖水平重叠）；
      3) 段内先“分行”（y接近），行内再按 x 排序；
      4) 中文内联空格清理，段间以空行分隔。
    """
    # ---------- 抽取框 ----------
    items = []
    for poly, (txt, prob) in (lines or []):
        if float(prob or 0.0) < min_prob:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        h = max(1.0, y1 - y0)
        t = re.sub(r"\s*\n\s*", " ", (txt or "").strip())
        t = re.sub(r"[ \t\u00A0]+", " ", t).strip()
        if not t:
            continue
        # Discard OCR boxes containing only page numbers or alphanumeric noise.
        if not has_cjk(t):
            continue
        items.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "h": h, "xmid": 0.5*(x0+x1), "t": t
        })
    if not items:
        return ""

    items.sort(key=lambda d: (d["y0"], d["x0"]))

    # ---------- 粗略列检测（最多两列） ----------
    xmids = sorted(it["xmid"] for it in items)
    widths = [it["x1"]-it["x0"] for it in items]
    medw = np.median(widths) if widths else 1.0
    split_x = None
    if len(xmids) >= 3:
        gaps = [xmids[i+1] - xmids[i] for i in range(len(xmids)-1)]
        i_max = int(np.argmax(gaps))
        if gaps[i_max] > 1.1 * medw:  # 大间隙 → 两列
            split_x = 0.5 * (xmids[i_max] + xmids[i_max+1])

    if split_x is None:
        columns = [items]
    else:
        left  = [it for it in items if it["xmid"] <  split_x]
        right = [it for it in items if it["xmid"] >= split_x]
        columns = [c for c in (left, right) if c]  # 防御：避免某列误空

    # ---------- 列内：按垂直间距切“段” ----------
    col_paragraphs = []  # [(col_idx, top, text), ...]

    for col_idx, col in enumerate(columns):
        col.sort(key=lambda d: (d["y0"], d["x0"]))
        med_h = np.median([it["h"] for it in col]) if col else 1.0
        para_vgap = vgap_factor * med_h

        current = []
        last_y1 = None
        current = []
        last_y1 = None
        for it in col:
            if last_y1 is None:
                current = [it]
                last_y1 = it["y1"]
                continue

            # 与上一行/框的垂直间距
            vgap = it["y0"] - last_y1

            # 若垂直间距过大，认为换段
            if vgap > para_vgap:
                p = _finalize_paragraph(current)
                col_paragraphs.append((col_idx, p["top"], p["text"]))
                current = [it]
            else:
                current.append(it)

            last_y1 = max(last_y1, it["y1"])

        # 收尾
        if current:
            p = _finalize_paragraph(current)
            col_paragraphs.append((col_idx, p["top"], p["text"]))

    # 关键：如果有 split_x（即两列），严格按 col_idx 排序（左→右），每列内部再按 top（上→下）
    # 如果没有 split_x，就等价于单列：col_idx 全是0，仍然按 top
    col_paragraphs.sort(key=lambda x: (x[0], x[1]))
    return "\n\n".join(t for _, _, t in col_paragraphs)

def _finalize_paragraph(items_in_para):
    """ 将段内 items 分行→行内按x→合并成中文文本，返回 {top, text} """
    items_in_para.sort(key=lambda d: (d["y0"], d["x0"]))
    h_ref = np.median([it["h"] for it in items_in_para]) if items_in_para else 1.0
    lines_in_para = []
    for it in items_in_para:
        placed = None
        for ln in lines_in_para:
            if abs(it["y0"] - ln["y_ref"]) <= 0.6 * max(it["h"], h_ref):  # 行聚类自适应
                placed = ln; break
        if placed is None:
            lines_in_para.append({"y_ref": it["y0"], "items": [it]})
        else:
            placed["items"].append(it)
            placed["y_ref"] = 0.5 * (placed["y_ref"] + it["y0"])
    lines_in_para.sort(key=lambda ln: ln["y_ref"])

    line_texts = []
    for ln in lines_in_para:
        ln["items"].sort(key=lambda d: d["x0"])
        txt_line = "".join([d["t"] for d in ln["items"]])      # 中文不加空格
        txt_line = normalize_zh_inline_spaces(txt_line)
        line_texts.append(txt_line)

    para_text = " ".join(line_texts)                            # 行与行之间留 1 空格
    para_text = normalize_zh_inline_spaces(para_text)           # 再清一次
    top = min(it["y0"] for it in items_in_para)
    return {"top": top, "text": para_text}

# ---------- 处理单本书 ----------

def process_one_book(ocr, book_src_dir, book_id, title, book_type,
                     picture_root, books_root, scale, min_prob, text_band_y, dilate, radius, inpaint_method,
                     x_overlap, vgap_factor, line_merge_factor,
                     use_textband=False, mask_min_prob=0.90, max_mask_area_frac=0.08, max_mask_h_factor=2.5,
                     verbose=True):

    images = list_images(book_src_dir)
    if not images:
        print(f"[WARN] 缺图：{book_id}（目录存在但未发现图片） -> {book_src_dir}")
        return {"book_id": book_id, "status":"no_images", "pages":0}

    # 输出目录
    picture_out_dir = str(Path(picture_root)/book_id)
    ensure_dir(picture_out_dir)
    ensure_dir(books_root)

    pages=[]
    n_total = len(images)
    for idx, img_path in enumerate(images, start=1):
        try:
            img0 = cv2.imread(img_path)
            if img0 is None:
                print(f"[WARN] 读图失败: {img_path}")
                continue

            scale_applied = scale if scale and scale>0 else 1.0
            img_big = upscale(img0, scale_applied)
            res = ocr.ocr(img_big, cls=True)
            lines_big = res[0] if res and res[0] else []

            # 把OCR框从“放大坐标系”映射回“原图坐标系”
            back_s = 1.0 / float(scale_applied)
            lines = []
            for poly, (txt, prob) in (lines_big or []):
                poly_small = poly_scale(poly, back_s)
                lines.append((poly_small, (txt, prob)))

            # 几何聚段 + 段内分行排序（建议也用回原图坐标系的 lines，逻辑更一致）
            page_text = merge_ocr_lines_to_paragraphs(
                lines,
                min_prob=min_prob,
                x_overlap=x_overlap,
                vgap_factor=vgap_factor,
                line_merge_factor=line_merge_factor
            )

            page_text = clean_page_text_noise(page_text)

            h, w = img0.shape[:2]

            # 生成用于抹字的 polys（此时 lines 已是原图坐标）
            polys_mask = polys_for_mask_from_lines(
                lines, h, w,
                mask_min_prob=mask_min_prob,
                max_area_frac=max_mask_area_frac,
                max_h_factor=max_mask_h_factor
            )

            mask1 = mask_from_polys(h, w, polys_mask, dilate=dilate) if polys_mask else np.zeros((h, w), np.uint8)

            # 2) 默认不启用 textband；启用时也要“底部确实检测到文字”才补抹
            mask = mask1
            if use_textband:
                y0 = int(h * text_band_y)
                # 如果 OCR 框在底部带（y > y0）几乎没有覆盖，就不要用 textband，避免误抹插画/表情
                bottom_has_text = False
                for poly in polys_mask:
                    ys = [pt[1] for pt in poly]
                    if min(ys) >= y0:
                        bottom_has_text = True
                        break
                if bottom_has_text:
                    mask2 = bottom_textband_mask(img0, y0_frac=text_band_y)
                    mask = cv2.bitwise_or(mask1, mask2)

            cleaned = inpaint(img0, mask, method=inpaint_method, radius=radius)

            fname = Path(img_path).name
            cv2.imwrite(str(Path(picture_out_dir)/fname), cleaned)
            pages.append({"page": idx, "text": page_text, "image": fname})

            kept = (page_text.count("\n\n")+1) if page_text else 0  # 段落数估计
            pct  = 100.0*idx/n_total
            print(f"[OK] {book_id} p{idx:02d}/{n_total} ({pct:5.1f}%) | ocr={len(lines)} paras_kept≈{kept}")
        except Exception as e:
            print(f"[ERR] {book_id} p{idx:02d}/{n_total} 处理失败：{e}")
            traceback.print_exc(file=sys.stdout)

    # JSON files are written directly under books_root, so picture_dir must be
    # relative to books_root (the JSON file's parent directory).
    books_root_abs = str(Path(books_root).resolve())
    picture_out_dir_abs = str(Path(picture_out_dir).resolve())
    picture_dir_rel = os.path.relpath(picture_out_dir_abs, start=books_root_abs).replace("\\", "/")

    out = {
        "book_id": book_id,
        "title": title,
        "book_type": book_type,
        "picture_dir": picture_dir_rel,
        "pages": pages
    }

    json_path = str(Path(books_root)/f"{book_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"==> JSON 写入: {json_path}  | picture_dir='{picture_dir_rel}'")

    status = "ok" if pages else "empty"
    return {"book_id": book_id, "status": status, "pages": len(pages)}

# ---------- 选择器与遍历 ----------

def parse_books_selector(selector: str):
    """
    解析 --books 的选择表达式：
    - 逗号分隔：book_1,book_3,book_7..book_12
    - 支持区间：book_7..book_12（按编号闭区间）
    - 也允许直接写目录名（将用前缀匹配）
    返回：{"ids": set([book_1,...]), "ranges":[(7,12),...], "raw_tokens":[...]}
    """
    out_ids = set()
    ranges = []
    tokens = []
    if not selector:
        return {"ids": out_ids, "ranges": ranges, "tokens": tokens}

    for tok in [t.strip() for t in selector.split(",") if t.strip()]:
        tokens.append(tok)
        mrange = re.match(r"^(book_)?(\d+)\.\.(book_)?(\d+)$", tok, flags=re.IGNORECASE)
        if mrange:
            a = int(mrange.group(2)); b = int(mrange.group(4))
            if a > b: a,b = b,a
            ranges.append((a,b))
            continue
        mid = re.match(r"^(book_\d+)$", tok, flags=re.IGNORECASE)
        if mid:
            out_ids.add(mid.group(1))
            continue
    return {"ids": out_ids, "ranges": ranges, "tokens": tokens}

def id_from_name(book_dir_name: str):
    bid, _ = parse_book_folder(book_dir_name)
    return bid

def take_by_selector(book_dirs, selector_obj, fuzzy=True):
    """
    根据 selector 结果在给定的书目录列表中过滤。
    book_dirs: [Path(...), ...]（每个为一本书的目录Path）
    """
    if not selector_obj or (not selector_obj["ids"] and not selector_obj["ranges"] and not selector_obj["tokens"]):
        return book_dirs  # 未指定 => 全部

    ids = selector_obj["ids"]
    ranges = selector_obj["ranges"]
    tokens = selector_obj["tokens"]

    picks = []
    for d in book_dirs:
        name = d.name
        bid, title = parse_book_folder(d.name)
        keep = False
        if bid in ids:
            keep = True
        if not keep and ranges:
            mid = re.match(r"^book_(\d+)$", bid, flags=re.IGNORECASE)
            if mid:
                num = int(mid.group(1))
                for (a,b) in ranges:
                    if a <= num <= b:
                        keep = True
                        break
        if not keep and fuzzy and tokens:
            for tok in tokens:
                if tok.lower().startswith("book_"):
                    if tok.lower() == bid.lower():
                        keep = True; break
                elif tok.isdigit():
                    if bid.lower() == f"book_{tok}".lower():
                        keep = True; break
                if tok in name or tok in title:
                    keep = True; break
        if keep:
            picks.append(d)
    return picks

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial_root", default="./inputs/raw_books")
    ap.add_argument("--picture_root", default="./outputs/ocr/images")
    ap.add_argument("--books_root", default="./outputs/ocr/books")
    # 选择器
    ap.add_argument("--types", default="", help="仅处理这些类别（逗号分隔），例如：情景教育类,益智类")
    ap.add_argument("--books", default="", help="选择书本：支持列表与区间，如：book_2,book_5..book_8 或 2..8，或关键字/目录名/书名片段")
    ap.add_argument("--list", action="store_true", help="仅列出将要处理的书，不执行")
    ap.add_argument("--dry_run", action="store_true", help="打印即将处理的对象，但不写文件")
    # 抹字 & OCR 参数
    ap.add_argument("--scale", type=float, default=1.8)
    ap.add_argument("--min_prob", type=float, default=0.80)
    ap.add_argument("--use_textband", action="store_true", help="启用底部文字带补抹（默认关闭，避免误抹插画/表情）")
    ap.add_argument("--text_band_y", type=float, default=0.60, help="底部文字带起始位置（高度比例）")
    ap.add_argument("--dilate", type=int, default=2, help="OCR框mask膨胀核尺寸（建议2~3）")
    ap.add_argument("--radius", type=int, default=1, help="inpaint半径（建议1，避免糊）")

    # 仅用于“抹字”的OCR框过滤（避免插画纹理误检）
    ap.add_argument("--mask_min_prob", type=float, default=0.90, help="用于mask的最小置信度（比文本更严格）")
    ap.add_argument("--max_mask_area_frac", type=float, default=0.08, help="单个mask框最大面积占比（防止大片误抹）")
    ap.add_argument("--max_mask_h_factor", type=float, default=2.5, help="单个mask框高度上限=中位高度*该倍数")

    ap.add_argument("--inpaint", default="ns", choices=["telea","ns"])
    # 段落几何合并阈值（可由 sh 传入）
    ap.add_argument("--x_overlap", type=float, default=0.35, help="段内水平重叠阈值")
    ap.add_argument("--vgap_factor", type=float, default=1.3, help="段内允许的行距倍数")
    ap.add_argument("--line_merge_factor", type=float, default=0.6, help="行聚类阈值")
    args = ap.parse_args()

    initial_root = Path(args.initial_root)
    if not initial_root.exists():
        raise SystemExit(f"初始目录不存在: {initial_root}")

    wanted_types = [t.strip() for t in args.types.split(",") if t.strip()]
    sel = parse_books_selector(args.books)

    print("==============================================")
    print("📚 绘本清理与OCR 处理")
    print(f"📁 初始目录: {initial_root}")
    print(f"🎯 类别筛选: {wanted_types if wanted_types else '[全部]'}")
    print(f"🎯 书本筛选: {args.books if args.books else '[全部]'}")
    print("==============================================")

    all_tasks = []
    all_types = sorted([d for d in initial_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    for cat in all_types:
        book_type = cat.name.strip()
        if wanted_types and (book_type not in wanted_types):
            continue
        book_dirs = sorted([d for d in cat.iterdir() if d.is_dir()], key=lambda p: p.name)
        book_dirs = take_by_selector(book_dirs, sel, fuzzy=True)
        for d in book_dirs:
            all_tasks.append((book_type, d))

    if not all_tasks:
        print("⚠️ 未匹配到任何待处理书本（请检查 --types / --books 选择条件）。")
        return

    print(f"📋 将要处理的书本（共 {len(all_tasks)} 本）：")
    for i,(bt,d) in enumerate(all_tasks, start=1):
        bid, title = parse_book_folder(d.name)
        print(f"  {i:3d}. [{bt}] {bid}  ——  {title}   <{d}>")

    if args.list:
        print("（已按 --list 仅列出，不执行）")
        return
    if args.dry_run:
        print("（已按 --dry_run 仅预览，不执行写入）")
        return

    print("==============================================")
    print("🚀 开始处理 ...")
    print("==============================================")

    ocr = ocr_engine()
    ensure_dir(args.picture_root)
    ensure_dir(args.books_root)

    ok, empty, noimg, fail = 0, 0, 0, 0
    for idx,(book_type, book_dir) in enumerate(all_tasks, start=1):
        bid, title = parse_book_folder(book_dir.name)
        print("----------------------------------------------")
        print(f"▶️  {idx}/{len(all_tasks)}  开始处理：[{book_type}] {bid} —— {title}")
        print(f"    源：{book_dir}")
        try:
            res = process_one_book(
                ocr=ocr,
                book_src_dir=str(book_dir),
                book_id=bid,
                title=title,
                book_type=book_type,
                picture_root=str(Path(args.picture_root)),
                books_root=str(Path(args.books_root)),
                scale=args.scale,
                min_prob=args.min_prob,
                text_band_y=args.text_band_y,
                dilate=args.dilate,
                radius=args.radius,
                inpaint_method=args.inpaint,
                x_overlap=args.x_overlap,
                vgap_factor=args.vgap_factor,
                line_merge_factor=args.line_merge_factor,
                verbose=True,
                use_textband = args.use_textband,
                mask_min_prob = args.mask_min_prob,
                max_mask_area_frac = args.max_mask_area_frac,
                max_mask_h_factor = args.max_mask_h_factor
            )
            st = res.get("status","")
            if st=="ok":
                ok += 1
            elif st=="empty":
                empty += 1
            elif st=="no_images":
                noimg += 1
            else:
                ok += 1  # 容错：未知状态按ok计
        except Exception as e:
            fail += 1
            print(f"[FATAL] 处理失败：{bid} —— {e}")
            traceback.print_exc(file=sys.stdout)

    print("==============================================")
    print("✅ 处理完成！汇总：")
    print(f"   成功：{ok} 本")
    print(f"   空页：{empty} 本（无有效页面写入）")
    print(f"   缺图：{noimg} 本（目录存在但无图片）")
    print(f"   失败：{fail} 本")
    print("==============================================")

if __name__ == "__main__":
    main()
