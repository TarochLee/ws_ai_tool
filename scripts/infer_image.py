# pic_pro_brief.py
# pic.jpg -> OCR(离线) -> 专业长文总结（讲义体：事实/推测/扩展/路线/行动）
# 依赖：pip install -U llama-cpp-python pillow pyobjc

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps, ImageEnhance
from llama_cpp import Llama

import Quartz
import Vision


IMG_PATH = "../pic.jpg"
TEXT_MODEL = "../models/GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"


# ---------- OCR ----------
def preprocess_for_ocr(src: str, out: str) -> str:
    im = Image.open(src).convert("RGB")
    im = ImageOps.grayscale(im)
    im = ImageOps.autocontrast(im)
    im = ImageEnhance.Contrast(im).enhance(1.6)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "JPEG", quality=95)
    return out


def ocr_macos_vision(image_path: str) -> str:
    im = Image.open(image_path).convert("RGB")
    data = im.tobytes()
    width, height = im.size
    bytes_per_row = width * 3

    color_space = Quartz.CGColorSpaceCreateDeviceRGB()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    cgimg = Quartz.CGImageCreate(
        width, height, 8, 24, bytes_per_row,
        color_space,
        Quartz.kCGBitmapByteOrderDefault,
        provider,
        None, False,
        Quartz.kCGRenderingIntentDefault
    )

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    req.setRecognitionLanguages_(["zh-Hans", "en-US"])

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")

    lines: List[str] = []
    for r in (req.results() or []):
        top = r.topCandidates_(1)
        if top and len(top) > 0:
            s = str(top[0].string()).strip()
            if s:
                lines.append(s)
    return "\n".join(lines)


# ---------- 清洗（通用，不靠“每个领域的关键词表”） ----------
_NOISE_PATTERNS = [
    r"大家都在搜", r"换一换", r"广告", r"立即体验", r"发私信", r"关注他",
    r"赞同", r"收藏", r"分享", r"评论", r"热", r"新", r"关注者", r"回答",
]

def _is_noise_line(line: str) -> bool:
    if len(line) <= 1:
        return True
    if re.fullmatch(r"[\W_]+", line):
        return True
    for p in _NOISE_PATTERNS:
        if re.search(p, line, flags=re.IGNORECASE):
            return True
    return False


def _line_score(line: str) -> float:
    s = line.strip()
    if not s:
        return -1.0
    length = len(s)
    zh = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    alnum = sum(1 for ch in s if ch.isalnum())
    readable_ratio = (zh + alnum) / max(length, 1)

    score = 0.0
    score += min(length, 180) * 0.02
    score += readable_ratio * 2.0
    # 解释型/定义型句式加分（通用）
    if re.search(r"(是什么|用于|通过|因此|可以|形成|测量|单位|误差|积分|校正|标定|对齐|外参)", s):
        score += 0.7
    if length <= 4:
        score -= 0.8
    return score


def clean_ocr_text(raw: str, max_lines: int = 320, max_chars: int = 11000) -> str:
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln and not _is_noise_line(ln)]

    # 保序去重
    seen = set()
    uniq: List[str] = []
    for ln in lines:
        if ln not in seen:
            uniq.append(ln)
            seen.add(ln)

    scored: List[Tuple[float, int, str]] = [(_line_score(ln), i, ln) for i, ln in enumerate(uniq)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = scored[:max_lines]
    picked.sort(key=lambda x: x[1])

    text = "\n".join(x[2] for x in picked)

    # 轻度合并断行
    merged: List[str] = []
    buf = ""
    for ln in text.splitlines():
        if buf and len(ln) <= 18 and not re.search(r"[。！？:：]$", buf):
            buf += ln
        else:
            if buf:
                merged.append(buf)
            buf = ln
    if buf:
        merged.append(buf)

    text = "\n".join(merged)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[已截断]"
    return text


# ---------- 专业长文生成 ----------
def generate_professional_brief(ocr_text: str) -> str:
    llm = Llama(
        model_path=TEXT_MODEL,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )

    prompt = (
        "你将收到一段从截图OCR提取的文本，可能有噪声或不完整。请写一份“专业、详细、讲义体”的技术笔记，"
        "面向具备理工科背景的读者，允许使用 SLAM/IMU/预积分/外参标定 等术语。\n"
        "写作与结构要求（必须遵守）：\n"
        "1) 用 Markdown 分级标题组织内容；每个小节至少3-6句，句子不要过短；\n"
        "2) 明确区分【可从OCR确定的事实】与【合理推测/假设】；推测必须带“可能/推测”措辞并引用OCR中的证据短语；\n"
        "3) 在“扩展知识”里补充与主题高度相关的背景：例如 IMU 误差来源、积分漂移、预积分目的、与视觉/激光融合时的常见工程坑等；\n"
        "4) 给出“学习路线”与“实践清单”：各不少于5条，且尽量具体（例如可做的实验/验证点/对比指标）；\n"
        "5) 不要捏造OCR中不存在的具体数据、人物言论或引用来源。\n"
        "——OCR文本开始——\n"
        f"{ocr_text}\n"
        "——OCR文本结束——\n"
    )

    res = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,   # 兼顾严谨与扩写
        max_tokens=1200,   # 输出更长、更细
    )
    return res["choices"][0]["message"]["content"].strip()


def main() -> None:
    if not Path(IMG_PATH).exists():
        raise FileNotFoundError(f"找不到 {IMG_PATH}")

    ocr_img = preprocess_for_ocr(IMG_PATH, "tmp/pic_ocr.jpg")
    raw = ocr_macos_vision(ocr_img)
    cleaned = clean_ocr_text(raw)

    Path("ocr.txt").write_text(cleaned, encoding="utf-8")

    if len(cleaned.strip()) < 60:
        out = "OCR文本过短，无法生成可靠的专业长文。"
    else:
        out = generate_professional_brief(cleaned)

    Path("output.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()