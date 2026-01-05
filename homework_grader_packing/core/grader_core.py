
"""
grader_core.py

目标：
- 把 app.py 里的“核心能力”（OCR -> 评分 LLM -> 结果落盘/错题本）抽出来，供 Streamlit / CLI / batch_app 复用
- 不依赖 streamlit（没有 st.cache_*），可在任意 Python 环境直接 import 使用

你现在的约束：
- 50 份左右图片作业
- 云端 API 并发限制 2
- 不用 zip 上传：前端一次传多个文件即可（st.file_uploader(accept_multiple_files=True)）
- 输出到 ./runs/<timestamp>/，每个学生一个文件夹；错题本也按学生名分目录
"""

from __future__ import annotations

# ---- Output compaction helpers (avoid token blow-up) ----
def _compact_text(s: str, max_len: int = 60) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = re.sub(r"\s+", " ", s)
    if len(s) <= max_len:
        return s
    # Try keep final result (often at the end)
    return "…" + s[-max_len:]

def compact_item_fields(item: dict, ans_max: int = 40, cmt_max: int = 40) -> dict:
    if not isinstance(item, dict):
        return item
    ra = item.get("recognized_answer")
    cm = item.get("comment")
    if isinstance(ra, str):
        item["recognized_answer"] = _compact_text(ra, ans_max)
    if isinstance(cm, str):
        item["comment"] = _compact_text(cm, cmt_max)
    # normalize unknown->uncertain if present
    if "unknown" in item and "uncertain" not in item:
        item["uncertain"] = bool(item.get("unknown"))
        item.pop("unknown", None)
    return item

import base64
import hashlib
import hmac
import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from email.utils import formatdate
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import requests
from PIL import Image, ImageEnhance
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -------------------------
# Small utils
# -------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def safe_filename(name: str, fallback: str = "unknown") -> str:
    """Make a filesystem-safe folder/file name."""
    name = (name or "").strip()
    if not name:
        return fallback
    # remove illegal characters (Windows + POSIX safe-ish)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:80] if len(name) > 80 else name


# -------------------------
# Config
# -------------------------
@dataclass
class AppConfig:
    provider: str = "spark_http"  # mock | cloud_ocr_only | spark_http

    # Spark OpenAPI (HTTP)
    spark_http_url: str = ""
    spark_api_password: str = ""
    spark_model: str = "x1"

    # LLM defaults (Spark HTTP)
    llm_timeout_sec: int = 180
    llm_min_timeout_sec: int = 120
    llm_min_timeout_json_sec: int = 180
    llm_max_tokens: int = 1800
    llm_temperature: float = 0.0
    llm_top_k: int = 1

    # Cloud OCR (OCRforLLM WebAPI)
    ocr_url: str = ""
    ocr_app_id: str = ""
    ocr_api_key: str = ""
    ocr_api_secret: str = ""


# -------------------------
# HTTP session w/ retries
# -------------------------
def make_http_session(pool: int = 10) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=0,  # do not retry read timeouts (handle at app level)

        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# -------------------------
# Image preprocess (speed + quality)
# -------------------------
def preprocess_for_ocr(
    image_bytes: bytes,
    crop_ratio: float = 0.82,
    max_side: int = 1600,
    jpeg_quality: int = 75,
    enhance_contrast: float = 1.10,
) -> bytes:
    """
    Practical OCR preprocessing:
    - center-crop (keeps main content; avoids margins / page numbers)
    - resize so long edge <= max_side
    - mild contrast enhance
    - export to JPEG (smaller upload)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    crop_ratio = max(0.5, min(1.0, crop_ratio))
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))

    w2, h2 = img.size
    scale = min(max_side / max(w2, h2), 1.0)
    if scale < 1.0:
        img = img.resize((int(w2 * scale), int(h2 * scale)))

    if enhance_contrast and enhance_contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(enhance_contrast)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    return out.getvalue()


# -------------------------
# OCR helpers (same structure as app.py)
# -------------------------
def _build_ocr_signed_url(ocr_url: str, api_key: str, api_secret: str) -> str:
    u = urlparse(ocr_url)
    host = u.netloc
    path = u.path
    date = formatdate(timeval=None, localtime=False, usegmt=True)
    signature_origin = f"host: {host}\n" f"date: {date}\n" f"POST {path} HTTP/1.1"

    sig_bytes = hmac.new(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    signature = base64.b64encode(sig_bytes).decode("utf-8")

    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    return (
        f"{ocr_url}?"
        f"authorization={quote(authorization)}&"
        f"host={quote(host)}&"
        f"date={quote(date)}"
    )


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_text_from_ocr_obj(obj: Any) -> str:
    """
    Best-effort extraction from OCR JSON structures.
    Tries common keys: lines/regions/textline/paragraph/pages/blocks/items, etc.
    """
    if obj is None:
        return ""

    if isinstance(obj, dict) and isinstance(obj.get("text"), str) and obj["text"].strip():
        return obj["text"].strip()

    lines: List[str] = []

    def walk(o: Any):
        if o is None:
            return
        if isinstance(o, str):
            t = o.strip()
            if t:
                lines.append(t)
            return
        if isinstance(o, dict):
            for k in ["line", "content", "txt", "words", "w", "value"]:
                v = o.get(k)
                if isinstance(v, str) and v.strip():
                    lines.append(v.strip())
            for k in ["lines", "textline", "textlines", "paragraph", "paragraphs", "blocks", "regions", "pages", "items", "data", "result"]:
                if k in o:
                    walk(o[k])
            for v in o.values():
                if isinstance(v, (dict, list)):
                    walk(v)
            return
        if isinstance(o, list):
            for it in o:
                walk(it)

    walk(obj)

    seen = set()
    out = []
    for t in lines:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return "\n".join(out).strip()


def decode_ocr_payload_text(text_b64: str) -> str:
    if not text_b64:
        return ""
    decoded = base64.b64decode(text_b64).decode("utf-8", errors="ignore").strip()
    if not decoded:
        return ""
    obj = _safe_json_loads(decoded)
    if obj is not None:
        t = extract_text_from_ocr_obj(obj)
        return t if t else decoded
    return decoded


def cloud_ocr(image_bytes: bytes, cfg: AppConfig, *, session: Optional[requests.Session] = None, timeout: int = 35) -> Dict[str, Any]:
    """
    Returns dict: {"raw_text": str, "extracted_text": str, "resp": dict}
    """
    if not (cfg.ocr_url and cfg.ocr_app_id and cfg.ocr_api_key and cfg.ocr_api_secret):
        raise RuntimeError("OCR 配置缺失：需要 OCR_URL / OCR_APP_ID / OCR_API_KEY / OCR_API_SECRET")

    sess = session or make_http_session()
    signed_url = _build_ocr_signed_url(cfg.ocr_url, cfg.ocr_api_key, cfg.ocr_api_secret)
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    body = {
        "header": {"app_id": cfg.ocr_app_id, "uid": "hwgrader", "status": 0},
        "parameter": {
            "ocr": {
                "result_option": "normal,char",
                "result_format": "json",
                "output_type": "one_shot",
                "exif_option": "1",
                "alpha_option": "1",
                "rotation_min_angle": 3,
                "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
            }
        },
        "payload": {"image": {"encoding": "jpg", "image": img_b64, "status": 0, "seq": 0}},
    }

    r = sess.post(signed_url, json=body, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"OCR HTTP {r.status_code}: {r.text}")

    data = r.json()
    header = data.get("header", {})
    if header.get("code", 0) != 0:
        raise RuntimeError(f"OCR error: {header}")

    payload = data.get("payload", {}).get("result", {})
    raw_text = decode_ocr_payload_text(payload.get("text", ""))

    obj = _safe_json_loads(raw_text)
    extracted = extract_text_from_ocr_obj(obj) if obj is not None else raw_text
    return {"raw_text": raw_text, "extracted_text": extracted, "resp": data}


# -------------------------
# LLM helpers (Spark HTTP)
# -------------------------

def spark_http_chat(
    cfg: AppConfig,
    messages: List[Dict[str, str]],
    *,
    session: Optional[requests.Session] = None,
    timeout_sec: int = 180,
    force_json: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    retries: int = 1,
    no_retry_on_timeout: bool = True,
    return_fallback_json_on_timeout: bool = True,
) -> str:
    """
    Spark OpenAPI HTTP 调用（批量友好版）：
    - read timeout 默认 180s（避免 60s 频繁超时）
    - 超时默认不重试（批量时一张慢卷子不拖死整批）
    - 可选返回兜底 JSON，保证上层解析不会直接崩
    """
    if not cfg.spark_http_url:
        raise RuntimeError("未配置 SPARK_HTTP_URL")
    if not cfg.spark_api_password:
        raise RuntimeError("未配置 SPARK_API_PASSWORD（APIPassword）")

    sess = session or make_http_session()

    # 允许外部传参，否则用保守默认（更省 token、更稳定）
    mt = int(max_tokens if max_tokens is not None else cfg.llm_max_tokens)
    temp = float(temperature if temperature is not None else cfg.llm_temperature)
    tk = int(top_k if top_k is not None else cfg.llm_top_k)

    # read timeout：取最大值，避免被调用方传 60 仍然超时
    min_timeout = int(cfg.llm_min_timeout_json_sec if force_json else cfg.llm_min_timeout_sec)
    read_timeout = max(int(timeout_sec), int(cfg.llm_timeout_sec), min_timeout)

    headers = {
        "Authorization": f"Bearer {cfg.spark_api_password}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": cfg.spark_model,
        "messages": messages,
        "temperature": temp,
        "top_k": tk,
        "max_tokens": mt,
        "stream": False,
    }
    if force_json:
        body["response_format"] = {"type": "json_object"}

    last_err: Optional[Exception] = None

    # 轻量重试：只针对 5xx/网络抖动；默认超时不重试
    for attempt in range(max(1, int(retries))):
        try:
            r = sess.post(cfg.spark_http_url, headers=headers, json=body, timeout=(10, read_timeout))
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Spark HTTP {r.status_code}: {r.text}")
            if r.status_code >= 400:
                raise RuntimeError(f"Spark HTTP {r.status_code}: {r.text}")

            data = r.json()
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(data, ensure_ascii=False)

        except requests.exceptions.ReadTimeout as e:
            last_err = e
            if no_retry_on_timeout or attempt == retries - 1:
                if return_fallback_json_on_timeout and force_json:
                    # 返回一个合法 JSON，让上层能继续跑（不会卡死整批）
                    return json.dumps(
                        {
                            "overall": {"verdict": "无法判断", "uncertain": True},
                            "items": [],
                            "comment": "Spark 超时未返回（read timeout）",
                        },
                        ensure_ascii=False,
                    )
                raise
            # 否则允许重试（不推荐批量这么做）
            continue

        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt == retries - 1:
                raise
            continue

        except Exception as e:
            last_err = e
            if attempt == retries - 1:
                raise
            continue

    # 理论上不会走到这里
    if last_err:
        raise last_err
    raise RuntimeError("Spark 调用失败")



def compress_ocr_text(text: str, *, limit_chars: int = 1800) -> str:
    """
    低成本（无LLM）压缩 OCR 文本：去重、去噪、优先保留题号/答案相关行。
    目标：把输入给 LLM 的内容从 2000+ 字压到 ~800-1800 字，减少 token 与跑偏。
    """
    import re
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    def is_noise(ln: str) -> bool:
        if len(ln) <= 1:
            return True
        if re.fullmatch(r"[\W_]+", ln):
            return True
        return False

    lines = [ln for ln in lines if not is_noise(ln)]

    # 去重（保持顺序）
    seen = set()
    uniq = []
    for ln in lines:
        key = re.sub(r"\s+", " ", ln)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ln)

    qid_pat = re.compile(r"^\s*\d+(?:\.\d+)*[\.、]?\s*")
    opt_pat = re.compile(r"\b[A-D]\b|（[A-D]）|选[A-D]|答案[:：]\s*[A-D]", re.IGNORECASE)
    blank_pat = re.compile(r"_{2,}|（\s*）|填空|答案", re.IGNORECASE)

    keep, maybe = [], []
    for ln in uniq:
        if qid_pat.search(ln) or opt_pat.search(ln) or blank_pat.search(ln):
            keep.append(ln)
        else:
            maybe.append(ln)

    if len(keep) < 6:
        keep.extend(maybe[:20])

    out = "\n".join(keep).strip()
    if len(out) > limit_chars:
        out = out[:limit_chars] + "\n(…OCR已压缩截断…)"
    return out


def estimate_items_count(text: str) -> int:
    """估计题目数量，用于动态设置 max_tokens，避免输出被截断。"""
    import re
    if not text:
        return 6
    pats = re.findall(r"(?m)^\s*(\d+(?:\.\d+)*)[\.、]?\s*", text)
    uniq = []
    for p in pats:
        if p not in uniq:
            uniq.append(p)
    n = len(uniq)
    if n == 0:
        n = max(4, min(12, len(re.findall(r"\d+(?:\.\d+)*", text))))
    return max(4, min(25, n))


def extract_json_obj(text: str):
    """
    更鲁棒的 JSON 提取：支持 ```json``` 包裹、支持前后杂文本、支持括号配对。
    若被截断（缺少右大括号），返回 None。
    """
    import json, re
    if not text:
        return None
    t = text.strip()

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = t[start:i+1]
                    try:
                        obj = json.loads(cand)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def _strip_json_fences(s: str) -> str:
    """Remove common ```json fences if present (best-effort)."""
    s = (s or "").strip()
    # Remove single top-level fence pair
    if s.startswith("```") and s.endswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3:
            s = "\n".join(lines[1:-1]).strip()
    # Remove leading ```json or ``` and trailing ```
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_braced_json(s: str) -> Optional[str]:
    """
    Extract first complete {...} JSON object substring by brace matching,
    respecting double-quoted strings.
    Returns None if not found or appears truncated.
    """
    if not s:
        return None
    t = s.strip()
    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i+1]
    # truncated (missing closing brace)
    return None


def _repair_common_json_issues(s: str) -> str:
    """
    Conservative JSON repair for common LLM slips:
    - values wrapped in single quotes:  "comment": 'xxx'
      (only fixes single-quoted STRING VALUES after a colon; does not touch keys)
    """
    if not s:
        return s

    # Replace : '...'(no newlines) with : "..."
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        # Escape backslash + double quote to keep valid JSON
        inner = inner.replace("\\", "\\\\").replace('"', '\\"')
        return ': "' + inner + '"'

    return re.sub(r":\s*'([^'\r\n]*)'\s*(?=[,}])", repl, s)


def try_parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON extractor for model outputs (robust).

    Handles:
    - raw JSON object
    - JSON inside ```json fences
    - extra chatter before/after JSON
    - common slip: single-quoted string values (e.g., "comment": '...')

    Returns a dict or None.
    """
    if not text:
        return None

    # First pass: existing extractor (fences + brace matching)
    obj = extract_json_obj(text)
    if isinstance(obj, dict):
        return obj

    t = _strip_json_fences(text)

    # If the model returned a quoted JSON string, unquote it (rare)
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        try:
            decoded = json.loads(t)
            if isinstance(decoded, str):
                t = decoded.strip()
        except Exception:
            pass

    cand = _extract_braced_json(t)
    if not cand:
        return None

    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        cand2 = _repair_common_json_issues(cand)
        try:
            obj2 = json.loads(cand2)
            return obj2 if isinstance(obj2, dict) else None
        except Exception:
            return None

def pick_exam_region(text: str, max_chars: int = 2200) -> str:
    if not text:
        return ""
    t = text
    for marker in ["[基础过关训练]", "基础过关训练", "试题", "练习", "一、", "一."]:
        idx = t.find(marker)
        if idx != -1:
            t = t[idx:]
            break
    t = t.strip()
    if len(t) > max_chars:
        t = t[:max_chars] + "\n(…后略…)"
    return t


def llm_cleanup_ocr(cfg: AppConfig, extracted_text: str, *, session: Optional[requests.Session] = None) -> str:
    extracted_text = (extracted_text or "").strip()
    if not extracted_text:
        return ""

    prompt = f"""
你是OCR文本清洗器。给你一段OCR识别的文本，请你做“轻度清洗整理”：
- 不要编造未出现的信息
- 删除明显无关的页码/噪声（比如孤立的编号、重复）
- 合并被错误断行的句子
- 保留数学符号、单位、数字
只输出清洗后的文本本身，不要输出任何解释或多余文字。

OCR文本：
{extracted_text}
""".strip()

    try:
        out = spark_http_chat(cfg, [{"role": "user", "content": prompt}], session=session, timeout_sec=120, force_json=False)
        return out.strip()
    except Exception:
        # Cleanup is optional; fall back to original extracted text.
        return extracted_text


def build_grading_prompt(subject: str, mode: str, question_text: str, answer_key: str, student_text: str, *, enable_cleanup: bool = False) -> str:
    """
    输出严格 JSON：
    {
      "overall": {"verdict": "...", "uncertain": false},
      "items": [{"qid":"1","verdict":"...","uncertain":false,"recognized_answer":"...","comment":"..."}]
    }
    """
    comment_req = "每道题 comment 写 1-2 句话（各题独立）。" if mode == "check_and_comment" else "每道题 comment 写 1 句话以内或写“OK”。"
    cleanup_req = (
        "你可以做轻度清洗：纠正明显OCR错字/断行/空格，但不要凭空补充内容。"
        if enable_cleanup else
        "不要做任何清洗或纠错：严格基于原始OCR文本判题。"
    )

    schema_example = (
        "{\n"
        '  "overall": {"verdict": "正确", "uncertain": false},\n'
        '  "items": [\n'
        '    {"qid": "1", "verdict": "正确", "uncertain": false, "recognized_answer": "…", "comment": "…"}\n'
        "  ]\n"
        "}"
    )

    return f"""
你是一位严格的作业批改助手。你会收到 OCR 识别到的学生作答文本（可能有错字/断行）、题目文本、标准答案/评分要点。
你必须只输出严格 JSON（不要 Markdown、不要代码块、不要解释性文字）。

科目：{subject}
题目/说明（可能包含多题）：{(question_text or "").strip() or "无"}
标准答案（可能包含多题）：{(answer_key or "").strip() or "无"}

学生作答文本（OCR 原文）：
{(student_text or "").strip() or "（未识别到学生答案）"}

规则：
- 【输出长度约束】recognized_answer ≤ 40 字符；comment ≤ 40 字符；不要复述题干/推导，只写最终答案和一句理由。

- {cleanup_req}
- 对每道题给出 verdict：只能是 正确/错误/接近正确/部分正确/无法判断
- {comment_req}
- 如果题目无法明确拆分多题，也必须返回 items 且至少包含一个元素 qid="1"。

输出 JSON schema（必须完全匹配，且只输出这一份 JSON）：
{schema_example}
""".strip()


# -------------------------
# Student name extraction (best-effort)
# -------------------------
_NAME_PATTERNS = [
    # 中文常见：姓名：张三 / 学生：张三 / 学号：2023xxxx
    re.compile(r"(?:姓名|学生|学生姓名|Name)\s*[:：]\s*([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{1,20})"),
    # “张三  班级xxx” 的情况（极弱规则：取前 1-6 个中文/字母）
    re.compile(r"^\s*([A-Za-z\u4e00-\u9fff]{2,8})\s+(?:班级|Class|学号|ID)\b", re.MULTILINE),
    # 学号开头
    re.compile(r"(?:学号|ID)\s*[:：]\s*([A-Za-z0-9\-]{4,30})"),
]

_CLASS_PATTERNS = [
    re.compile(r"(?:班级|Class)\s*[:：]\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,30})"),
]


def infer_student_info(ocr_text: str, *, filename: str = "") -> Tuple[str, str]:
    """
    从 OCR 文本里尽量找 student_name / student_class。
    找不到就退化到 filename（去掉扩展名）或 "unknown"。
    """
    text = (ocr_text or "").strip()

    student_name = ""
    student_class = ""

    if text:
        for p in _NAME_PATTERNS:
            m = p.search(text)
            if m:
                student_name = m.group(1).strip()
                break
        for p in _CLASS_PATTERNS:
            m = p.search(text)
            if m:
                student_class = m.group(1).strip()
                break

    if not student_name:
        base = Path(filename).stem if filename else ""
        base = re.sub(r"[_\-]+", " ", base).strip()
        student_name = base or "unknown"

    return student_name, student_class


# -------------------------
# Grading providers
# -------------------------
def mock_provider() -> Dict[str, Any]:
    return {"verdict": "无法判断", "comment": "mock", "uncertain": True, "recognized_answer": ""}


def cloud_ocr_only_provider(cfg: AppConfig, pre_bytes: bytes, *, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    t0 = time.time()
    ocr = cloud_ocr(pre_bytes, cfg, session=session)
    dt = time.time() - t0
    text = (ocr.get("extracted_text") or "").strip()
    return {
        "verdict": "无法判断",
        "comment": f"OK（OCR {dt:.1f}s）",
        "uncertain": (not bool(text)),
        "recognized_answer": text,
        "ocr_extracted_text": text,
        "ocr_raw_text": ocr.get("raw_text", ""),
        "ocr_response": ocr.get("resp", {}),
        "spark_raw": "",
        "items": [],
        "overall": {"verdict": "无法判断", "uncertain": True},
    }


def spark_http_provider(
    cfg: AppConfig,
    pre_bytes: bytes,
    *,
    subject: str,
    mode: str,
    question_text: str,
    answer_key: str,
    enable_cleanup: bool,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    OCR -> (optional) LLM cleanup -> Spark grading.
    Always returns OCR fields + spark_raw for debugging.
    """
    sess = session or make_http_session()

    # OCR
    t0 = time.time()
    ocr = cloud_ocr(pre_bytes, cfg, session=sess)
    ocr_dt = time.time() - t0

    image_sha1 = hashlib.sha1(pre_bytes).hexdigest()
    extracted = (ocr.get("extracted_text") or "").strip()
    extracted_region = pick_exam_region(extracted, max_chars=2200)

    cleaned = extracted_region
    compressed_for_llm = compress_ocr_text(extracted_region, limit_chars=1800)
    cleanup_dt = 0.0
    if enable_cleanup and extracted_region:
        t_clean = time.time()
        try:
            cleaned = llm_cleanup_ocr(cfg, extracted_region[:1800], session=sess).strip()
        except Exception:
            cleaned = extracted_region
        cleanup_dt = time.time() - t_clean

    prompt = build_grading_prompt(subject, mode, question_text, answer_key, cleaned, enable_cleanup=False)

    base_debug = {
        "image_sha1": image_sha1,
        "ocr_extracted_text": extracted,
        "ocr_clean_text": cleaned,
        "ocr_raw_text": ocr.get("raw_text", ""),
        "ocr_response": ocr.get("resp", {}),
        "spark_raw": "",
        "timing": {"ocr_sec": ocr_dt, "cleanup_sec": cleanup_dt, "spark_sec": 0.0},
    }

    last_err: Optional[Exception] = None
    spark_dt = 0.0

    # --- Spark grading (batch-friendly) ---
    t_spark = time.time()
    try:
        out = spark_http_chat(
            cfg,
            [{"role": "user", "content": prompt}],
            session=sess,
            force_json=True,   # JSON 输出：spark_http_chat 内部会用更长 read timeout（例如 180s）
        )
    except Exception as e:
        spark_dt = time.time() - t_spark
        base_debug["timing"]["spark_sec"] = spark_dt
        base_debug["spark_raw"] = f"EXCEPTION: {e}"
        timing = f"OCR {ocr_dt:.1f}s, cleanup {cleanup_dt:.1f}s, LLM {spark_dt:.1f}s"
        return {
            "verdict": "无法判断",
            "comment": f"Spark 调用失败（{timing}）：{e}",
            "uncertain": True,
            "recognized_answer": cleaned or extracted_region or extracted,
            "items": [],
            "overall": {"verdict": "无法判断", "uncertain": True},
            **base_debug,
        }
    spark_dt = time.time() - t_spark
    base_debug["timing"]["spark_sec"] = spark_dt
    base_debug["spark_raw"] = out

    parsed = try_parse_json(out)

    if not parsed:
        timing = f"OCR {ocr_dt:.1f}s, LLM {spark_dt:.1f}s"
        return {
            "verdict": "无法判断",
            "comment": f"模型输出无法解析为JSON（{timing}）",
            "uncertain": True,
            "recognized_answer": cleaned or extracted_region or extracted,
            "overall": {"verdict": "无法判断", "uncertain": True},
            "items": [],
            **base_debug,
        }

    # normalize schema
    items = parsed.get("items", [])
    overall = parsed.get("overall", {})
    if not isinstance(items, list):
        items = []
    if not isinstance(overall, dict):
        overall = {}

    allowed = {"正确", "错误", "接近正确", "部分正确", "无法判断"}
    o_verdict = overall.get("verdict", "无法判断")
    o_uncertain = bool(overall.get("uncertain", True))
    if o_verdict not in allowed:
        o_verdict, o_uncertain = "无法判断", True

    norm_items: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("qid", "1"))
        v = str(it.get("verdict", "无法判断"))
        u = bool(it.get("uncertain", True))
        if v not in allowed:
            v, u = "无法判断", True
        norm_items.append({
            "qid": qid,
            "verdict": v,
            "uncertain": u,
            "recognized_answer": ensure_str(it.get("recognized_answer", "")),
            "comment": ensure_str(it.get("comment", "")),
        })

    # compact per-item fields to avoid long outputs
    norm_items = [compact_item_fields(it) for it in norm_items]

    if not norm_items:
        norm_items = [{
            "qid": "1",
            "verdict": o_verdict,
            "uncertain": o_uncertain,
            "recognized_answer": cleaned or extracted_region or extracted,
            "comment": "",
        }]

    # Backward-compatible top-level fields
    top_verdict = o_verdict
    top_uncertain = o_uncertain
    top_comment = "；".join(
        [f"题{it['qid']}：{it.get('comment','')}".strip("；") for it in norm_items if it.get("comment")]
    )[:400]

    return {
        "verdict": top_verdict,
        "comment": top_comment or "",
        "uncertain": top_uncertain,
        "recognized_answer": cleaned or extracted_region or extracted,
        "overall": {"verdict": o_verdict, "uncertain": o_uncertain},
        "items": norm_items,
        **base_debug,
    }



# -------------------------
# Wrongbook (JSONL) + export helpers
# -------------------------
def ensure_wrongbook_file(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")


def append_wrongbook_records(*, result: Dict[str, Any], meta: Dict[str, Any], out_path: str) -> int:
    """
    逐题写入 JSONL（一行一题）。
    - Treat verdict != "正确" OR uncertain=True as wrong
    """
    items = result.get("items", [])
    if not isinstance(items, list) or not items:
        return 0

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("a", encoding="utf-8") as f:
        for it in items:
            if not isinstance(it, dict):
                continue
            verdict = str(it.get("verdict", "无法判断"))
            uncertain = bool(it.get("uncertain", True))
            if verdict == "正确" and not uncertain:
                continue

            rec = {
                "ts": now_iso(),
                **meta,
                "qid": str(it.get("qid", "1")),
                "verdict": verdict,
                "uncertain": uncertain,
                "recognized_answer": ensure_str(it.get("recognized_answer", "")),
                "comment": ensure_str(it.get("comment", "")),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def _safe_read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def _norm_date(ts: Any) -> str:
    s = ensure_str(ts)
    if not s:
        return "unknown"
    # keep YYYY-MM-DD
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else s[:10]


def export_wrongbook_markdown(jsonl_path: str, out_md_path: str) -> int:
    rows = _safe_read_jsonl(jsonl_path)
    if not rows:
        Path(out_md_path).write_text("# 错题本导出\n\n（空）\n", encoding="utf-8")
        return 0

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in rows:
        student = (r.get("student_name") or "unknown").strip() or "unknown"
        date = _norm_date(r.get("ts"))
        grouped.setdefault(student, {}).setdefault(date, []).append(r)

    def _qid_sort_key(x: Dict[str, Any]):
        q = str(x.get("qid", ""))
        try:
            return (0, int(q))
        except Exception:
            return (1, q)

    outp = Path(out_md_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        f.write("# 错题本导出\n\n")
        f.write(f"- 来源：{jsonl_path}\n")
        f.write(f"- 导出时间：{now_iso()}\n")
        f.write(f"- 记录条数：{len(rows)}\n\n")
        for student in sorted(grouped.keys()):
            f.write(f"## {student}\n\n")
            dates = grouped[student]
            for d in sorted(dates.keys()):
                f.write(f"### {d}\n\n")
                items = sorted(dates[d], key=_qid_sort_key)
                for r in items:
                    qid = r.get("qid", "")
                    verdict = r.get("verdict", "")
                    uncertain = r.get("uncertain", False)
                    rec = (r.get("recognized_answer") or "").strip()
                    comment = (r.get("comment") or "").strip()
                    subject = (r.get("subject") or "").strip()
                    f.write(f"#### 题 {qid}（{subject}）\n\n")
                    f.write(f"- 判定：{verdict}{'（不确定）' if uncertain else ''}\n")
                    if rec:
                        f.write(f"- 学生答案：{rec}\n")
                    if comment:
                        f.write(f"- 评语：{comment}\n")
                    f.write("\n")
    return len(rows)


# -------------------------
# Batch grading (concurrency capped)
# -------------------------
def grade_one_image(
    *,
    cfg: AppConfig,
    raw_bytes: bytes,
    filename: str,
    subject: str,
    mode: str,
    question_text: str,
    answer_key: str,
    enable_cleanup: bool,
    crop_ratio: float = 0.82,
    max_side: int = 1600,
    jpeg_quality: int = 75,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    单张图片：预处理 -> provider -> 自动从 OCR 推断学生名
    返回 result（包含 student_name/student_class 字段）
    """
    pre_bytes = preprocess_for_ocr(
        raw_bytes,
        crop_ratio=crop_ratio,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        enhance_contrast=1.10,
    )

    if cfg.provider == "mock":
        res = mock_provider()
        ocr_text = ""
    elif cfg.provider == "cloud_ocr_only":
        res = cloud_ocr_only_provider(cfg, pre_bytes, session=session)
        ocr_text = ensure_str(res.get("ocr_extracted_text", res.get("recognized_answer", "")))
    else:
        res = spark_http_provider(
            cfg,
            pre_bytes,
            subject=subject,
            mode=mode,
            question_text=question_text,
            answer_key=answer_key,
            enable_cleanup=enable_cleanup,
            session=session,
        )
        ocr_text = ensure_str(res.get("ocr_extracted_text", res.get("recognized_answer", "")))

    student_name, student_class = infer_student_info(ocr_text, filename=filename)
    res["student_name"] = student_name
    res["student_class"] = student_class
    res["filename"] = filename
    res["preprocess"] = {"crop_ratio": crop_ratio, "max_side": max_side, "jpeg_quality": jpeg_quality}
    return res


def write_student_run(
    *,
    run_dir: str,
    student_name: str,
    filename: str,
    raw_bytes: bytes,
    result: Dict[str, Any],
    save_wrongbook: bool = True,
) -> Dict[str, str]:
    """
    输出结构：
    runs/<ts>/<student>/input/<original_filename>
    runs/<ts>/<student>/result.json
    runs/<ts>/<student>/summary.md
    runs/<ts>/<student>/wrongbook.jsonl   (可选)
    """
    run_root = Path(run_dir)
    student_dir = run_root / safe_filename(student_name, fallback="unknown")
    input_dir = student_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # save original
    in_path = input_dir / safe_filename(filename, fallback="input.jpg")
    try:
        in_path.write_bytes(raw_bytes)
    except Exception:
        # some uploaded files may be bytes already ok; ignore if fails
        pass

    # full json
    json_path = student_dir / "result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # lightweight md
    md_path = student_dir / "summary.md"
    overall = result.get("overall", {})
    items = result.get("items", [])
    lines = []
    lines.append(f"# 批改结果：{student_name}")
    lines.append("")
    lines.append(f"- 文件：{filename}")
    lines.append(f"- 时间：{now_iso()}")
    lines.append(f"- 总评：{overall.get('verdict','无法判断')}（不确定={bool(overall.get('uncertain', True))}）")
    lines.append("")
    lines.append("## 分题")
    if isinstance(items, list) and items:
        for it in items:
            qid = it.get("qid", "")
            v = it.get("verdict", "")
            u = it.get("uncertain", False)
            cm = (it.get("comment") or "").strip()
            lines.append(f"- 题 {qid}：{v}{'（不确定）' if u else ''}  {cm}")
    else:
        lines.append("- （无分题输出）")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = {
        "student_dir": str(student_dir),
        "input_path": str(in_path),
        "result_json": str(json_path),
        "summary_md": str(md_path),
    }

    if save_wrongbook:
        wb_path = student_dir / "wrongbook.jsonl"
        ensure_wrongbook_file(str(wb_path))
        meta = {
            "provider": result.get("provider", ""),
            "subject": result.get("subject", ""),
            "mode": result.get("mode", ""),
            "student_name": student_name,
            "student_class": result.get("student_class", ""),
            "image_sha1": result.get("image_sha1", ""),
            "filename": filename,
        }
        append_wrongbook_records(result=result, meta=meta, out_path=str(wb_path))
        out["wrongbook_jsonl"] = str(wb_path)

    return out


def batch_grade_images(
    *,
    cfg: AppConfig,
    images: List[Tuple[str, bytes]],  # [(filename, raw_bytes), ...]
    subject: str,
    mode: str,
    question_text: str,
    answer_key: str,
    enable_cleanup: bool,
    run_root: str = "./runs",
    max_concurrency: int = 2,
    crop_ratio: float = 0.82,
    max_side: int = 1600,
    jpeg_quality: int = 75,
    save_wrongbook: bool = True,
) -> Dict[str, Any]:
    """
    同步接口（内部用线程池限制并发 max_concurrency）。
    返回：
    {
      "run_dir": ".../runs/2025-.../",
      "students": [{"student_name":..., "student_dir":..., ...}, ...],
      "errors": [{"filename":..., "error":...}, ...],
    }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = str(Path(run_root) / ts)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    sess = make_http_session(pool=max(10, max_concurrency * 2))

    students_out: List[Dict[str, Any]] = []
    errors_out: List[Dict[str, Any]] = []

    def _worker(fn: str, b: bytes):
        res = grade_one_image(
            cfg=cfg,
            raw_bytes=b,
            filename=fn,
            subject=subject,
            mode=mode,
            question_text=question_text,
            answer_key=answer_key,
            enable_cleanup=enable_cleanup,
            crop_ratio=crop_ratio,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            session=sess,
        )
        # attach some top-level meta for downstream
        res["provider"] = cfg.provider
        res["subject"] = subject
        res["mode"] = mode

        paths = write_student_run(
            run_dir=run_dir,
            student_name=res.get("student_name", "unknown"),
            filename=fn,
            raw_bytes=b,
            result=res,
            save_wrongbook=save_wrongbook,
        )
        return {"result": res, "paths": paths}

    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futs = {ex.submit(_worker, fn, b): fn for (fn, b) in images}
        for fut in as_completed(futs):
            fn = futs[fut]
            try:
                out = fut.result()
                students_out.append({
                    "filename": fn,
                    "student_name": out["result"].get("student_name", "unknown"),
                    "student_class": out["result"].get("student_class", ""),
                    **out["paths"],
                    "overall": out["result"].get("overall", {}),
                })
            except Exception as e:
                errors_out.append({"filename": fn, "error": str(e)})

    # quick index file
    index_path = Path(run_dir) / "index.json"
    index_path.write_text(json.dumps({"students": students_out, "errors": errors_out}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"run_dir": run_dir, "index_json": str(index_path), "students": students_out, "errors": errors_out}


# ============================================================
# HOTFIX OVERRIDE (keep this block at the VERY END of file)
# Reason: the file previously contained duplicated older defs
# (AppConfig/make_http_session/spark_http_chat) later in the file,
# which override the patched versions at runtime.
# ============================================================

from dataclasses import dataclass as _dataclass  # noqa: F401

@_dataclass
class AppConfig:  # type: ignore[no-redef]
    provider: str = "spark_http"  # mock | cloud_ocr_only | spark_http

    # Spark OpenAPI (HTTP)
    spark_http_url: str = ""
    spark_api_password: str = ""
    spark_model: str = "x1"

    # LLM defaults (Spark HTTP)
    llm_timeout_sec: int = 180
    llm_min_timeout_sec: int = 120
    llm_min_timeout_json_sec: int = 180
    llm_max_tokens: int = 1800
    llm_temperature: float = 0.0
    llm_top_k: int = 1

    # Cloud OCR (OCRforLLM WebAPI)
    ocr_url: str = ""
    ocr_app_id: str = ""
    ocr_api_key: str = ""
    ocr_api_secret: str = ""


def make_http_session(pool: int = 10) -> requests.Session:  # type: ignore[no-redef]
    """Create a requests Session with NO urllib3 retries.

    We control retries in spark_http_chat() so error messages stay clean
    and we never see misleading 'Max retries exceeded' wrappers.
    """
    s = requests.Session()
    retry = Retry(
        total=0,
        connect=0,
        read=0,
        backoff_factor=0,
        status_forcelist=[],
        allowed_methods=False,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def spark_http_chat(  # type: ignore[no-redef]
    cfg: AppConfig,
    messages: List[Dict[str, str]],
    *,
    session: Optional[requests.Session] = None,
    timeout_sec: int = 180,
    force_json: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    retries: int = 1,
    no_retry_on_timeout: bool = True,
    return_fallback_json_on_timeout: bool = True,
) -> str:
    """Spark OpenAPI HTTP call with hard timeout floors (120/180s)."""
    if not cfg.spark_http_url:
        raise RuntimeError("未配置 SPARK_HTTP_URL")
    if not cfg.spark_api_password:
        raise RuntimeError("未配置 SPARK_API_PASSWORD（APIPassword）")

    sess = session or make_http_session()

    mt = int(max_tokens if max_tokens is not None else getattr(cfg, "llm_max_tokens", 450))
    temp = float(temperature if temperature is not None else getattr(cfg, "llm_temperature", 0.0))
    tk = int(top_k if top_k is not None else getattr(cfg, "llm_top_k", 1))

    min_timeout = int(getattr(cfg, "llm_min_timeout_json_sec", 180) if force_json else getattr(cfg, "llm_min_timeout_sec", 120))
    read_timeout = max(int(timeout_sec), int(getattr(cfg, "llm_timeout_sec", timeout_sec)), min_timeout)

    headers = {
        "Authorization": f"Bearer {cfg.spark_api_password}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": cfg.spark_model,
        "messages": messages,
        "temperature": temp,
        "top_k": tk,
        "max_tokens": mt,
        "stream": False,
    }
    if force_json:
        body["response_format"] = {"type": "json_object"}

    last_err: Optional[Exception] = None

    for attempt in range(max(1, int(retries))):
        try:
            r = sess.post(cfg.spark_http_url, headers=headers, json=body, timeout=(10, read_timeout))

            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"Spark HTTP {r.status_code}: {r.text}")
                if attempt < retries - 1:
                    time.sleep(min(0.8 * (2 ** attempt), 6.0))
                    continue
                raise last_err

            if r.status_code >= 400:
                raise RuntimeError(f"Spark HTTP {r.status_code}: {r.text}")

            data = r.json()
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(data, ensure_ascii=False)

        except requests.exceptions.ReadTimeout as e:
            last_err = e
            if no_retry_on_timeout or attempt == retries - 1:
                if return_fallback_json_on_timeout and force_json:
                    return json.dumps(
                        {
                            "overall": {"verdict": "无法判断", "uncertain": True},
                            "items": [],
                            "comment": f"Spark 超时未返回（read timeout={read_timeout}s）",
                        },
                        ensure_ascii=False,
                    )
                raise
            time.sleep(min(0.8 * (2 ** attempt), 6.0))
            continue

        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt == retries - 1:
                raise
            time.sleep(min(0.8 * (2 ** attempt), 6.0))
            continue

    if last_err:
        raise last_err
    raise RuntimeError("Spark 调用失败（未知原因）")
