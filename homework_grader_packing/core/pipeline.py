import time
from .schemas import GradeResult

def grade_one_image(
    image_bytes: bytes,
    file_name: str,
    job_id: str,
    cfg: dict,
    ocr_provider,
    llm_provider,
    enable_cleanup: bool = True,
) -> GradeResult:
    t0 = time.time()
    res = GradeResult(file_name=file_name, job_id=job_id, latency_ms={})

    try:
        # 1) OCR
        t = time.time()
        ocr_out = ocr_provider(image_bytes, cfg)  # 你已有的 OCR 函数/类
        res.latency_ms["ocr"] = int((time.time() - t) * 1000)
        res.ocr_raw = ocr_out if isinstance(ocr_out, dict) else None
        res.ocr_text = (ocr_out.get("text") if isinstance(ocr_out, dict) else str(ocr_out)) or ""

        # 2) 清洗（可选）
        t = time.time()
        if enable_cleanup:
            res.cleaned_text = cleanup_text(res.ocr_text)  # 你已有清洗逻辑
        else:
            res.cleaned_text = res.ocr_text
        res.latency_ms["cleanup"] = int((time.time() - t) * 1000)

        # 3) LLM 批改
        t = time.time()
        llm_out = llm_provider(res.cleaned_text, cfg)  # 你已有 spark_http_provider 等
        res.latency_ms["llm"] = int((time.time() - t) * 1000)
        res.llm_raw = llm_out if isinstance(llm_out, dict) else None

        # 4) 结构化字段（按你现在的输出协议对齐）
        if isinstance(llm_out, dict):
            res.verdict = llm_out.get("verdict", res.verdict)
            res.comment = llm_out.get("comment", "")
            res.uncertain = bool(llm_out.get("uncertain", True))
            res.recognized_answer = llm_out.get("recognized_answer", "") or ""
        else:
            res.comment = str(llm_out)

    except Exception as e:
        res.error = str(e)
        res.verdict = "无法判断"
        res.uncertain = True

    res.latency_ms["total"] = int((time.time() - t0) * 1000)
    return res


def cleanup_text(text: str) -> str:
    # 先给个占位：把你现在 app.py 的清洗逻辑挪过来
    return text.strip()
