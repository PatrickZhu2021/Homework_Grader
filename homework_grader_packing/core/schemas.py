from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict

@dataclass
class GradeResult:
    file_name: str
    job_id: str

    # 关键产物
    ocr_text: str = ""
    cleaned_text: str = ""
    recognized_answer: str = ""

    verdict: str = "无法判断"
    comment: str = ""
    uncertain: bool = True

    # 统计与错误
    latency_ms: Dict[str, int] = None
    error: Optional[str] = None

    # 可选：原始返回，便于排查
    llm_raw: Optional[Dict[str, Any]] = None
    ocr_raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["latency_ms"] is None:
            d["latency_ms"] = {}
        return d
