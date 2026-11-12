"""沿用 auto_eval 的时间处理；实验台直接接受服务器本地视频路径。"""
import os
import math
import json
from pathlib import Path
from numbers import Real
from .media import DEFAULT_TASK_START_TIME, KEYFRAME_ALGORITHM_VERSION
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
_TASK_TIME_FIELDS = ("task_start_time", "task_end_time")

def operation_video_roots(base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """返回相对视频路径的查找目录，不限制绝对路径。"""
    roots = [base_dir.resolve()]
    for raw in os.getenv("OPERATION_VIDEO_ROOTS", "").split(os.pathsep):
        if raw.strip():
            root = Path(raw.strip()).expanduser()
            if not root.is_absolute():
                root = base_dir / root
            roots.append(root.resolve())
    return list(dict.fromkeys(roots))

def resolve_operation_video_path(
    raw_path: str,
    *,
    base_dir: Path = PROJECT_ROOT,
) -> Path:
    """解析服务器本地视频；绝对路径直接读取，相对路径按根目录查找。"""
    candidate = Path(raw_path).expanduser()
    roots = operation_video_roots(base_dir)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        attempted: list[Path] = []
        matches: list[Path] = []
        for root in roots:
            resolved = (root / candidate).resolve()
            attempted.append(resolved)
            if resolved.is_file() and resolved not in matches:
                matches.append(resolved)
        if not matches:
            locations = "\n".join(f"- {path}" for path in attempted)
            suffix = f"；已按以下根目录查找：\n{locations}" if locations else ""
            raise ValueError(f"视频文件不存在：{raw_path}{suffix}")
        if len(matches) > 1:
            locations = "\n".join(f"- {path}" for path in matches)
            raise ValueError(
                f"视频相对路径存在多个匹配：{raw_path}\n{locations}\n"
                "请提供更完整的相对路径或绝对路径"
            )
        candidate = matches[0]
    if not candidate.is_file():
        raise ValueError(f"视频文件不存在：{raw_path}")
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"不支持的视频格式 {candidate.suffix or '(无扩展名)'}；支持：{supported}")
    return candidate

def _operation_timing(
    item: dict,
    duration: float,
    *,
    default_start_time: float = DEFAULT_TASK_START_TIME,
) -> tuple[dict[str, float], str, list[str], set[str]]:
    """校验任务起止时间，并生成抽帧参数、缓存键及回退警告。"""
    supplied: dict[str, float] = {}
    warnings: list[str] = []
    ignored_fields: set[str] = set()
    for field in _TASK_TIME_FIELDS:
        value = item.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} 必须是有限数字（单位：秒）")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{field} 必须是有限数字（单位：秒）")
        if normalized < 0:
            raise ValueError(f"{field} 不能小于 0")
        if normalized > duration:
            fallback = "默认开始时间" if field == "task_start_time" else "默认结束时间"
            warnings.append(
                f"{field}={normalized:g} 秒超出视频时长 {duration:g} 秒，"
                f"已忽略该值并使用{fallback}"
            )
            ignored_fields.add(field)
            continue
        supplied[field] = normalized

    effective_start = supplied.get("task_start_time", default_start_time)
    task_end = supplied.get("task_end_time")
    if task_end is not None and task_end <= effective_start:
        raise ValueError("task_end_time 必须大于 task_start_time")

    cache_payload = {
        "algorithm_version": KEYFRAME_ALGORITHM_VERSION,
        "task_start_time": effective_start,
        "task_end_time": task_end,
    }
    cache_key = json.dumps(
        cache_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return supplied, cache_key, warnings, ignored_fields
