"""路径与时间处理摘自 auto_eval，保持原规则。"""
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
    """返回批量清单允许读取、解析相对路径的视频根目录。"""
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
    """解析本地视频路径，并阻止读取未授权目录。"""
    candidate = Path(raw_path).expanduser()
    roots = operation_video_roots(base_dir)
    if candidate.is_absolute():
        candidate = candidate.resolve()
        if not any(candidate.is_relative_to(root) for root in roots):
            raise ValueError(
                "视频路径不在允许目录中；外部目录需通过 "
                "OPERATION_VIDEO_ROOTS 配置"
            )
    else:
        attempted: list[Path] = []
        matches: list[Path] = []
        for root in roots:
            resolved = (root / candidate).resolve()
            if not resolved.is_relative_to(root):
                continue
            attempted.append(resolved)
            if resolved.is_file():
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

    effective_start = supplied.get("task_start_time", DEFAULT_TASK_START_TIME)
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
