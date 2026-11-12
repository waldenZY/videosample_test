"""输入解析：粘贴文本 / 上传 jsonl → 标准化题目列表。

每题返回 dict：
  single : {query, context?, answer, competitor?, reference?}
  compare: {query, context?, answer_a, answer_b, reference?}
  online : {query, context?, reference?}
  process: {query, context?, answer, trace, reference?}
  operation: {id?, query, query_images?, context?, video_path, answer?,
              task_start_time?, task_end_time?}
  rich_content: {id?, query, context?, video_path, answer_text?,
                 content_start_time?, content_end_time?, category?}

文本格式在 query 后支持可选的显式背景段：
  query ||| @context: 背景信息 ||| 其余原有字段
没有该段时完全按旧格式解析；空背景段会被忽略。
"""
from __future__ import annotations

import json
import math
from numbers import Real
from typing import Literal

from ..media import DEFAULT_TASK_START_TIME

Mode = Literal[
    "single",
    "compare",
    "online",
    "process",
    "operation",
    "rich_content",
    "rich_content_quality",
]
_CONTEXT_PREFIX = "@context:"
MAX_QUERY_IMAGES = 4


def operation_query_images(obj: dict) -> list[str]:
    """读取可选图片路径/链接；标准数组优先，兼容单图字段。"""
    value = obj.get("query_images")
    if value is None or value == []:
        for field in ("query_image_path", "attachment_path"):
            single = obj.get(field)
            if single is None or single == "":
                continue
            if not isinstance(single, str):
                raise ValueError(f"{field} 必须是图片路径或 HTTP/HTTPS 链接字符串")
            if single.strip():
                value = [single]
                break
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("query_images 必须是图片路径字符串数组")
    if len(value) > MAX_QUERY_IMAGES:
        raise ValueError(f"query_images 最多支持 {MAX_QUERY_IMAGES} 张")
    images: list[str] = []
    for index, path in enumerate(value, start=1):
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"query_images 第 {index} 项必须是非空图片路径")
        normalized = path.strip()
        if normalized not in images:
            images.append(normalized)
    return images


def _json_safe_source(value):
    """保留原始 JSONL 字段，同时把 NaN/Infinity 归一为空值。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe_source(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_source(item) for item in value]
    return value


def _source_data_with_flat_metadata(obj: dict) -> dict:
    """保留原始 metadata，并将其第一层字段无前缀提升；外层字段优先。"""
    source = _json_safe_source(obj)
    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            source.setdefault(str(key), value)
    return source


def _extract_text_context(parts: list[str]) -> tuple[list[str], str | None]:
    """提取紧跟 query 的可选 ``@context: ...`` 段，并保持旧位置格式兼容。"""
    if len(parts) < 2 or not parts[1].lower().startswith(_CONTEXT_PREFIX):
        return parts, None
    context = parts[1][len(_CONTEXT_PREFIX):].strip()
    return [parts[0], *parts[2:]], context or None


def _operation_times(obj: dict) -> dict[str, float]:
    """读取任务类（录屏）JSONL 的可选任务起止时间（单位：秒）。"""
    times: dict[str, float] = {}
    for field in ("task_start_time", "task_end_time"):
        value = obj.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} 必须是有限数字（单位：秒）")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{field} 必须是有限数字（单位：秒）")
        if normalized < 0:
            raise ValueError(f"{field} 不能小于 0")
        times[field] = normalized

    start = times.get("task_start_time", DEFAULT_TASK_START_TIME)
    end = times.get("task_end_time")
    if end is not None and end <= start:
        raise ValueError("task_end_time 必须大于 task_start_time")
    return times


def _rich_content_times(obj: dict) -> dict[str, float]:
    """读取富内容视频中回答内容的可选起止时间（单位：秒）。"""
    times: dict[str, float] = {}
    for field in ("content_start_time", "content_end_time"):
        value = obj.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} 必须是有限数字（单位：秒）")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{field} 必须是有限数字（单位：秒）")
        if normalized < 0:
            raise ValueError(f"{field} 不能小于 0")
        times[field] = normalized
    start = times.get("content_start_time", 0.0)
    end = times.get("content_end_time")
    if end is not None and end <= start:
        raise ValueError("content_end_time 必须大于 content_start_time")
    return times


def parse_text(text: str, mode: Mode) -> tuple[list[dict], list[str]]:
    """解析 ||| 分隔的粘贴文本。返回 (items, errors)。"""
    if mode in ("operation", "rich_content", "rich_content_quality"):
        label = "任务类" if mode == "operation" else "垂域视觉评测"
        return [], [f"{label}评测请逐题上传视频或导入 JSONL，不支持文本粘贴解析"]
    items: list[dict] = []
    errors: list[str] = []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|||")]
        parts, context = _extract_text_context(parts)
        try:
            if mode == "single":
                if len(parts) < 2:
                    raise ValueError("单评模式每行至少需 query ||| answer")
                item = {"query": parts[0], "answer": parts[1]}
                if len(parts) >= 3 and parts[2]:
                    item["competitor"] = parts[2]  # 第3段：竞品结果（产品专家用）
                if len(parts) >= 4 and parts[3]:
                    item["reference"] = parts[3]  # 第4段：参考答案
            elif mode == "compare":
                if len(parts) < 3:
                    raise ValueError("对比模式每行至少需 query ||| answerA ||| answerB")
                item = {"query": parts[0], "answer_a": parts[1], "answer_b": parts[2]}
                if len(parts) >= 4 and parts[3]:
                    item["reference"] = parts[3]
            elif mode == "online":
                if len(parts) < 1 or not parts[0]:
                    raise ValueError("在线模式每行至少需 query")
                item = {"query": parts[0]}
                if len(parts) >= 2 and parts[1]:
                    item["reference"] = parts[1]
            else:  # process
                if len(parts) < 3:
                    raise ValueError("过程模式每行至少需 query ||| answer ||| trace")
                item = {"query": parts[0], "answer": parts[1], "trace": parts[2]}
                if len(parts) >= 4 and parts[3]:
                    item["reference"] = parts[3]
            if context:
                item["context"] = context
            items.append(item)
        except ValueError as e:
            errors.append(f"第 {ln} 行：{e}（原文：{raw[:40]}）")
    return items, errors


def parse_jsonl(content: str, mode: Mode) -> tuple[list[dict], list[str]]:
    """解析 jsonl；任务类（录屏）的任务起止时间为可选秒数，空值使用默认策略。"""
    items: list[dict] = []
    errors: list[str] = []
    video_item_ids: set[str] = set()
    for ln, raw in enumerate(content.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"第 {ln} 行 JSON 错误：{e}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"第 {ln} 行必须是 JSON 对象")
            continue
        q = obj.get("question") or obj.get("query")
        if not isinstance(q, str) or not q.strip():
            errors.append(f"第 {ln} 行缺少 question")
            continue
        item: dict = {"query": q.strip()}
        context = obj.get("context")
        if context is not None and not isinstance(context, str):
            errors.append(f"第 {ln} 行 context 必须是字符串")
            continue
        if context and context.strip():
            item["context"] = context.strip()
        if mode == "single":
            a = obj.get("answer")
            if a is None:
                errors.append(f"第 {ln} 行 single 模式缺少 answer")
                continue
            item["answer"] = a
            if obj.get("competitor"):
                item["competitor"] = obj["competitor"]
        elif mode == "compare":
            aa = obj.get("answer_a") or obj.get("answerA")
            ab = obj.get("answer_b") or obj.get("answerB")
            if aa is None or ab is None:
                errors.append(f"第 {ln} 行 compare 模式缺少 answer_a/answer_b")
                continue
            item["answer_a"], item["answer_b"] = aa, ab
        elif mode == "process":
            a = obj.get("answer")
            tr = obj.get("trace")
            if a is None or tr is None:
                errors.append(f"第 {ln} 行 process 模式缺少 answer/trace")
                continue
            item["answer"], item["trace"] = a, tr
        elif mode in ("operation", "rich_content", "rich_content_quality"):
            video_path = obj.get("video_path")
            if not isinstance(video_path, str) or not video_path.strip():
                errors.append(f"第 {ln} 行 {mode} 模式缺少 video_path")
                continue
            try:
                video_times = (
                    _operation_times(obj)
                    if mode == "operation"
                    else _rich_content_times(obj)
                )
            except ValueError as exc:
                errors.append(f"第 {ln} 行 {exc}")
                continue
            item_id = obj.get("id")
            if item_id is not None:
                if not isinstance(item_id, str) or not item_id.strip():
                    errors.append(f"第 {ln} 行 id 必须是非空字符串")
                    continue
                item_id = item_id.strip()
                if item_id in video_item_ids:
                    errors.append(f"第 {ln} 行 id 重复：{item_id}")
                    continue
                video_item_ids.add(item_id)
                item["id"] = item_id
            item["video_path"] = video_path.strip()
            item["source_line"] = ln
            item.update(video_times)
            if mode == "operation":
                try:
                    query_images = operation_query_images(obj)
                except ValueError as exc:
                    errors.append(f"第 {ln} 行 {exc}")
                    continue
                if query_images:
                    item["query_images"] = query_images
                statement = obj.get("agent_statement", obj.get("answer"))
                if statement is not None and not isinstance(statement, str):
                    errors.append(f"第 {ln} 行 agent_statement 必须是字符串")
                    continue
                item["category"] = obj.get("category") or "operation"
                if statement and statement.strip():
                    item["answer"] = statement.strip()
            else:  # rich_content / rich_content_quality
                answer_text = obj.get("answer_text")
                if answer_text is not None and not isinstance(answer_text, str):
                    errors.append(f"第 {ln} 行 answer_text 必须是字符串")
                    continue
                expected_visual = obj.get("expected_visual")
                if expected_visual is not None and not isinstance(expected_visual, dict):
                    errors.append(f"第 {ln} 行 expected_visual 必须是 JSON 对象")
                    continue
                item["category"] = obj.get("category") or "default"
                if answer_text and answer_text.strip():
                    item["answer_text"] = answer_text.strip()
                if expected_visual is not None:
                    # 仅用于 mini/元评测核对，不会进入视觉裁判 prompt。
                    item["expected_visual"] = expected_visual
        # online 不需要 answer
        if obj.get("reference"):
            item["reference"] = obj["reference"]
        if obj.get("category"):
            item["category"] = obj["category"]
        # 原始字段仅用于历史追溯和导出，不会进入 EvalItem 或裁判 prompt。
        item["source_data"] = _source_data_with_flat_metadata(obj)
        items.append(item)
    return items, errors
