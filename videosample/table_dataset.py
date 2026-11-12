"""将 XLSX/CSV 表格转换为保序的任务类 JSONL 数据集。

核心字段映射：

* id: 优先复用每行非空的 ``id``；为空或没有该列时，使用
  ``<input_prefix>_<index>``，输入表没有 ``index`` 列时使用
  ``<input_prefix>_<序号>``；均不可用时以源表行号兜底并警告。
* 序号: 保留源表序号，例如 ``simple_001``。
* query/context/answer/task_start_time/task_end_time: 沿用原任务类转换规则。
* video_path: 输入表有 ``video_path`` 列时直接复用；否则按
  ``<video_prefix>/[文件路径]/<原文件名>`` 构造，其中“文件路径”可选。
* 原表列全部保留：标准关键字段位于前面，其余输入列按原列顺序平铺追加到
  每条 JSON 对象末尾；被系统生成值占用的同名原始列使用 ``_1``、``_2``
  后缀保存；直接复用的 ``id`` 不重复生成副本。

默认 context 地点为“浙江省杭州市滨江区滨康路101号”，可通过
``--current-location`` 修改。

示例：

    python scripts/excel_to_jsonl_dataset.py \
      "data/0805/V1/V1_录屏0805_复杂任务.csv" \
      --input-prefix "V1_录屏0805" \
      --video-prefix "data/0805/V1"

CSV 输入使用相同命令；默认输出为输入文件同目录、同名的 ``.jsonl``。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_COLUMN = "序号"
INDEX_COLUMN = "index"
QUERY_COLUMN = "query"
CONTEXT_COLUMN = "context"
ANSWER_COLUMNS = ("agent_statement", "回复内容", "answer")
START_TIME_COLUMNS = ("task_start_time", "开始时间")
END_TIME_COLUMNS = ("task_end_time", "结束时间")
CONTEXT_SOURCE_COLUMNS = (
    ("开始时间节点", "交互发生时间"),
    ("位置信息", "交互发生位置"),
    ("定位信息", "交互发生位置"),
    ("当前位置", "交互发生位置"),
)
INTERACTION_LOCATION_LABEL = "交互发生位置"
VIDEO_PATH_COLUMN = "video_path"
QUERY_IMAGES_COLUMN = "query_images"
QUERY_IMAGE_PATH_COLUMN = "query_image_path"
ATTACHMENT_PATH_COLUMN = "attachment_path"
VIDEO_DIRECTORY_COLUMN = "文件路径"
VIDEO_FILENAME_COLUMN = "原文件名"
DEFAULT_VIDEO_PREFIX = "data/"
DEFAULT_CURRENT_LOCATION = "浙江省杭州市滨江区滨康路101号"
MAX_QUERY_IMAGES = 4

VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
}

EMPTY_TEXT = {"", "nan", "none", "null"}
EMPTY_ANSWER = EMPTY_TEXT | {"n/a", "error"}
EMPTY_PATH = EMPTY_TEXT | {"n/a"}


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    rows: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    missing_video_ids: list[str]
    ignored_empty_rows: int
    direct_standard_columns: tuple[str, ...]
    selected_sheet: str | int | None


def _is_empty(value: Any, *, answer: bool = False) -> bool:
    if value is None:
        return True
    if not isinstance(value, (str, bytes)):
        try:
            if bool(pd.isna(value)):
                return True
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        empty_values = EMPTY_ANSWER if answer else EMPTY_TEXT
        return value.strip().lower() in empty_values
    return False


def _json_value(value: Any) -> Any:
    """把 pandas/Excel 标量转换为严格 JSON 可序列化值。"""
    if value is None:
        return ""
    if not isinstance(value, (str, bytes)):
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _table_query_images(value: Any) -> list[str]:
    """表格单元格支持单路径；query_images 也可填写 JSON 字符串数组。"""
    if _is_empty(value):
        return []
    parsed = value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("query_images 不是合法 JSON 数组") from exc
    values = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    if len(values) > MAX_QUERY_IMAGES:
        raise ValueError(f"query_images 最多支持 {MAX_QUERY_IMAGES} 张")
    images: list[str] = []
    for image in values:
        if _is_empty(image):
            continue
        if not isinstance(image, str):
            raise ValueError("query_images 中的图片路径必须是字符串")
        normalized = image.strip().replace("\\", "/")
        if normalized not in images:
            images.append(normalized)
    return images


def _normalize_identifier(value: Any) -> str:
    if _is_empty(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    return str(value).strip()


def _normalize_prefix(value: str) -> str:
    prefix = re.sub(r"\s+", "_", value.strip())
    prefix = prefix.strip("_")
    if not prefix:
        raise ValueError("input_prefix 不能为空")
    if "/" in prefix or "\\" in prefix:
        raise ValueError("input_prefix 不能包含路径分隔符")
    return prefix


def _normalize_context_value(value: Any) -> str:
    if _is_empty(value):
        return ""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (int, float)) and 20_000 <= float(value) <= 80_000:
        timestamp = pd.to_datetime(value, unit="D", origin="1899-12-30")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return str(value).strip()


def _normalize_location(value: Any) -> str:
    location = _normalize_context_value(value)
    return re.sub(
        rf"^(?:当前位置|{INTERACTION_LOCATION_LABEL})\s*[:：]\s*",
        "",
        location,
    ).strip()


def _row_location(row: pd.Series) -> str:
    for column, label in CONTEXT_SOURCE_COLUMNS:
        if label != INTERACTION_LOCATION_LABEL or column not in row.index:
            continue
        location = _normalize_location(row.get(column))
        if location:
            return location
    return ""


def _build_context(row: pd.Series, *, current_location: str) -> str:
    explicit_context = row.get(CONTEXT_COLUMN)
    if not _is_empty(explicit_context):
        context = str(explicit_context).strip()
        if re.search(r"(?:当前位置|交互发生位置)\s*[:：]", context):
            return context
        location = _row_location(row) or _normalize_location(current_location)
        return (
            f"{context}；{INTERACTION_LOCATION_LABEL}：{location}"
            if location else context
        )

    parts: list[str] = []
    labels_seen: set[str] = set()
    for column, label in CONTEXT_SOURCE_COLUMNS:
        if column not in row.index or label in labels_seen:
            continue
        value = _normalize_context_value(row.get(column))
        if label == INTERACTION_LOCATION_LABEL:
            value = _normalize_location(value)
        if value:
            parts.append(f"{label}：{value}")
            labels_seen.add(label)
    if INTERACTION_LOCATION_LABEL not in labels_seen:
        location = _normalize_location(current_location)
        if location:
            parts.append(f"{INTERACTION_LOCATION_LABEL}：{location}")
    return "；".join(parts)


def _first_nonempty(row: pd.Series, columns: tuple[str, ...], *, answer: bool = False) -> Any:
    for column in columns:
        if column in row.index and not _is_empty(row.get(column), answer=answer):
            return row.get(column)
    return None


def _parse_nonnegative_number(value: Any) -> tuple[float | None, str | None]:
    if _is_empty(value):
        return None, None
    if isinstance(value, bool):
        return None, "必须是非负有限数字"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "必须是非负有限数字"
    if not math.isfinite(number) or number < 0:
        return None, "必须是非负有限数字"
    return number, None


def _read_time(
    row: pd.Series,
    columns: tuple[str, ...],
) -> tuple[float | None, str | None]:
    value = _first_nonempty(row, columns)
    return _parse_nonnegative_number(value)


def _posix_fragment(value: Any) -> str:
    if _is_empty(value):
        return ""
    return str(value).replace("\\", "/").strip().strip("/")


def _is_empty_path(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and value.strip().lower() in EMPTY_PATH
    ) or _is_empty(value)


def _join_video_path(video_prefix: str, directory: Any, filename: Any) -> str:
    prefix = str(video_prefix).replace("\\", "/").strip()
    prefix_is_absolute = prefix.startswith("/")
    fragments = [
        _posix_fragment(prefix),
        _posix_fragment(directory),
        _posix_fragment(filename),
    ]
    fragments = [fragment for fragment in fragments if fragment]
    joined = PurePosixPath(*fragments).as_posix() if fragments else ""
    return f"/{joined}" if prefix_is_absolute else joined


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value)
    return safe.strip("._") or "unknown"


def _numbered_collision_name(column_name: str, existing: dict[str, Any]) -> str:
    """为冲突列生成稳定的 ``字段_1``、``字段_2`` 名称。"""
    if column_name not in existing:
        return column_name
    match = re.match(r"^(.*)_(\d+)$", column_name)
    base_name = (
        match.group(1)
        if match and match.group(1) in existing
        else column_name
    )
    suffix = 1
    while f"{base_name}_{suffix}" in existing:
        suffix += 1
    return f"{base_name}_{suffix}"


def _build_video_path(
    row: pd.Series,
    *,
    video_prefix: str,
    item_id: str,
    has_video_path_column: bool,
) -> tuple[str, str | None]:
    # 表级优先级：只要源表定义了 video_path 列，所有行都以该列为准，
    # 不再使用“文件路径 / 原文件名”拼接，避免已有路径被意外覆盖。
    if has_video_path_column:
        explicit_video_path = row.get(VIDEO_PATH_COLUMN)
        if not _is_empty_path(explicit_video_path):
            return str(explicit_video_path).replace("\\", "/").strip(), None
        return "__missing_video__", "missing_video_mapping"

    directory = row.get(VIDEO_DIRECTORY_COLUMN)
    filename = row.get(VIDEO_FILENAME_COLUMN)
    # “文件路径”只是 video_prefix 与文件名之间的可选子目录；
    # 只要“原文件名”存在，即可构造确定的录屏路径。
    if not _is_empty_path(filename):
        return _join_video_path(video_prefix, directory, filename), None

    placeholder = _join_video_path(
        video_prefix,
        "__missing_video__",
        f"{_safe_filename(item_id)}.mp4",
    )
    return placeholder, "missing_video_mapping"


def _video_exists(video_path: str, *, project_root: Path) -> bool:
    path = Path(video_path)
    if not path.is_absolute():
        path = project_root / path
    return (
        path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and path.stat().st_size > 0
    )


def _path_for_json(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _recover_video_path(
    row: pd.Series,
    *,
    sequence: str,
    video_prefix: str,
    project_root: Path,
) -> str | None:
    """拼接路径失效时，按当前序号和原文件名唯一找回视频。

    只接受唯一匹配，避免同名录屏被静默错配；源表内容不会被修改。
    """
    filename = row.get(VIDEO_FILENAME_COLUMN)
    if not sequence or _is_empty_path(filename):
        return None

    search_root = Path(str(video_prefix).replace("\\", "/"))
    if not search_root.is_absolute():
        search_root = project_root / search_root
    if not search_root.is_dir():
        return None

    filename_text = str(filename).strip()
    candidates = [
        path
        for path in search_root.rglob(filename_text)
        if path.is_file()
        and path.stat().st_size > 0
        and sequence in path.parts
    ]
    if len(candidates) != 1:
        return None
    return _path_for_json(candidates[0], project_root=project_root)


def _read_input(
    input_path: Path,
    *,
    sheet: str | int,
    encoding: str,
) -> tuple[pd.DataFrame, str | int | None]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        requested_encoding = str(encoding or "auto").strip().lower()
        candidates = (
            ("utf-8-sig", "gb18030")
            if requested_encoding == "auto"
            else (requested_encoding,)
        )
        last_error: UnicodeDecodeError | None = None
        for candidate in candidates:
            try:
                return pd.read_csv(
                    input_path,
                    keep_default_na=False,
                    dtype=object,
                    encoding=candidate,
                ), None
            except UnicodeDecodeError as exc:
                last_error = exc
        tried = "、".join(str(value) for value in candidates)
        raise ValueError(
            f"CSV 编码解析失败：{input_path}；已尝试 {tried}。"
            "可使用 --encoding 显式指定实际编码。"
        ) from last_error
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        if sheet != "auto":
            return pd.read_excel(
                input_path,
                sheet_name=sheet,
                keep_default_na=False,
                dtype=object,
            ), sheet
        workbook = pd.ExcelFile(input_path)
        candidates = list(workbook.sheet_names)
        if "合并数据" in candidates:
            candidates.remove("合并数据")
            candidates.insert(0, "合并数据")
        for sheet_name in candidates:
            frame = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                keep_default_na=False,
                dtype=object,
            )
            if len(frame.columns) and not frame.empty:
                return frame, sheet_name
        raise ValueError("Excel 中没有可导入的非空工作表")
    raise ValueError(
        f"不支持的输入格式 {suffix or '<无扩展名>'}；仅支持 .xlsx/.xls/.xlsm/.csv"
    )


def convert_table(
    input_path: Path,
    *,
    input_prefix: str | None = None,
    output_path: Path | None = None,
    video_prefix: str = DEFAULT_VIDEO_PREFIX,
    current_location: str = DEFAULT_CURRENT_LOCATION,
    sheet: str | int = 0,
    encoding: str = "auto",
    project_root: Path = PROJECT_ROOT,
) -> ConversionResult:
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    prefix = _normalize_prefix(input_prefix or input_path.stem)
    output_path = (output_path or input_path.with_suffix(".jsonl")).resolve()
    df, selected_sheet = _read_input(
        input_path,
        sheet=sheet,
        encoding=encoding,
    )
    raw_row_count = len(df)
    nonempty_mask = df.apply(
        lambda row: any(not _is_empty(value) for value in row),
        axis=1,
    )
    df = df.loc[nonempty_mask]
    ignored_empty_rows = raw_row_count - len(df)

    required = {QUERY_COLUMN}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"输入表格缺少必需列：{', '.join(missing_columns)}")
    has_id_column = "id" in df.columns
    has_index_column = INDEX_COLUMN in df.columns
    has_sequence_column = SEQUENCE_COLUMN in df.columns
    if not has_id_column and not has_index_column and not has_sequence_column:
        raise ValueError(
            f"输入表格必须至少包含 id、{INDEX_COLUMN} 或 {SEQUENCE_COLUMN} 列"
        )

    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing_video_ids: list[str] = []
    used_ids: dict[str, int] = {}
    has_context_column = CONTEXT_COLUMN in df.columns
    has_answer_column = "answer" in df.columns
    has_start_time_column = START_TIME_COLUMNS[0] in df.columns
    has_end_time_column = END_TIME_COLUMNS[0] in df.columns
    has_video_path_column = VIDEO_PATH_COLUMN in df.columns
    has_query_images_column = QUERY_IMAGES_COLUMN in df.columns
    has_query_image_path_column = QUERY_IMAGE_PATH_COLUMN in df.columns
    has_attachment_path_column = ATTACHMENT_PATH_COLUMN in df.columns
    direct_standard_columns = tuple(
        column
        for column, present in (
            (CONTEXT_COLUMN, has_context_column),
            ("answer", has_answer_column),
            (START_TIME_COLUMNS[0], has_start_time_column),
            (END_TIME_COLUMNS[0], has_end_time_column),
            (VIDEO_PATH_COLUMN, has_video_path_column),
            (QUERY_IMAGES_COLUMN, has_query_images_column),
        )
        if present
    )

    for zero_index, (source_index, row) in enumerate(df.iterrows()):
        source_row = (
            int(source_index) + 2
            if isinstance(source_index, Integral)
            else zero_index + 2
        )
        row_warnings: list[str] = []

        sequence = (
            _normalize_identifier(row.get(SEQUENCE_COLUMN))
            if has_sequence_column else ""
        )
        id_source_column = (
            INDEX_COLUMN if has_index_column
            else SEQUENCE_COLUMN if has_sequence_column else "id"
        )
        id_suffix = _normalize_identifier(row.get(id_source_column))
        item_id = _normalize_identifier(row.get("id"))
        if not item_id:
            if not id_suffix:
                id_suffix = f"excel_row_{source_row:04d}"
                row_warnings.append(f"missing_{id_source_column}")
            item_id = f"{prefix}_{id_suffix}"
        if item_id in used_ids:
            first_row = used_ids[item_id]
            raise ValueError(
                f"输入或生成的 id 重复：{item_id}，首次出现在第 {first_row} 行，"
                f"再次出现在第 {source_row} 行"
            )
        used_ids[item_id] = source_row

        raw_query = row.get(QUERY_COLUMN)
        if _is_empty(raw_query):
            query = f"[缺失query：输入表第{source_row}行]"
            row_warnings.append("missing_query")
        else:
            query = str(raw_query).strip()

        item: dict[str, Any] = {"id": item_id}
        if has_index_column:
            item[INDEX_COLUMN] = _json_value(row.get(INDEX_COLUMN))
        if has_sequence_column:
            item[SEQUENCE_COLUMN] = sequence
        context = (
            _json_value(row.get(CONTEXT_COLUMN))
            if has_context_column
            else _build_context(row, current_location=current_location)
        )
        item.update({"query": query, "context": context})

        raw_query_images = _first_nonempty(
            row,
            tuple(
                column
                for column, present in (
                    (QUERY_IMAGES_COLUMN, has_query_images_column),
                    (QUERY_IMAGE_PATH_COLUMN, has_query_image_path_column),
                    (ATTACHMENT_PATH_COLUMN, has_attachment_path_column),
                )
                if present
            ),
        )
        try:
            query_images = _table_query_images(raw_query_images)
        except ValueError as exc:
            query_images = []
            row_warnings.append(f"invalid_query_images:{exc}")
        if query_images:
            item[QUERY_IMAGES_COLUMN] = query_images

        if has_answer_column:
            item["answer"] = _json_value(row.get("answer"))
        else:
            answer = _first_nonempty(row, ANSWER_COLUMNS[:-1], answer=True)
            if answer is not None:
                item["answer"] = str(answer).strip()

        start_time, start_error = _read_time(
            row,
            START_TIME_COLUMNS[:1] if has_start_time_column
            else START_TIME_COLUMNS[1:],
        )
        end_time, end_error = _read_time(
            row,
            END_TIME_COLUMNS[:1] if has_end_time_column
            else END_TIME_COLUMNS[1:],
        )
        if start_error:
            row_warnings.append(f"invalid_task_start_time:{start_error}")
        elif start_time is not None:
            item["task_start_time"] = start_time
        if end_error:
            row_warnings.append(f"invalid_task_end_time:{end_error}")
        elif end_time is not None:
            item["task_end_time"] = end_time
        if (
            start_time is not None
            and end_time is not None
            and end_time <= start_time
        ):
            item.pop("task_end_time", None)
            row_warnings.append("invalid_task_time_order")

        video_path, video_mapping_warning = _build_video_path(
            row,
            video_prefix=video_prefix,
            item_id=item_id,
            has_video_path_column=has_video_path_column,
        )
        item["video_path"] = video_path
        if video_mapping_warning:
            row_warnings.append(video_mapping_warning)
        # URL 延迟到开始评估时下载、校验，导入时不作本地存在性判断。
        if not video_path.lower().startswith(("http://", "https://")) and not _video_exists(video_path, project_root=project_root):
            recovered_video_path = _recover_video_path(
                row,
                sequence=sequence or id_suffix,
                video_prefix=video_prefix,
                project_root=project_root,
            ) if not has_video_path_column else None
            if recovered_video_path is not None:
                item["video_path"] = recovered_video_path
                row_warnings.append("video_path_recovered_by_sequence_and_filename")
            else:
                row_warnings.append("video_file_missing_or_empty")
                missing_video_ids.append(item_id)

        # 标准关键字段保持在前面；其余原表字段按原列顺序追加。
        # 含义一致且直接采用的标准列不重复；若系统生成值占用了原始列名，
        # 则以 _1、_2 这样的数字后缀保留原始值，避免静默丢列。
        direct_source_columns = {
            "id",
            QUERY_COLUMN,
            *(column for column in (INDEX_COLUMN, SEQUENCE_COLUMN) if column in df.columns),
            *(column for column in direct_standard_columns if column in df.columns),
        }
        for column in df.columns:
            column_name = str(column)
            if column_name in item and column_name in direct_source_columns:
                continue
            target_name = _numbered_collision_name(column_name, item)
            item[target_name] = _json_value(row.get(column))

        rows.append(item)
        if row_warnings:
            warnings.append(
                {
                    "输入行号": source_row,
                    "id": item_id,
                    "问题": row_warnings,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(
                json.dumps(item, ensure_ascii=False, allow_nan=False)
            )
            handle.write("\n")

    return ConversionResult(
        output_path=output_path,
        rows=rows,
        warnings=warnings,
        missing_video_ids=missing_video_ids,
        ignored_empty_rows=ignored_empty_rows,
        direct_standard_columns=direct_standard_columns,
        selected_sheet=selected_sheet,
    )


def _parse_sheet(value: str) -> str | int:
    value = value.strip()
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 XLSX/CSV 保序转换为任务类 JSONL 数据集"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="输入 .xlsx/.xls/.xlsm/.csv 文件",
    )
    parser.add_argument(
        "--input",
        "--excel",
        dest="input_option",
        type=Path,
        help="输入文件；与位置参数二选一，--excel 作为兼容别名",
    )
    parser.add_argument(
        "--input-prefix",
        help="id 为空时生成题号的前缀，例如 0730众测；默认使用输入文件名",
    )
    parser.add_argument(
        "--video-prefix",
        default=DEFAULT_VIDEO_PREFIX,
        help=f"video_path 前缀，默认 {DEFAULT_VIDEO_PREFIX}",
    )
    parser.add_argument(
        "--current-location",
        default=DEFAULT_CURRENT_LOCATION,
        help=f"context 默认交互发生位置，默认 {DEFAULT_CURRENT_LOCATION}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 JSONL；默认与输入文件同目录、同名",
    )
    parser.add_argument(
        "--sheet",
        type=_parse_sheet,
        default=0,
        help="Excel 工作表名称、从 0 开始的序号或 auto；默认 0",
    )
    parser.add_argument(
        "--encoding",
        default="auto",
        help="CSV 编码；默认 auto，依次尝试 utf-8-sig 和 gb18030",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_file and args.input_option:
        parser.error("位置参数和 --input/--excel 不能同时指定")
    input_path = args.input_file or args.input_option
    if input_path is None:
        parser.error("必须指定一个输入 XLSX/CSV 文件")

    result = convert_table(
        input_path,
        input_prefix=args.input_prefix,
        output_path=args.output,
        video_prefix=args.video_prefix,
        current_location=args.current_location,
        sheet=args.sheet,
        encoding=args.encoding,
    )

    if result.direct_standard_columns:
        columns = "、".join(result.direct_standard_columns)
        print(
            "警告：输入表包含可自动生成的同名标准字段，"
            f"将直接使用原列，不再执行对应的自动拼接或映射：{columns}"
        )
    print("===== 转换完成 =====")
    print(f"输入行数：{len(result.rows)}")
    print(f"JSONL行数：{len(result.rows)}（严格保序，不丢行）")
    print(f"忽略完全空白行：{result.ignored_empty_rows}")
    print(f"有警告的行数：{len(result.warnings)}")
    print(f"磁盘缺失或空视频：{len(result.missing_video_ids)}")
    if result.missing_video_ids:
        preview = "、".join(result.missing_video_ids[:10])
        suffix = "……" if len(result.missing_video_ids) > 10 else ""
        print(f"缺失视频id：{preview}{suffix}")
    print(f"JSONL：{result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
