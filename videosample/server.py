"""单服务、文件存储的抽帧实验台；不导入任何模型或评估流程。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import media
from .download import download_remote_video, is_remote_video_url
from .preparation import _operation_timing, resolve_operation_video_path
from .table_dataset import convert_table
from .web.parse_input import parse_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("VIDEOSAMPLE_RUNS", str(ROOT / "runs"))).resolve()
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame")
STAGE_NAMES = {"collected": "最初收集", "timestamps": "合并相近时刻",
               "range": "确定任务范围", "states": "合并重复页面",
               "endpoints": "保留任务首尾", "budget": "控制图片数量",
               "final": "最终结果"}


def run_directory(identity: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", identity):
        raise HTTPException(404, "批次不存在")
    directory = RUNS / identity
    if not (directory / "run.json").is_file():
        raise HTTPException(404, "批次不存在")
    return directory


def read_run(identity: str) -> dict:
    with LOCK:
        return json.loads((run_directory(identity) / "run.json").read_text(encoding="utf-8"))


def save_run(run: dict) -> None:
    directory = RUNS / run["id"]
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "run.tmp"
    temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "run.json")


def update_row(identity, index, **values):
    with LOCK:
        run = read_run(identity)
        run["rows"][index].update(values)
        statuses = [row["status"] for row in run["rows"]]
        if not any(status in {"queued", "running"} for status in statuses):
            run["status"] = "done"
        save_run(run)


@asynccontextmanager
async def lifespan(app):
    RUNS.mkdir(parents=True, exist_ok=True)
    with LOCK:
        for path in RUNS.glob("*/run.json"):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                if run["status"] == "running":
                    run["status"] = "interrupted"
                    for row in run["rows"]:
                        if row["status"] in {"queued", "running"}:
                            row.update(status="interrupted", error="服务已重启，本题未完成；可新建一次重跑")
                    save_run(run)
            except (OSError, ValueError, KeyError):
                continue
    yield


app = FastAPI(title="抽帧实验台", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    try:
        media.resolve_ffmpeg_executable()
        return {"ok": True, "ffmpeg": True, "version": media.KEYFRAME_ALGORITHM_VERSION}
    except media.FFmpegUnavailableError as exc:
        return {"ok": False, "ffmpeg": False, "error": str(exc)}


@app.post("/api/import")
def import_table(file: UploadFile = File(...)):
    filename = Path((file.filename or "dataset.xlsx").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".xls", ".xlsm", ".csv", ".jsonl"}:
        raise HTTPException(422, "请选择 Excel、CSV 或 JSONL 文件")
    content = file.file.read(100 * 1024 * 1024 + 1)
    if not content or len(content) > 100 * 1024 * 1024:
        raise HTTPException(422, "文件为空或超过 100MB")
    identity = uuid.uuid4().hex
    directory = RUNS / identity
    directory.mkdir(parents=True)
    source = directory / f"input{suffix}"
    source.write_bytes(content)
    try:
        if suffix == ".jsonl":
            jsonl = content.decode("utf-8-sig")
            warnings, sheet = [], None
            summary = {}
        else:
            converted = convert_table(source, input_prefix=Path(filename).stem,
                                      output_path=directory / "converted.jsonl", sheet="auto",
                                      project_root=ROOT)
            jsonl = "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in converted.rows)
            warnings, sheet = converted.warnings, converted.selected_sheet
            summary = {"source_rows": len(converted.rows), "ignored_empty_rows": converted.ignored_empty_rows,
                       "missing_video_rows": len(converted.missing_video_ids)}
        items, errors = parse_jsonl(jsonl, "operation")
    except (ValueError, ImportError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    run = {"id": identity, "name": Path(filename).stem,
           "filename": filename, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "version": media.KEYFRAME_ALGORITHM_VERSION, "status": "imported",
           "warnings": warnings, "errors": errors, "sheet": sheet, "summary": summary,
           "rows": [{"index": index, "item": item, "status": "ready", "message": "等待开始"}
                    for index, item in enumerate(items)]}
    with LOCK:
        save_run(run)
    return run


@app.get("/api/runs")
def list_runs():
    result = []
    for path in RUNS.glob("*/run.json"):
        try:
            run = read_run(path.parent.name)
            result.append({key: run[key] for key in ("id", "name", "status", "created_at", "version")}
                          | {"count": len(run["rows"]),
                             "done": sum(row["status"] == "done" for row in run["rows"])})
        except (OSError, ValueError, HTTPException):
            continue
    return sorted(result, key=lambda row: row["created_at"], reverse=True)


@app.get("/api/runs/{identity}")
def get_run(identity: str):
    return read_run(identity)


class StartRequest(BaseModel):
    indices: list[int] | None = None
    name: str = ""


@app.post("/api/runs/{identity}/start")
def start(identity: str, request: StartRequest):
    with LOCK:
        run = read_run(identity)
        if run["status"] != "imported":
            raise HTTPException(409, "本次已开始，请新建一次重跑")
        indices = sorted(set(request.indices if request.indices is not None else range(len(run["rows"]))))
        if not indices or any(index < 0 or index >= len(run["rows"]) for index in indices):
            raise HTTPException(422, "请勾选有效样本")
        selected = set(indices)
        for row in run["rows"]:
            row.update(status="queued" if row["index"] in selected else "skipped",
                       message="等待抽帧" if row["index"] in selected else "未勾选")
        run["name"] = request.name.strip() or run["name"]
        run["status"] = "running"
        save_run(run)
    for index in indices:
        POOL.submit(process_row, identity, index)
    return read_run(identity)


@app.post("/api/runs/{identity}/copy")
def copy_run(identity: str):
    run = read_run(identity)
    run["id"] = uuid.uuid4().hex
    run["name"] += " · 重跑"
    run["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    run["status"] = "imported"
    run["rows"] = [{"index": row["index"], "item": row["item"], "status": "ready", "message": "等待开始"}
                   for row in run["rows"]]
    run["version"] = media.KEYFRAME_ALGORITHM_VERSION
    with LOCK:
        save_run(run)
    return run


def process_row(identity: str, index: int):
    started = time.monotonic()
    directory = run_directory(identity) / "items" / f"{index:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    item = read_run(identity)["rows"][index]["item"]
    try:
        update_row(identity, index, status="running", message="正在准备视频")
        raw = item["video_path"]
        if is_remote_video_url(raw):
            update_row(identity, index, message="正在下载视频")
            video = download_remote_video(raw, directory / "video")
        else:
            video = resolve_operation_video_path(raw, base_dir=ROOT)
        (directory / "video_ref.json").write_text(json.dumps({"path": str(video)}, ensure_ascii=False), encoding="utf-8")
        duration = media.probe_duration(video)
        if duration <= 0:
            raise ValueError("无法读取视频或视频时长为 0")
        kwargs, _, warnings, _ = _operation_timing(item, duration)
        with video.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        update_row(identity, index, message="正在抽帧并记录处理过程", duration=duration,
                   warnings=warnings, video_sha256=digest.hexdigest())
        with media.record_process(directory / "process") as recorder:
            frames = media.extract_scene_keyframes(video, directory / "final", **kwargs)
        if not frames:
            raise ValueError("视频未能抽出有效图片，请检查视频与任务时间范围")
        counts = {stage["id"]: len(stage["output_ids"]) for stage in recorder.stages}
        update_row(identity, index, status="done", message="抽帧完成", frame_count=len(frames),
                   stage_counts=counts, elapsed_s=round(time.monotonic() - started, 2))
    except Exception as exc:
        update_row(identity, index, status="error", message="本题失败", error=str(exc),
                   elapsed_s=round(time.monotonic() - started, 2))


def item_directory(identity: str, index: int):
    run = read_run(identity)
    if index < 0 or index >= len(run["rows"]):
        raise HTTPException(404, "样本不存在")
    return run_directory(identity) / "items" / f"{index:05d}"


@app.get("/api/runs/{identity}/items/{index}/trace")
def trace(identity: str, index: int):
    directory = item_directory(identity, index)
    path = directory / "process" / "trace.json"
    if not path.is_file():
        raise HTTPException(404, "暂时没有过程结果，请等待抽帧完成")
    result = json.loads(path.read_text(encoding="utf-8"))
    metadata = directory / "final" / "keyframes.json"
    if metadata.is_file():
        result["metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
    return result


@app.get("/api/runs/{identity}/items/{index}/frames/{frame_id}")
def image(identity: str, index: int, frame_id: str, thumb: bool = False):
    if not re.fullmatch(r"f[0-9]{5,}", frame_id):
        raise HTTPException(404, "图片不存在")
    directory = item_directory(identity, index) / "process"
    path = directory / "candidates" / f"{frame_id}.jpg"
    if not path.is_file():
        raise HTTPException(404, "图片不存在")
    if thumb:
        from PIL import Image
        thumbnail = directory / "thumbs" / path.name
        with LOCK:
            if not thumbnail.is_file():
                thumbnail.parent.mkdir(exist_ok=True)
                with Image.open(path) as picture:
                    picture.thumbnail((260, 300))
                    picture.convert("RGB").save(thumbnail, quality=78)
        path = thumbnail
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/runs/{identity}/items/{index}/video")
def video(identity: str, index: int):
    directory = item_directory(identity, index)
    reference = directory / "video_ref.json"
    if not reference.is_file():
        raise HTTPException(404, "视频尚未准备好")
    path = Path(json.loads(reference.read_text(encoding="utf-8"))["path"])
    if not path.is_file():
        raise HTTPException(404, "原视频已不存在")
    return FileResponse(path)
