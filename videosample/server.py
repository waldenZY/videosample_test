"""单服务、文件存储的抽帧实验台；不导入任何模型或评估流程。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from numbers import Real
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import media
from . import algorithms
from .download import download_remote_video, is_remote_video_url
from .preparation import _operation_timing, resolve_operation_video_path
from .table_dataset import convert_table
from .web.parse_input import parse_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("VIDEOSAMPLE_RUNS", str(ROOT / "runs"))).resolve()
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame")
VIDEO_LOCKS = {}


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


@app.get("/api/algorithms")
def list_algorithms():
    return [algorithms.snapshot(key) for key in ("base", "current")]


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
           "version": algorithms.snapshot("current")["algorithm_version"],
           "algorithm": algorithms.snapshot("current", max_frames=15), "run_mode": "current",
           "experiment_max_frames": 15, "status": "imported",
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
            if run.get("group_id") and run["id"] != run["group_id"]:
                continue
            chains = chain_summaries(run)
            combined_status = "running" if any(c["status"] == "running" for c in chains) else run["status"]
            result.append({key: run[key] for key in ("id", "name", "status", "created_at", "version")}
                          | {"status": combined_status, "count": sum(c["count"] for c in chains),
                             "algorithm": run.get("algorithm"),
                             "chains": chains, "done": sum(c["done"] for c in chains)})
        except (OSError, ValueError, HTTPException):
            continue
    return sorted(result, key=lambda row: row["created_at"], reverse=True)


@app.get("/api/runs/{identity}")
def get_run(identity: str):
    run = read_run(identity)
    run["statistics"] = chain_statistics(run)
    run["chains"] = chain_summaries(run)
    return run


def chain_statistics(run):
    """只统计本次链路的成功样本；缺失数据不按零计入平均值。"""
    rows = run.get("rows", [])
    successful = [row for row in rows if row.get("status") == "done"]

    def metric(values):
        valid = [float(value) for value in values if isinstance(value, Real)
                 and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
        return {"samples": len(valid), "total": round(sum(valid), 3) if valid else None,
                "mean": round(sum(valid) / len(valid), 3) if valid else None}

    durations = [(row.get('timings_s') or {}).get('extraction') for row in successful]
    durations = [float(value) for value in durations if isinstance(value, Real)
                 and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
    distribution = {'1-5': 0, '6-10': 0, '11-14': 0, '15': 0, 'other': 0}
    distribution_samples = 0
    for row in successful:
        value = row.get('frame_count')
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        bucket = '1-5' if 1 <= value <= 5 else '6-10' if 6 <= value <= 10 else '11-14' if 11 <= value <= 14 else '15' if value == 15 else 'other'
        distribution[bucket] += 1
        distribution_samples += 1
    selections = [row['selection'] for row in successful if isinstance(row.get('selection'), dict)
                  and row['selection'].get('stop_reason') in {'covered', 'no_reliable_addition', 'limit_reached'}]
    reasons = {key: sum(selection['stop_reason'] == key for selection in selections)
               for key in ('covered', 'no_reliable_addition', 'limit_reached')}
    return {"total_samples": len(rows),
            "selected_samples": sum(row.get("status") != "skipped" for row in rows),
            "successful_samples": len(successful),
            "failed_samples": sum(row.get("status") in {"error", "interrupted"} for row in rows),
            "sample_time_s": metric(row.get("elapsed_s") for row in successful),
            "extraction_time_s": metric((row.get("timings_s") or {}).get("extraction") for row in successful),
            "extraction_max_s": round(max(durations), 3) if durations else None,
            "output_distribution": {'samples': distribution_samples, 'counts': distribution},
            "adaptive_selection": {'samples': len(selections), 'stop_reasons': reasons,
                                   'hit_limit_count': reasons['limit_reached'],
                                   'hit_limit_rate': round(reasons['limit_reached'] / len(selections), 4) if selections else None},
            "initial_frames": metric(row.get("initial_frame_count", (row.get("stage_counts") or {}).get("collected"))
                                     for row in successful),
            "final_frames": metric(row.get("frame_count") for row in successful)}


def chain_summaries(run):
    result = []
    for key, identity in run.get("chain_ids", {run.get("algorithm", {}).get("key", "base"): run["id"]}).items():
        chain = run if identity == run["id"] else read_run(identity)
        result.append({"key": key, "id": identity, "name": "base · 固定链路" if key == "base" else "实验链路",
                       "status": chain["status"], "count": len(chain["rows"]),
                       "version": chain.get("version"), "statistics": chain_statistics(chain),
                       "frame_limit": (chain.get("algorithm", {}).get("overrides", {}).get("max_frames")
                                       if key == "current" else None),
                       "done": sum(row["status"] == "done" for row in chain["rows"]),
                       "failed": sum(row["status"] in {"error", "interrupted"} for row in chain["rows"])})
    return result


class StartRequest(BaseModel):
    indices: list[int] | None = None
    name: str = ""
    run_mode: Literal["base", "current", "both"] = "both"
    experiment_max_frames: Literal[5, 10, 15] = 15


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
        keys = {"base": ["base"], "current": ["current"], "both": ["base", "current"]}[request.run_mode]
        chain_ids = {key: identity if index == 0 else uuid.uuid4().hex for index, key in enumerate(keys)}
        run.update(group_id=identity, chain_ids=chain_ids, run_mode=request.run_mode,
                   experiment_max_frames=request.experiment_max_frames)
        for key, chain_id in chain_ids.items():
            chain = deepcopy(run)
            chain["id"] = chain_id
            chain["algorithm"] = algorithms.snapshot(key, max_frames=request.experiment_max_frames)
            chain["version"] = chain["algorithm"]["algorithm_version"]
            save_run(chain)
            algorithms.save_source(key, run_directory(chain_id))
    for index in indices:
        for chain_id in chain_ids.values():
            POOL.submit(process_row, chain_id, index)
    return get_run(identity)


@app.post("/api/runs/{identity}/copy")
def copy_run(identity: str):
    run = read_run(identity)
    run["id"] = uuid.uuid4().hex
    run["name"] += " · 重跑"
    run["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    run["status"] = "imported"
    run.pop("group_id", None)
    run.pop("chain_ids", None)
    run["rows"] = [{"index": row["index"], "item": row["item"], "status": "ready", "message": "等待开始"}
                   for row in run["rows"]]
    run["run_mode"] = run.get("run_mode", "current")
    run["experiment_max_frames"] = 15
    run["algorithm"] = algorithms.snapshot("current" if run["run_mode"] == "current" else "base",
                                           max_frames=run["experiment_max_frames"])
    run["version"] = run["algorithm"]["algorithm_version"]
    with LOCK:
        save_run(run)
    return run


def prepared_video(run, index, item):
    """两条链路使用同一份视频；远程视频每题只下载一次。"""
    if not is_remote_video_url(item["video_path"]):
        return resolve_operation_video_path(item["video_path"], base_dir=ROOT)
    group_id = run.get("group_id", run["id"])
    with LOCK:
        lock = VIDEO_LOCKS.setdefault((group_id, index), threading.Lock())
    directory = run_directory(group_id) / "inputs" / f"{index:05d}"
    with lock:
        path = directory / "video.mp4"
        if path.is_file():
            return path
        return download_remote_video(item["video_path"], directory)


def process_row(identity: str, index: int):
    started = time.monotonic()
    directory = run_directory(identity) / "items" / f"{index:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    item = read_run(identity)["rows"][index]["item"]
    timings = {}
    recorder = None
    try:
        run = read_run(identity)
        algorithm = algorithms.implementation(run.get("algorithm", {}).get("key", "base"))
        update_row(identity, index, status="running", message="正在准备视频")
        if is_remote_video_url(item["video_path"]):
            update_row(identity, index, message="正在下载视频")
        with measure_phase(timings, "video_prepare"):
            video = prepared_video(run, index, item)
            (directory / "video_ref.json").write_text(json.dumps({"path": str(video)}, ensure_ascii=False), encoding="utf-8")
        with measure_phase(timings, "probe"):
            duration = algorithm.probe_duration(video)
            if duration <= 0:
                raise ValueError("无法读取视频或视频时长为 0")
            kwargs, _, warnings, _ = _operation_timing(item, duration,
                                                  default_start_time=algorithm.DEFAULT_TASK_START_TIME)
            if run.get("algorithm", {}).get("key") == "current":
                overrides = run["algorithm"].get("overrides", {})
                if "max_frames" in overrides:
                    kwargs["max_frames"] = overrides["max_frames"]
        with measure_phase(timings, "fingerprint"):
            with video.open("rb") as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        update_row(identity, index, message="正在抽帧并记录处理过程", duration=duration,
                   warnings=warnings, video_sha256=digest.hexdigest(), timings_s=timings)
        with measure_phase(timings, "extraction"):
            with algorithm.record_process(directory / "process") as recorder:
                frames = algorithm.extract_scene_keyframes(video, directory / "final", **kwargs)
        if hasattr(recorder, "timing_summary"):
            timings["stages"] = recorder.timing_summary()
        if not frames:
            raise ValueError("视频未能抽出有效图片，请检查视频与任务时间范围")
        counts = {stage["id"]: len(stage["output_ids"]) for stage in recorder.stages}
        metadata = json.loads((directory / "final" / "keyframes.json").read_text(encoding="utf-8"))
        update_row(identity, index, status="done", message="抽帧完成", frame_count=len(frames),
                   initial_frame_count=getattr(recorder, "initial_frame_count", counts.get("collected")),
                   effective_config=metadata["config"],
                   selection=metadata.get('selection'),
                   stages=[{"id": stage["id"], "title": stage["title"]} for stage in recorder.stages],
                   stage_counts=counts, timings_s=timings, elapsed_s=round(time.monotonic() - started, 2))
    except Exception as exc:
        if recorder is not None and hasattr(recorder, "timing_summary"):
            timings["stages"] = recorder.timing_summary()
        update_row(identity, index, status="error", message="本题失败", error=str(exc),
                   timings_s=timings, elapsed_s=round(time.monotonic() - started, 2))


@contextmanager
def measure_phase(timings, key):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round(time.perf_counter() - started, 4)


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
