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
import io
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from numbers import Real
from typing import Literal
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import media
from . import algorithms
from .download import download_remote_video, is_remote_video_url
from .preparation import _operation_timing, resolve_operation_video_path
from .table_dataset import convert_table
from .web.parse_input import parse_jsonl
from .storage import managed_directory, remove_download, remove_tombstone

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("VIDEOSAMPLE_RUNS", str(ROOT / "runs"))).resolve()
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame")
VIDEO_LOCKS = {}
ACTIVE_JOBS = set()
LOCAL_TIME = timezone(timedelta(hours=8), name='Asia/Shanghai')


def task_name(filename, created_at):
    return f"{created_at[:19].replace('T', ' ')}_{filename}"


def saved_config(run):
    """Only saved defaults/overrides; never infer historical values from today's code."""
    algorithm = run.get('algorithm') or {}
    return {**algorithm.get('defaults', {}), **algorithm.get('overrides', {})}


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
    for path in RUNS.glob('.deleting-*'):
        try:
            remove_tombstone(RUNS, path.name)
        except (OSError, ValueError):
            # Retain a failed deletion for the next restart; never recreate the run.
            continue
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
    # Recover downloads left by an interrupted process. Old records without a
    # retention setting keep their files; only explicitly opted-out runs qualify.
    for path in RUNS.glob('*/run.json'):
        try:
            run = read_run(path.parent.name)
            if run.get('group_id', run['id']) != run['id'] or run.get('retain_downloaded_video', True):
                continue
            for folder in (path.parent / 'inputs').glob('*'):
                if folder.name.isdigit():
                    cleanup_sample_download(run['id'], int(folder.name))
        except (OSError, ValueError, HTTPException):
            continue
    yield


app = FastAPI(title="抽帧实验台", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    labels = {'experiment_max_frames': '最多保留张数须为至少 2 的整数',
              'experiment_sample_fps': '每秒采样帧数须为大于 0 的有限数字，可以填小数',
              'experiment_selection_scoring': '实验评分固定为新增比例，不能选择其他策略'}
    messages = [labels[e['loc'][-1]] for e in exc.errors() if e.get('loc') and e['loc'][-1] in labels]
    errors = json.loads(json.dumps(exc.errors(), default=str), parse_constant=lambda value: value)
    return JSONResponse(status_code=422, content={'detail': '；'.join(dict.fromkeys(messages)) if messages else errors})


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
    created_at = datetime.now(LOCAL_TIME).isoformat(timespec='seconds')
    run = {"id": identity, "name": task_name(filename, created_at),
           "filename": filename, "created_at": created_at,
           "version": algorithms.snapshot("current")["algorithm_version"],
           "algorithm": algorithms.snapshot("current", max_frames=15), "run_mode": "current",
           "experiment_max_frames": 15, "experiment_sample_fps": 3.0,
           "experiment_selection_scoring": "new_content_ratio", "retain_downloaded_video": False, "status": "imported",
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
                             "retain_downloaded_video": run.get('retain_downloaded_video', True),
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


class AnnotationRequest(BaseModel):
    verdict: Literal['missing', 'no_missing'] | None
    note: str = Field(default='', max_length=2000)


@app.post('/api/runs/{identity}/items/{index}/annotation')
def save_annotation(identity: str, index: int, request: AnnotationRequest):
    with LOCK:
        run = read_run(identity)
        if not 0 <= index < len(run['rows']):
            raise HTTPException(404, '样本不存在')
        row = run['rows'][index]
        if row['status'] != 'done':
            raise HTTPException(409, '样本抽帧完成后可标注')
        note = request.note.strip()
        annotation = None
        if request.verdict is not None or note:
            annotation = {'verdict': request.verdict, 'note': note,
                          'updated_at': datetime.now(LOCAL_TIME).isoformat(timespec='seconds')}
            row['annotation'] = annotation
        else:
            row.pop('annotation', None)
        save_run(run)
    return {'annotation': annotation}


def related_runs(run):
    """Read and validate a whole batch, including a request made via its child."""
    root = read_run(run.get('group_id', run['id']))
    ids = list(dict.fromkeys((root.get('chain_ids') or {'single': root['id']}).values()))
    if root['id'] not in ids or run['id'] not in ids:
        raise HTTPException(409, '批次关联信息不完整，不能清理')
    chains = [read_run(identity) for identity in ids]
    if any(c.get('group_id', c['id']) != root['id'] for c in chains):
        raise HTTPException(409, '批次关联信息不一致，不能清理')
    return root, chains


def cleanup_sample_download(identity, index):
    with LOCK:
        run = read_run(identity)
        if run.get('retain_downloaded_video', True):
            return
        root, chains = related_runs(run)
        if root.get('retain_downloaded_video', True):
            return
        if index < 0 or any(index >= len(c['rows']) for c in chains):
            return
        if any((c['id'], index) in ACTIVE_JOBS or c['rows'][index]['status'] in {'queued', 'running'} for c in chains):
            return
        VIDEO_LOCKS.pop((root['id'], index), None)
        if not is_remote_video_url(root['rows'][index]['item'].get('video_path', '')):
            return
        try:
            removed = remove_download(RUNS, root['id'], index)
        except (OSError, ValueError) as exc:
            for chain in chains:
                row = chain['rows'][index]
                message = f'下载副本清理失败，文件暂留：{exc}'
                row['download_cleanup_error'] = message
                row['warnings'] = [w for w in row.get('warnings', []) if not w.startswith('下载副本清理失败')] + [message]
                save_run(chain)
        else:
            if removed:
                for chain in chains:
                    row = chain['rows'][index]
                    if row.pop('download_cleanup_error', None):
                        row['warnings'] = [w for w in row.get('warnings', []) if not w.startswith('下载副本清理失败')]
                        save_run(chain)


@app.delete('/api/runs/{identity}')
def delete_run(identity: str):
    renamed = []
    with LOCK:
        root, chains = related_runs(read_run(identity))
        ids = {c['id'] for c in chains}
        if any(job[0] in ids for job in ACTIVE_JOBS) or any(
            c.get('status') == 'running' or any(r['status'] in {'queued', 'running'} for r in c['rows']) for c in chains):
            raise HTTPException(409, '本批仍有排队或运行任务，请全部结束后再删除')
        try:
            # Validate every absolute target before any recursive removal/move.
            sources = [managed_directory(RUNS, c['id']) for c in chains]
            targets = [managed_directory(RUNS, '.deleting-' + uuid.uuid4().hex, deleting=True) for _ in chains]
            for source, target in zip(sources, targets):
                source.rename(target)
                renamed.append((source, target))
        except (OSError, ValueError) as exc:
            for source, target in reversed(renamed):
                target.rename(source)
            raise HTTPException(409, f'暂时无法删除批次：{exc}') from exc
        for key in list(VIDEO_LOCKS):
            if key[0] in ids:
                VIDEO_LOCKS.pop(key, None)
    failures = []
    for _, target in renamed:
        try:
            remove_tombstone(RUNS, target.name)
        except (OSError, ValueError):
            failures.append(target.name)
    return {'deleted_ids': sorted(ids), 'cleanup_pending': bool(failures),
            'message': '记录已删除；部分文件暂时被占用，下次启动将补清理' if failures else '已删除本批记录及占用文件'}


@app.get('/api/runs/{identity}/export.xlsx')
def export_run(identity: str):
    from .export import workbook_bytes
    run = read_run(identity)
    timestamp = datetime.now(LOCAL_TIME)
    config = saved_config(run)
    data = workbook_bytes(run, run_directory(identity), chain_statistics(run), timestamp.isoformat(timespec='seconds'))
    name = f"{run.get('name', identity)}_{run.get('algorithm', {}).get('key', '历史')}_{config.get('max_frames', '未记录')}张_{config.get('sample_fps', '未记录')}fps_{timestamp:%Y%m%d_%H%M%S}.xlsx"
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', name)
    return StreamingResponse(io.BytesIO(data), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             headers={'Content-Disposition': f"attachment; filename=results.xlsx; filename*=UTF-8''{quote(name)}"})


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
    distribution = {'1-5': 0, '6-10': 0, '11-14': 0, '15': 0, '16+': 0, 'other': 0}
    distribution_samples = 0
    for row in successful:
        value = row.get('frame_count')
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        bucket = '1-5' if 1 <= value <= 5 else '6-10' if 6 <= value <= 10 else '11-14' if 11 <= value <= 14 else '15' if value == 15 else '16+' if value > 15 else 'other'
        distribution[bucket] += 1
        distribution_samples += 1
    selections = [row['selection'] for row in successful if isinstance(row.get('selection'), dict)
                  and row['selection'].get('stop_reason') in {'covered', 'no_reliable_addition', 'limit_reached'}]
    reasons = {key: sum(selection['stop_reason'] == key for selection in selections)
               for key in ('covered', 'no_reliable_addition', 'limit_reached')}
    local = [row['local_analysis'] for row in successful if isinstance(row.get('local_analysis'), dict)]
    local_statistics = {'samples': len(local),
                        'elapsed_s': metric(value.get('elapsed_s') for value in local),
                        'input_frames': metric(value.get('input_frames') for value in local)}
    for key in ('pair_attempts', 'original_position_hits', 'full_search_attempts',
                'template_calls', 'template_cache_hits', 'failed_pairs'):
        local_statistics[key] = metric(value.get(key) for value in local)
    local_statistics['timing_breakdown'] = {
        key: metric((value.get('timing_breakdown') or {}).get(key) for value in local)
        for key in ('comparison_wall_s', 'background_strict_wall_s', 'other_preparation_wall_s',
                    'position_search_work_s', 'strict_check_work_s')}
    return {"total_samples": len(rows),
            "selected_samples": sum(row.get("status") != "skipped" for row in rows),
            "successful_samples": len(successful),
            "failed_samples": sum(row.get("status") in {"error", "interrupted"} for row in rows),
            "local_analysis": local_statistics,
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
                       "selection_scoring": (chain.get('algorithm', {}).get('overrides', {}).get('selection_scoring',
                           chain.get('algorithm', {}).get('defaults', {}).get('selection_scoring', 'content_weight'))
                           if key == 'current' else None),
                       "frame_limit": (chain.get("algorithm", {}).get("overrides", {}).get("max_frames")
                                       if key == "current" else None),
                       "effective_config": saved_config(chain),
                       "done": sum(row["status"] == "done" for row in chain["rows"]),
                       "failed": sum(row["status"] in {"error", "interrupted"} for row in chain["rows"])})
    return result


class StartRequest(BaseModel):
    indices: list[int] | None = None
    name: str = ""
    run_mode: Literal["base", "current", "both"] = "both"
    experiment_max_frames: int = Field(default=15, strict=True, ge=2)
    experiment_sample_fps: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    experiment_selection_scoring: Literal['new_content_ratio'] | None = None
    retain_downloaded_video: bool = Field(default=False, strict=True)

    @field_validator('experiment_sample_fps', mode='before')
    @classmethod
    def numeric_fps(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('每秒采样帧数必须是大于 0 的数字，可以填小数')
        return value


@app.post("/api/runs/{identity}/start")
def start(identity: str, request: StartRequest):
    with LOCK:
        run = read_run(identity)
        if run["status"] != "imported":
            raise HTTPException(409, "本次已开始，请新建一次重跑")
        indices = sorted(set(request.indices if request.indices is not None else range(len(run["rows"]))))
        if not indices or any(index < 0 or index >= len(run["rows"]) for index in indices):
            raise HTTPException(422, "请勾选有效样本")
        scoring = 'new_content_ratio'
        selected = set(indices)
        for row in run["rows"]:
            row.update(status="queued" if row["index"] in selected else "skipped",
                       message="等待抽帧" if row["index"] in selected else "未勾选")
        run["name"] = request.name.strip() or run["name"]
        run["status"] = "running"
        keys = {"base": ["base"], "current": ["current"], "both": ["base", "current"]}[request.run_mode]
        chain_ids = {key: identity if index == 0 else uuid.uuid4().hex for index, key in enumerate(keys)}
        run.update(group_id=identity, chain_ids=chain_ids, run_mode=request.run_mode,
                   experiment_max_frames=request.experiment_max_frames,
                   experiment_sample_fps=request.experiment_sample_fps,
                   retain_downloaded_video=request.retain_downloaded_video,
                   experiment_selection_scoring=scoring)
        for key, chain_id in chain_ids.items():
            chain = deepcopy(run)
            chain["id"] = chain_id
            chain["algorithm"] = algorithms.snapshot(key, max_frames=request.experiment_max_frames,
                                                       sample_fps=request.experiment_sample_fps,
                                                       selection_scoring=scoring)
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
    run["created_at"] = datetime.now(LOCAL_TIME).isoformat(timespec="seconds")
    run["name"] = task_name(run.get('filename', '历史输入'), run['created_at'])
    run["status"] = "imported"
    run.pop("group_id", None)
    run.pop("chain_ids", None)
    run["rows"] = [{"index": row["index"], "item": row["item"], "status": "ready", "message": "等待开始"}
                   for row in run["rows"]]
    run["run_mode"] = run.get("run_mode", "current")
    run["experiment_max_frames"] = 15
    run['experiment_sample_fps'] = 3.0
    run['retain_downloaded_video'] = False
    run['experiment_selection_scoring'] = 'new_content_ratio'
    run["algorithm"] = algorithms.snapshot("current" if run["run_mode"] == "current" else "base",
                                           max_frames=run["experiment_max_frames"],
                                           sample_fps=run['experiment_sample_fps'],
                                           selection_scoring=run['experiment_selection_scoring'])
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
    with LOCK:
        read_run(identity)
        ACTIVE_JOBS.add((identity, index))
    try:
        _process_row(identity, index)
    finally:
        with LOCK:
            ACTIVE_JOBS.discard((identity, index))
            cleanup_sample_download(identity, index)


def _process_row(identity: str, index: int):
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
                if 'sample_fps' in overrides:
                    kwargs['sample_fps'] = overrides['sample_fps']
                kwargs['selection_scoring'] = 'new_content_ratio'
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
                   local_analysis=getattr(recorder, 'local_analysis', None),
                   stages=[{"id": stage["id"], "title": stage["title"]} for stage in recorder.stages],
                   stage_counts=counts, timings_s=timings, elapsed_s=round(time.monotonic() - started, 2))
    except Exception as exc:
        if recorder is not None and hasattr(recorder, "timing_summary"):
            timings["stages"] = recorder.timing_summary()
        update_row(identity, index, status="error", message="本题失败", error=str(exc),
                   local_analysis=getattr(recorder, 'local_analysis', None),
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
    run = read_run(identity)
    if index < 0 or index >= len(run['rows']):
        raise HTTPException(404, '样本不存在')
    source = run['rows'][index]['item']['video_path']
    if is_remote_video_url(source):
        return RedirectResponse(source.strip(), status_code=307)
    try:
        path = resolve_operation_video_path(source, base_dir=ROOT)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path)
