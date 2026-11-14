"""任务类（录屏）的混合关键帧抽取与帧编码。

流程：固定频率与变化取图 → 原有时间/内容筛选 →
局部内容与代表图检查 → 按新增内容自适应输出。

FFmpeg 取图，Pillow/NumPy 比较；OpenCV 仅用于最后选图的局部检查。
"""
from __future__ import annotations

import base64
from bisect import bisect_left
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import lru_cache, wraps
from numbers import Integral, Real
from pathlib import Path

import numpy as np


KEYFRAME_ALGORITHM_VERSION = "hybrid-state-exp-v4.7"
DEFAULT_TASK_START_TIME = 7.0
DEFAULT_ANALYSIS_MASK_BOX = (.63, .005, .74, .040)

# 仅供实验台记录过程；不参与任何选帧决策。定稿迁回时可整体移除。
_PROCESS = ContextVar("keyframe_process", default=None)
_FEATURES = ContextVar("keyframe_features", default=None)
_FEATURE_KEYS = ContextVar("keyframe_feature_keys", default=None)
_SMALL_FEATURES = ContextVar("keyframe_small_features", default=None)
_ANALYSIS_MASK = ContextVar("keyframe_analysis_mask", default=DEFAULT_ANALYSIS_MASK_BOX)


def _cached_feature(function):
    @wraps(function)
    def cached(path):
        cache = _FEATURES.get()
        if cache is None:
            return function(path)
        small = function.__name__ in {'_layout_signature', '_assistant_shell_signature'}
        if small and _SMALL_FEATURES.get() is not None:
            cache = _SMALL_FEATURES.get()
        capacity = 4096 if small else 2048 if function.__name__ == '_content_signature' else 512
        identities = _FEATURE_KEYS.get()
        identity = str(path)
        if identities is not None:
            if identity not in identities:
                try:
                    # 候选图在单次抽帧中不再改写；字节完全相同的图可以复用同一份特征。
                    identities[identity] = hashlib.sha256(Path(path).read_bytes()).digest()
                except OSError:
                    identities[identity] = identity
                identity_capacity = 2048 if small or function.__name__ == '_content_signature' else 512
                if len(identities) > identity_capacity:
                    identities.popitem(last=False)
            else:
                identities.move_to_end(identity)
            identity = identities[identity]
        key = (function.__name__, identity)
        if key not in cache:
            cache[key] = function(path)
            if len(cache) > capacity:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return cache[key]
    return cached


def _with_feature_cache(function):
    @wraps(function)
    def cached(*args, **kwargs):
        token = _FEATURES.set(OrderedDict())
        identities_token = _FEATURE_KEYS.set(OrderedDict())
        small_token = _SMALL_FEATURES.set(OrderedDict())
        config = kwargs.get('config') or (args[2] if len(args) > 2 else None)
        mask_token = _ANALYSIS_MASK.set(getattr(config, 'analysis_mask_box', _ANALYSIS_MASK.get()))
        try:
            return function(*args, **kwargs)
        finally:
            _FEATURES.reset(token)
            _FEATURE_KEYS.reset(identities_token)
            _SMALL_FEATURES.reset(small_token)
            _ANALYSIS_MASK.reset(mask_token)
    return cached


def _prewarm_features(candidates, function):
    """并行读取/计算独立特征，结果按原输入顺序写入单次抽帧缓存。"""
    features, identities = _FEATURES.get(), _FEATURE_KEYS.get()
    if features is None or identities is None or len(candidates) < 8:
        return
    small = function.__name__ in {'_layout_signature', '_assistant_shell_signature'}
    cache = _SMALL_FEATURES.get() if small else features
    capacity = 4096 if small else 2048
    paths, seen = [], set()
    for candidate in candidates:
        path = candidate.path
        digest = identities.get(str(path))
        key = digest if isinstance(digest, bytes) else str(path)
        if key in seen or (function.__name__, digest) in cache:
            continue
        seen.add(key)
        paths.append(path)
    if not paths:
        return
    mask_box = _ANALYSIS_MASK.get()
    def load(path):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).digest()
        mask_token = _ANALYSIS_MASK.set(mask_box)
        try:
            value = function.__wrapped__(io.BytesIO(payload))
        finally:
            _ANALYSIS_MASK.reset(mask_token)
        return path, digest, value
    worker_count = min(8, max(1, (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='frame-feature') as workers:
        for path, digest, value in workers.map(load, paths):
            identities[str(path)] = digest
            identities.move_to_end(str(path))
            if len(identities) > 2048:
                identities.popitem(last=False)
            key = (function.__name__, digest)
            if key not in cache:
                cache[key] = value
                if len(cache) > capacity:
                    cache.popitem(last=False)


class _ProcessRecord:
    def __init__(self, directory: Path):
        self.directory = directory
        (directory / "candidates").mkdir(parents=True, exist_ok=True)
        self.identities = {}
        self.frames = []
        self.stages = []
        self.notes = {}
        self.initial_frame_count = None
        self.selection = None
        self.local_analysis = None
        self.timings = {}
        self.record_io_s = 0.0
        self.last_stage_end = time.perf_counter()

    def identify(self, candidate, *, copy_image=True):
        identity = self.identities.get(id(candidate))
        if identity is None:
            identity = f"f{len(self.frames) + 1:05d}"
            self.identities[id(candidate)] = identity
            target = self.directory / "candidates" / f"{identity}.jpg"
            if copy_image:
                shutil.copy2(candidate.path, target)
            self.frames.append({"id": identity, "time": round(candidate.time, 3),
                                "source": candidate.source,
                                "image": f"candidates/{identity}.jpg", "history": []})
        return identity

    def stage(self, key, title, before, after, fallback):
        started = time.perf_counter()
        elapsed = started - self.last_stage_end
        self.timings[key] = self.timings.get(key, 0.0) + elapsed
        if not self.frames and not before and after:
            copies = []
            for candidate in after:
                identity = self.identify(candidate, copy_image=False)
                copies.append((candidate.path, self.directory / 'candidates' / f'{identity}.jpg'))
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix='frame-copy') as workers:
                list(workers.map(lambda pair: shutil.copy2(*pair), copies))
        before_ids = [self.identify(c) for c in before]
        after_ids = [self.identify(c) for c in after]
        if self.initial_frame_count is None:
            self.initial_frame_count = len(after_ids)
        selected = set(after_ids)
        events = []
        seen = set()
        for candidate in before + after:
            identity = self.identify(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            note = self.notes.get(id(candidate), {})
            kept = identity in selected
            event = {"frame_id": identity, "stage": key,
                     "status": "kept" if kept else "dropped",
                     "reason": note.get("reason") or ("本步保留" if kept else fallback),
                     "related": [self.identify(c) for c in note.get("related", [])],
                     "details": note.get("details", {}), "keep_reason": candidate.keep_reason}
            events.append(event)
            self.frames[int(identity[1:]) - 1]["history"].append(event)
        self.stages.append({"id": key, "title": title, "elapsed_s": round(elapsed, 4), "input_ids": before_ids,
                            "output_ids": after_ids,
                            "removed_ids": [i for i in before_ids if i not in selected],
                            "events": events})
        self.notes.clear()
        self.record_io_s += time.perf_counter() - started
        self.save()
        self.last_stage_end = time.perf_counter()

    def save(self):
        started = time.perf_counter()
        payload = {"algorithm_version": KEYFRAME_ALGORITHM_VERSION,
                   "initial_frame_count": self.initial_frame_count,
                   "selection": self.selection,
                   "local_analysis": self.local_analysis,
                   "frames": self.frames, "stages": self.stages,
                   "timings_s": {**{key: round(value, 4) for key, value in self.timings.items()},
                                 "process_io": round(self.record_io_s, 4)}}
        target = self.directory / "trace.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding="utf-8")
        temporary.replace(target)
        self.record_io_s += time.perf_counter() - started

    def timing_summary(self):
        return {**{key: round(value, 4) for key, value in self.timings.items()},
                "process_io": round(self.record_io_s, 4)}


@contextmanager
def record_process(directory: Path | str):
    """实验时保留候选图与每步决定；不改变抽帧入口及返回值。"""
    recorder = _ProcessRecord(Path(directory))
    token = _PROCESS.set(recorder)
    completed = False
    try:
        yield recorder
        completed = True
    finally:
        try:
            key = "other_processing" if completed else "unfinished"
            recorder.timings[key] = time.perf_counter() - recorder.last_stage_end
            recorder.save()
        finally:
            _PROCESS.reset(token)


def _note(candidate, reason, related=None, **details):
    recorder = _PROCESS.get()
    if recorder is not None:
        recorder.notes[id(candidate)] = {"reason": reason, "related": related or [], "details": details}


def _stage(key, title, before, after, fallback="本步未选中"):
    recorder = _PROCESS.get()
    if recorder is not None:
        recorder.stage(key, title, before, after, fallback)


class FFmpegUnavailableError(RuntimeError):
    """系统和 Pip 视频依赖中都找不到可用的 FFmpeg。"""


def resolve_ffmpeg_executable() -> str:
    """解析 FFmpeg 路径：显式配置 → 系统 PATH → Pip Wheel。"""
    configured = os.getenv("AUTO_EVAL_FFMPEG", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        raise FFmpegUnavailableError(
            f"AUTO_EVAL_FFMPEG 指向的可执行文件不存在：{configured}"
        )

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        bundled_ffmpeg = Path(get_ffmpeg_exe())
        if bundled_ffmpeg.is_file():
            return str(bundled_ffmpeg.resolve())
    except (ImportError, RuntimeError, OSError):
        pass

    raise FFmpegUnavailableError(
        "未找到 FFmpeg。请运行 python -m pip install -e \".[video]\"，"
        "或安装系统 FFmpeg；安装说明见 docs/FFmpeg安装与抽帧验证.md"
    )


def _resolve_ffprobe_executable() -> str | None:
    """FFprobe 是可选加速项；缺失时使用 FFmpeg 元数据回退。"""
    configured = os.getenv("AUTO_EVAL_FFPROBE", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        raise FFmpegUnavailableError(
            f"AUTO_EVAL_FFPROBE 指向的可执行文件不存在：{configured}"
        )
    if os.getenv("AUTO_EVAL_FFMPEG", "").strip():
        return None
    return shutil.which("ffprobe")


@dataclass(frozen=True)
class KeyframeConfig:
    """操作录屏关键帧算法配置。"""

    task_start_time: float = DEFAULT_TASK_START_TIME
    task_end_time: float | None = None
    max_frames: int = 15
    sample_fps: float = 3.0
    scene_threshold: float = 0.02
    scene_min_gap_s: float = 0.15
    timestamp_merge_gap_s: float = 0.25
    state_layout_threshold: float = 0.05
    stable_min_duration_s: float = 1.5
    transient_max_duration_s: float = 4.0
    transient_return_threshold: float = 0.065
    transient_min_changed_fraction: float = 0.02
    transient_max_changed_fraction: float = 0.55
    auto_task_end_min_stable_duration_s: float = 1.5
    auto_task_end_min_final_stable_duration_s: float = 1.5
    auto_task_end_max_return_shell_distance: float = 0.02
    auto_task_end_min_task_shell_distance: float = 0.02
    auto_task_end_confidence_threshold: float = 0.60
    final_dedup_rms_threshold: float = 0.008
    final_dedup_changed_fraction_threshold: float = 0.01
    max_edge: int = 720
    content_rms_threshold: float = 0.025
    content_changed_fraction: float = 0.06
    content_tile_threshold: float = 0.06
    global_dedup_rms_threshold: float = 0.004
    global_dedup_changed_fraction: float = 0.003
    global_dedup_tile_threshold: float = 0.01
    global_dedup_patch_threshold: float = 0.02
    local_analysis_max_edge: int = 320
    local_template_threshold: float = 0.90
    local_content_bin_height: int = 24
    local_fast_alignment: bool = True
    local_search_reuse: bool = True
    local_match_workers: int = 2
    analysis_mask_box: tuple | None = DEFAULT_ANALYSIS_MASK_BOX
    scroll_dedup_enabled: bool = True
    scroll_analysis_max_edge: int = 800
    scroll_min_width: int = 400
    scroll_min_height: int = 800
    scroll_max_related: int = 3
    scroll_checks_per_candidate: int = 3
    scroll_agent_indicator_box: tuple | None = (.32, .955, .70, .995)
    # 受保护采样：在开头和结尾两个时间窗口内，每 interval 秒额外标记一帧，
    # 仅当画面高度一致时才去重（用于捕获弹窗/横幅/开场状态等开头结尾的小变化）。
    protected_sample_interval: float = 2.0  # 0 = 禁用；>0 则每 N 秒一帧
    protected_begin_window: float = 20.0    # 从 task_start_time 起的前 N 秒
    protected_end_window: float = 5.0       # 从 algorithm_end 止的后 N 秒
    protected_sample_dedup_rms: float = 0.002    # 受保护帧的去重 RMS 阈值
    protected_sample_dedup_cf: float = 0.002      # 受保护帧的去重变化比例阈值

    def __post_init__(self) -> None:
        """规范化并校验任务时间参数。"""
        for name, value in (
            ("task_start_time", self.task_start_time),
            ("task_end_time", self.task_end_time),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} 必须是有限数字")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} 必须是有限数字")
            if normalized < 0:
                raise ValueError(f"{name} 不能小于 0")
            object.__setattr__(self, name, normalized)
        if (
            self.task_end_time is not None
            and self.task_end_time <= self.task_start_time
        ):
            raise ValueError("task_end_time 必须大于 task_start_time")
        if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, Integral) or not 2 <= self.max_frames <= 15:
            raise ValueError("实验版 max_frames 必须为 2 到 15 的整数")
        if self.local_analysis_max_edge < 64 or self.local_content_bin_height < 4:
            raise ValueError("局部分析尺寸或内容分段过小")
        if not 0 < self.local_template_threshold <= 1:
            raise ValueError("局部匹配阈值必须在 0 到 1 之间")
        for name in ('local_fast_alignment', 'local_search_reuse'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} 必须为布尔值")
        if (isinstance(self.local_match_workers, bool) or not isinstance(self.local_match_workers, Integral)
                or not 1 <= self.local_match_workers <= 4):
            raise ValueError("local_match_workers 必须为 1 到 4 的整数")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps 必须大于 0")
        if self.analysis_mask_box is not None:
            box = self.analysis_mask_box
            if (not isinstance(box, (tuple, list)) or len(box) != 4
                    or not all(isinstance(v, Real) and math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                raise ValueError("分析遮挡区域必须为 0 到 1 的左、上、右、下坐标，或 None")
            object.__setattr__(self, 'analysis_mask_box', tuple(box))
        if not isinstance(self.scroll_dedup_enabled, bool):
            raise ValueError("scroll_dedup_enabled 必须为布尔值")
        for name, value, lower, upper in (
            ('scroll_analysis_max_edge', self.scroll_analysis_max_edge, 64, 2048),
            ('scroll_min_width', self.scroll_min_width, 1, 8192),
            ('scroll_min_height', self.scroll_min_height, 1, 8192),
            ('scroll_max_related', self.scroll_max_related, 1, 15),
            ('scroll_checks_per_candidate', self.scroll_checks_per_candidate, 1, 15),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f"{name} 必须为 {lower} 到 {upper} 的整数")
        if self.scroll_agent_indicator_box is not None:
            box = self.scroll_agent_indicator_box
            if (not isinstance(box, (tuple, list)) or len(box) != 4
                    or not all(isinstance(v, Real) and math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                raise ValueError("滑动检查遮挡区域必须为归一化坐标或 None")
            object.__setattr__(self, 'scroll_agent_indicator_box', tuple(box))
        if self.max_edge <= 0:
            raise ValueError("max_edge 必须大于 0")
        if (
            self.protected_sample_interval < 0
            or (self.protected_sample_interval > 0 and self.protected_sample_interval < 0.5)
        ):
            raise ValueError("protected_sample_interval 必须为 0 或 >= 0.5")
        if self.protected_begin_window < 0:
            raise ValueError("protected_begin_window 不能小于 0")
        if self.protected_end_window < 0:
            raise ValueError("protected_end_window 不能小于 0")


@dataclass
class _Candidate:
    time: float
    path: Path
    source: str
    novelty: float = 0.0
    keep_reason: str = ""
    support_s: float = 0.0
    page_id: int = -1
    evidence_ids: tuple[str, ...] = ()


def probe_duration(video: Path | str) -> float:
    """读取视频时长；优先 FFprobe，缺失时解析 FFmpeg 元数据。"""
    ffprobe = _resolve_ffprobe_executable()
    if ffprobe:
        try:
            out = subprocess.check_output(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video),
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return float(out.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    ffmpeg = resolve_ffmpeg_executable()
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(video)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return 0.0
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        proc.stderr or "",
    )
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def scene_change_times(
    video: Path | str,
    threshold: float = 0.02,
    *,
    end_time: float | None = None,
) -> list[float]:
    """返回 FFmpeg scene 检测到的变化时间点；失败返回空列表。"""
    command = [resolve_ffmpeg_executable(), "-hide_banner", "-i", str(video)]
    if end_time is not None:
        command.extend(["-t", f"{end_time:.3f}"])
    box = _ANALYSIS_MASK.get()
    mask_filter = (f'drawbox=x=iw*{box[0]}:y=ih*{box[1]}:'
                   f'w=iw*{box[2] - box[0]}:h=ih*{box[3] - box[1]}:color=black:t=fill,') if box else ''
    command.extend(
        [
            "-an",
            "-vf",
            f"{mask_filter}select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]
    )
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []
    return [
        float(value)
        for value in re.findall(r"pts_time:([0-9.]+)", proc.stderr or "")
    ]


def _dedup(times: list[float], min_gap_s: float) -> list[float]:
    """相邻时间点小于间隔时取先者。"""
    out: list[float] = []
    for timestamp in sorted(times):
        if not out or timestamp - out[-1] >= min_gap_s:
            out.append(timestamp)
    return out


def _even_sample(times: list[float], count: int) -> list[float]:
    if len(times) <= count:
        return list(times)
    positions = np.linspace(0, len(times) - 1, count).round().astype(int)
    return [times[position] for position in positions]


def select_keyframe_times(
    video: Path | str,
    max_frames: int = 15,
    min_frames: int = 4,
    threshold: float = 0.02,
    min_gap_s: float = 0.15,
    *,
    task_start_time: float = DEFAULT_TASK_START_TIME,
    task_end_time: float | None = None,
) -> list[float]:
    """兼容旧调用的轻量时间规划器。

    真正的混合算法需要查看帧内容，因此由 :func:`extract_scene_keyframes`
    执行；本函数保留用于无需解码图片的快速诊断和旧调用兼容。
    """
    config = KeyframeConfig(
        task_start_time=task_start_time,
        task_end_time=task_end_time,
        max_frames=max_frames,
        scene_threshold=threshold,
        scene_min_gap_s=min_gap_s,
    )
    duration = probe_duration(video)
    if duration <= 0:
        return []
    start = min(
        max(0.0, config.task_start_time),
        max(0.0, duration - 0.5),
    )
    end = (
        min(max(start, config.task_end_time), max(start, duration - 0.3))
        if config.task_end_time is not None
        else max(start, duration - 0.3)
    )
    times = [
        timestamp
        for timestamp in _dedup(
            scene_change_times(video, threshold, end_time=end), min_gap_s
        )
        if start <= timestamp <= end
    ]
    times.insert(0, start)
    if len(times) < min_frames:
        uniform = np.linspace(start, end, min_frames).tolist()
        times = sorted(
            {
                round(timestamp, 6)
                for timestamp in times + uniform
                if start <= timestamp <= end
            }
        )
    if len(times) > max_frames - 1:
        times = _even_sample(times, max_frames - 1)
    terminal = max(0.0, duration - 0.3)
    if not any(abs(timestamp - terminal) < 0.25 for timestamp in times):
        times.append(terminal)
    return sorted(times)


def _extract_at(
    video: Path,
    timestamp: float,
    output: Path,
    *,
    max_edge: int,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                resolve_ffmpeg_executable(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale='min({max_edge},iw)':-2",
                "-q:v",
                "3",
                str(output),
            ],
            check=False,
            capture_output=True,
        )
    except Exception:
        return False
    return output.is_file()


def _extract_many_at(
    video: Path, timestamps: list[float], directory: Path, *, max_edge: int,
) -> dict[float, Path]:
    """一次解码取多个时刻；仍取三位小数时间点处或其后的第一帧。"""
    if not timestamps:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    targets = sorted({float(f"{timestamp:.3f}") for timestamp in timestamps})
    # prev_pts 是前一输入帧，而非前一选中帧。每个目标只命中跨过它的那一帧。
    crossings = "+".join(f"gte(t,{target:.3f})*lt(prev_pts*TB,{target:.3f})"
                         for target in targets)
    expression = f"if(isnan(prev_pts),gte(t,{targets[0]:.3f}),{crossings})"
    script = directory / "select.ffscript"
    script.write_text(f"select='{expression}',showinfo,scale='min({max_edge},iw)':-2", encoding="utf-8")
    try:
        process = subprocess.run(
            [resolve_ffmpeg_executable(), "-y", "-hide_banner", "-i", str(video),
             "-t", f"{targets[-1] + 1:.3f}", "-an", "-filter_script:v", str(script),
             "-vsync", "0", "-q:v", "3", str(directory / "batch_%05d.jpg")],
            check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        times = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", process.stderr or "")]
        paths = sorted(directory.glob("batch_*.jpg"))
        if process.returncode or len(times) != len(paths):
            return {}
        result = {}
        for timestamp in timestamps:
            target = float(f"{timestamp:.3f}")
            index = bisect_left(times, target - 1e-6)
            if index < len(paths):
                result[timestamp] = paths[index]
        return result
    except (OSError, subprocess.SubprocessError):
        return {}


def _masked_analysis_image(image, box):
    """只修改 convert 生成的内存副本，不改变候选和最终输出图片。"""
    if box is not None:
        width, height = image.size
        x0, y0, x1, y1 = box
        image.paste(0, (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)))
    return image


@_cached_feature
def _signature(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as original:
        image = _masked_analysis_image(original.convert('L'), _ANALYSIS_MASK.get()).resize((96, 160))
    array = np.asarray(image, dtype=np.float32)
    return array[12:154, :]


@_cached_feature
def _layout_signature(path: Path) -> np.ndarray:
    from PIL import Image, ImageFilter

    with Image.open(path) as original:
        image = _masked_analysis_image(original.convert('L'), _ANALYSIS_MASK.get())
    image = image.filter(ImageFilter.GaussianBlur(radius=12)).resize((16, 27))
    array = np.asarray(image, dtype=np.float32)
    return array[2:-1, :]


@_cached_feature
def _assistant_shell_signature(path: Path) -> np.ndarray:
    from PIL import Image, ImageFilter

    with Image.open(path) as original:
        image = _masked_analysis_image(original.convert('L'), _ANALYSIS_MASK.get())
    image = image.filter(ImageFilter.GaussianBlur(radius=8)).resize((32, 54))
    array = np.asarray(image, dtype=np.float32)
    return array[-8:-2, :]


def _visual_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float]:
    delta = np.abs(left - right)
    rms = float(np.sqrt(np.mean(delta * delta)) / 255.0)
    changed_fraction = float(np.mean(delta >= 12.0))
    return rms, changed_fraction


@_cached_feature
def _content_signature(path: Path) -> np.ndarray:
    """相近时刻合并时看完整彩色画面，避免漏掉顶部提示或颜色变化。"""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(_masked_analysis_image(image.convert('RGB'), _ANALYSIS_MASK.get()).resize((96, 160)), dtype=np.uint8)


def _content_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    delta = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return (float(np.sqrt(np.mean(delta * delta)) / 255.0),
            float(np.mean(np.any(delta >= 12.0, axis=-1))))


def _layout_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)) / 255.0)


def _extract_candidates(
    video: Path,
    output_dir: Path,
    config: KeyframeConfig,
    duration: float,
) -> tuple[float | None, list[_Candidate]]:
    effective_start = min(
        max(0.0, config.task_start_time),
        max(0.0, duration - 0.5),
    )
    effective_task_end: float | None = None
    if config.task_end_time is not None:
        effective_task_end = min(
            max(effective_start, config.task_end_time),
            max(effective_start, duration - 0.3),
        )
    algorithm_end = effective_task_end or max(effective_start, duration - 0.3)
    sample_start = min(duration, effective_start + 1.0 / config.sample_fps)

    every_second_dir = output_dir / "sampled"
    every_second_dir.mkdir(parents=True, exist_ok=True)
    if algorithm_end > sample_start:
        try:
            subprocess.run(
                [
                    resolve_ffmpeg_executable(),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{sample_start:.3f}",
                    "-i",
                    str(video),
                    "-t",
                    f"{algorithm_end - sample_start:.3f}",
                    "-vf",
                    (
                        f"fps={config.sample_fps:g},"
                        f"scale='min({config.max_edge},iw)':-2"
                    ),
                    "-q:v",
                    "3",
                    str(every_second_dir / "frame_%05d.jpg"),
                ],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass

    interval = 1.0 / config.sample_fps
    candidates = [
        _Candidate(
            sample_start + (index - 1) * interval,
            path,
            f"{config.sample_fps:g}fps",
        )
        for index, path in enumerate(
            sorted(every_second_dir.glob("frame_*.jpg")),
            start=1,
        )
    ]

    # 受保护采样：在开头 [start, start+begin_window] 和结尾 [end-end_window, end]
    # 两个窗口内，每隔 protected_sample_interval 秒将对应帧标记为 protected，
    # 使其在后续各阶段优先保留，仅当画面高度一致时才去重
    # （用于捕获弹窗/横幅/开场状态等开头结尾的小变化）。
    fps_source = f"{config.sample_fps:g}fps"
    ps_interval = config.protected_sample_interval
    if ps_interval > 0 and (config.protected_begin_window > 0 or config.protected_end_window > 0):
        protected_begin_end = effective_start + config.protected_begin_window
        protected_end_start = algorithm_end - config.protected_end_window
        for c in candidates:
            if c.source != fps_source:
                continue
            protected = False
            # 开头窗口：[effective_start, effective_start + protected_begin_window]
            # 帧时间相对起点的偏移能被 interval 整除则保留
            if config.protected_begin_window > 0 and effective_start <= c.time <= protected_begin_end:
                offset = (c.time - effective_start) % ps_interval
                if min(offset, ps_interval - offset) <= interval / 2 + 1e-6:
                    protected = True
            # 结尾窗口：[algorithm_end - protected_end_window, algorithm_end]
            # 帧时间相对终点的偏移能被 interval 整除则保留
            if not protected and config.protected_end_window > 0 and protected_end_start <= c.time <= algorithm_end:
                offset = (algorithm_end - c.time) % ps_interval
                if min(offset, ps_interval - offset) <= interval / 2 + 1e-6:
                    protected = True
            if protected:
                c.source = f"protected-{config.protected_sample_interval:g}s"

    start_path = output_dir / "start" / f"start_{effective_start:.3f}.jpg"
    if _extract_at(video, effective_start, start_path, max_edge=config.max_edge):
        candidates.append(
            _Candidate(
                effective_start,
                start_path,
                f"start-{effective_start:.1f}s",
            )
        )

    scene_times = _dedup(
        scene_change_times(
            video,
            config.scene_threshold,
            end_time=algorithm_end,
        ),
        config.scene_min_gap_s,
    )
    scene_images = _extract_many_at(
        video, [t for t in scene_times if effective_start <= t <= algorithm_end],
        output_dir / "scene_batch", max_edge=config.max_edge,
    )
    for index, timestamp in enumerate(scene_times, start=1):
        if not effective_start <= timestamp <= algorithm_end:
            continue
        path = output_dir / "scenes" / f"scene_{index:04d}_{timestamp:.3f}.jpg"
        if timestamp in scene_images:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scene_images[timestamp], path)
        elif not _extract_at(video, timestamp, path, max_edge=config.max_edge):
            continue
        if path.is_file():
            candidates.append(
                _Candidate(
                    timestamp,
                    path,
                    f"scene-{config.scene_threshold:g}",
                )
            )

    if effective_task_end is not None:
        task_end_path = (
            output_dir / "task_end" / f"task_end_{effective_task_end:.3f}.jpg"
        )
        if _extract_at(
            video,
            effective_task_end,
            task_end_path,
            max_edge=config.max_edge,
        ):
            candidates.append(
                _Candidate(effective_task_end, task_end_path, "task-end")
            )

    # 显式时间窗可能只是多轮录屏中的一轮，不能混入整段视频的末帧。
    # 未提供结束点时，末帧仍用于自动结束点推断和最终状态兜底。
    if config.task_end_time is None:
        for backoff in (0.3, 0.6, 1.0, 1.5, 2.0):
            timestamp = max(0.0, duration - backoff)
            terminal_path = (
                output_dir / "terminal" / f"terminal_{timestamp:.3f}.jpg"
            )
            if _extract_at(
                video,
                timestamp,
                terminal_path,
                max_edge=config.max_edge,
            ):
                candidates.append(
                    _Candidate(
                        timestamp,
                        terminal_path,
                        f"terminal-{backoff:.1f}s",
                    )
                )
                break

    candidates.sort(
        key=lambda item: (
            item.time,
            0 if item.source.startswith("scene") else 1,
        )
    )
    return effective_task_end, candidates


def _canonicalize_timestamps(
    candidates: list[_Candidate],
    gap: float,
    *,
    config: KeyframeConfig | None = None,
) -> list[_Candidate]:
    """只合并时间接近且画面相似的候选；每组时间范围有界。"""
    if not candidates:
        return []
    config = config or KeyframeConfig()
    signatures = {}

    def signature(candidate):
        if id(candidate) not in signatures:
            signatures[id(candidate)] = (candidate.time, _content_signature(candidate.path))
        return signatures[id(candidate)][1]

    def priority(candidate):
        if candidate.source.startswith("start"):
            return (0, candidate.time)
        if candidate.source == "task-end":
            return (1, candidate.time)
        if candidate.source.startswith("protected"):
            return (2, candidate.time)
        if candidate.source.startswith("scene"):
            return (3, candidate.time)
        return (4, candidate.time)

    def endpoint(candidate):
        return candidate.source.startswith("start") or candidate.source == "task-end"

    groups = []
    comparisons = {}
    for candidate in sorted(candidates, key=lambda c: c.time):
        for identity, (timestamp, _) in list(signatures.items()):
            if candidate.time - timestamp > gap + 1e-9:
                del signatures[identity]
        checks = []
        merged = False
        for group in reversed(groups):
            if candidate.time - group[0].time > gap + 1e-9:
                break
            winner = min(group, key=priority)
            if endpoint(candidate) and any(endpoint(c) for c in group):
                continue
            rms, fraction = _content_difference(signature(candidate), signature(winner))
            checks.append((winner, rms, fraction))
            if (rms < config.final_dedup_rms_threshold
                    and fraction < config.final_dedup_changed_fraction_threshold):
                group.append(candidate)
                merged = True
                break
        if not merged:
            groups.append([candidate])
        comparisons[id(candidate)] = checks

    result = []
    for group in groups:
        winner = min(group, key=priority)
        result.append(winner)
        for candidate in group:
            checks = comparisons[id(candidate)]
            related = [c for c in group if c is not candidate]
            related.extend(c for c, _, _ in checks if c is not candidate and c not in related)
            _note(candidate,
                  ("时间接近但没有相同画面，单独保留" if len(group) == 1 else "相近且相似的画面中保留这一张")
                  if candidate is winner else "相近时刻画面相似，合并到同组代表图",
                  related, merge_gap_s=gap, group_start=group[0].time, group_end=group[-1].time,
                  rms_threshold=config.final_dedup_rms_threshold,
                  changed_fraction_threshold=config.final_dedup_changed_fraction_threshold,
                  comparisons=[{"time": c.time, "rms": rms, "changed_fraction": fraction}
                               for c, rms, fraction in checks])
    return sorted(result, key=lambda c: c.time)


def _build_layout_runs(
    candidates: list[_Candidate],
    threshold: float,
) -> tuple[list[np.ndarray], list[list[int]]]:
    if not candidates:
        return [], []
    _prewarm_features(candidates, _layout_signature)
    layouts = [_layout_signature(candidate.path) for candidate in candidates]
    runs: list[list[int]] = [[0]]
    for index in range(1, len(candidates)):
        if _layout_distance(layouts[index - 1], layouts[index]) >= threshold:
            runs.append([index])
        else:
            runs[-1].append(index)
    return layouts, runs


def _infer_task_end_candidate(
    candidates: list[_Candidate],
    terminal: _Candidate | None,
    config: KeyframeConfig,
) -> tuple[_Candidate | None, float | None]:
    if len(candidates) < 2 or terminal is None:
        return None, None
    layouts, runs = _build_layout_runs(
        candidates,
        config.state_layout_threshold,
    )
    if len(runs) < 2:
        return None, None

    final_shell = _assistant_shell_signature(terminal.path)
    start_shell = _assistant_shell_signature(candidates[0].path)
    if (
        _layout_distance(start_shell, final_shell)
        > config.auto_task_end_max_return_shell_distance
    ):
        return None, None

    final_run = runs[-1]
    final_run_duration = (
        candidates[final_run[-1]].time - candidates[final_run[0]].time
    )
    final_layout = _layout_signature(terminal.path)
    if (
        final_run_duration
        < config.auto_task_end_min_final_stable_duration_s
        or _layout_distance(layouts[final_run[-1]], final_layout)
        >= config.state_layout_threshold
    ):
        return None, None

    eligible: list[tuple[float, float, list[int]]] = []
    for run in runs[:-1]:
        run_duration = candidates[run[-1]].time - candidates[run[0]].time
        if run_duration < config.auto_task_end_min_stable_duration_s:
            continue
        task_shell = _assistant_shell_signature(candidates[run[-1]].path)
        shell_distance = _layout_distance(task_shell, final_shell)
        if shell_distance <= config.auto_task_end_min_task_shell_distance:
            continue
        layout_distance = _layout_distance(layouts[run[-1]], final_layout)
        stable_score = min(1.0, run_duration / 8.0)
        layout_score = min(
            1.0,
            max(0.0, (layout_distance - 0.04) / 0.08),
        )
        shell_score = min(
            1.0,
            max(0.0, (shell_distance - 0.02) / 0.08),
        )
        confidence = (
            0.40 * stable_score
            + 0.20 * layout_score
            + 0.15 * shell_score
            + 0.15 * min(1.0, final_run_duration / 3.0)
            + 0.10
        )
        if confidence >= config.auto_task_end_confidence_threshold:
            eligible.append((run_duration, confidence, run))

    if not eligible:
        return None, None
    _, confidence, best_run = max(
        eligible,
        key=lambda item: (item[0], item[1], candidates[item[2][-1]].time),
    )
    return candidates[best_run[-1]], confidence


def _tile_difference(left, right):
    delta = np.max(np.abs(left.astype(np.float32) - right.astype(np.float32)), axis=-1) / 255.0
    # 8×6 个局部区域；同页小卡片变化不会被整张页面的平均值冲淡。
    return float(delta.reshape(8, 20, 6, 16).mean(axis=(1, 3)).max())


def _descriptor(pixels):
    return pixels.reshape(20, 8, 12, 8, 3).mean(axis=(1, 3)).astype(np.float32) / 255.0


def _content_groups(candidates, config):
    """按实际内容归组，允许少量像素差异，但检查局部变化。"""
    @lru_cache(maxsize=64)
    def pixels(path):
        return _content_signature(Path(path))

    groups = []
    descriptors = np.empty((len(candidates), 20, 12, 3), dtype=np.float32)
    coarse_descriptors = np.empty((len(candidates), 48), dtype=np.float32)
    for candidate in candidates:
        image = pixels(str(candidate.path))
        descriptor = _descriptor(image)
        found = None
        # 批量做廉价预筛；临界值仍按原标量公式复核，保持组顺序与决定不变。
        coarse = descriptor.reshape(4, 5, 4, 3, 3).mean(axis=(1, 3)).ravel()
        coarse_squared = np.mean((coarse_descriptors[:len(groups)] - coarse) ** 2, axis=1)
        possible = np.flatnonzero(coarse_squared <= (config.content_rms_threshold + 5e-7) ** 2)
        squared = np.mean((descriptors[possible] - descriptor) ** 2, axis=(1, 2, 3))
        possible = possible[squared <= (config.content_rms_threshold + 1e-7) ** 2]
        for index in reversed(possible):
            group = groups[int(index)]
            if float(np.sqrt(np.mean((descriptor - group["descriptor"]) ** 2))) > config.content_rms_threshold:
                continue
            other = pixels(str(group["reference"].path))
            rms, fraction = _content_difference(image, other)
            if (rms < config.content_rms_threshold and fraction < config.content_changed_fraction
                    and _tile_difference(image, other) < config.content_tile_threshold):
                found = group
                break
        if found is None:
            descriptors[len(groups)] = descriptor
            coarse_descriptors[len(groups)] = coarse
            groups.append({"members": [candidate], "representative": candidate, "reference": candidate,
                           "descriptor": descriptor, "support": candidate.support_s})
        else:
            found["members"].append(candidate)
            found["support"] += candidate.support_s
            current = found["representative"]
            if (candidate.support_s, candidate.time) > (current.support_s, current.time):
                found["representative"] = candidate
    return groups


def _deduplicate_states(candidates, config):
    if not candidates:
        return []
    layouts, runs = _build_layout_runs(candidates, config.state_layout_threshold)
    pages = []
    page_layouts = np.empty((len(runs), *layouts[0].shape), dtype=np.float32)
    coarse_layouts = np.empty((len(runs), 16), dtype=np.float32)
    selected = []
    interval = 1.0 / config.sample_fps
    for index, candidate in enumerate(candidates):
        candidate.support_s = min(interval, max(0.0, candidates[index + 1].time - candidate.time)) if index + 1 < len(candidates) else interval
    for run in runs:
        layout = layouts[run[0]]
        coarse = layout.reshape(4, 6, 4, 4).mean(axis=(1, 3)).ravel()
        coarse_distances = np.mean((coarse_layouts[:len(pages)] - coarse) ** 2, axis=1)
        possible = np.flatnonzero(coarse_distances <= (255 * config.state_layout_threshold + 1e-4) ** 2)
        distances = np.mean((page_layouts[possible] - layout) ** 2, axis=(1, 2))
        possible = possible[distances <= (255 * config.state_layout_threshold + 1e-5) ** 2]
        page = next((int(i) for i in possible
                     if _layout_distance(layout, pages[int(i)]) < config.state_layout_threshold), None)
        if page is None:
            page = len(pages)
            page_layouts[page] = layout
            coarse_layouts[page] = coarse
            pages.append(layout)
        for index in run:
            candidates[index].page_id = page
        groups = _content_groups([candidates[index] for index in run], config)
        for group in groups:
            representative = group["representative"]
            representative.support_s = group["support"]
            # 首帧不会因选择更晚的重复图而消失；任务结束帧后续仍独立补回。
            anchors = [c for c in group["members"] if c is candidates[0]]
            kept = anchors or [representative]
            for candidate in group["members"]:
                if candidate in kept:
                    candidate.keep_reason = "content-state"
                    _note(candidate, "保留本页这一种内容的代表画面",
                          page=page + 1, group_size=len(group["members"]), support_s=group["support"])
                else:
                    _note(candidate, "同页内容相近，已由另一张代表画面覆盖", kept,
                          page=page + 1, group_size=len(group["members"]))
            selected.extend(kept)
    return sorted(selected, key=lambda c: c.time)


def _merge_repeated_content(candidates, config):
    """分配数量前先合并重复内容；首尾位置仍保留。"""
    if not candidates:
        return []
    selected = []
    for group in _content_groups(candidates, config):
        anchors = [c for c in group["members"] if c is candidates[0] or c.keep_reason in {
            "task-end-explicit", "task-end-auto", "final-frame"}]
        kept = anchors or [group["representative"]]
        for candidate in group["members"]:
            if candidate in kept:
                candidate.support_s = group["support"]
                _note(candidate, "重复出现的相近内容只保留代表或首尾位置",
                      support_s=group["support"], occurrences=len(group["members"]))
            else:
                _note(candidate, "这份内容已在其他时刻保留，合并重复展示", kept,
                      occurrences=len(group["members"]))
        selected.extend(kept)
    return sorted(selected, key=lambda c: c.time)


@lru_cache(maxsize=1)
def _local_cv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("实验版局部检查需要 OpenCV，请在当前环境安装 requirements.txt") from exc
    # 平台本身并行两题；避免每次小图比较再启动一组原生工作线程。
    cv2.setNumThreads(1)
    return cv2


def _strict_region_equal(left, right, config):
    """同样的四项严格条件，应用于对齐后的局部内容，而非整页平均。"""
    cv = _local_cv()
    coarse_left = cv.resize(left, (8, 8), interpolation=cv.INTER_AREA).astype(np.float32)
    coarse_right = cv.resize(right, (8, 8), interpolation=cv.INTER_AREA).astype(np.float32)
    # 粗平均差异是后续 RMS 的下界；额外保留两级像素量化余量，
    # 明显不同的重叠区不用反复生成较大的比较数组。
    coarse_rms = float(np.sqrt(np.mean((coarse_left - coarse_right) ** 2)) / 255)
    if coarse_rms > config.global_dedup_rms_threshold + 2 / 255:
        return False
    left = cv.resize(left, (96, 160), interpolation=cv.INTER_AREA).astype(np.float32)
    right = cv.resize(right, (96, 160), interpolation=cv.INTER_AREA).astype(np.float32)
    delta = np.abs(left - right)
    if float(np.sqrt(np.mean(delta * delta)) / 255) >= config.global_dedup_rms_threshold:
        return False
    if float(np.mean(np.any(delta >= 12, axis=-1))) >= config.global_dedup_changed_fraction:
        return False
    maximum = delta.max(axis=-1) / 255
    return (float(maximum.reshape(8, 20, 6, 16).mean(axis=(1, 3)).max()) < config.global_dedup_tile_threshold
            and float(maximum.reshape(40, 4, 24, 4).mean(axis=(1, 3)).max()) < config.global_dedup_patch_threshold)


def _local_boxes(rgb):
    """颜色块和矩形轮廓只是关注区域线索，不是业务卡片识别。"""
    cv = _local_cv()
    height, width = rgb.shape[:2]
    corners = np.concatenate([rgb[:4, :4].reshape(-1, 3), rgb[:4, -4:].reshape(-1, 3),
                              rgb[-4:, :4].reshape(-1, 3), rgb[-4:, -4:].reshape(-1, 3)])
    background = np.median(corners, axis=0).astype(np.uint8)
    foreground = (np.max(np.abs(rgb.astype(np.int16) - background), axis=2) >= 18).astype(np.uint8)
    foreground = cv.morphologyEx(foreground, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv.connectedComponentsWithStats(foreground, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        if area / max(1, w * h) >= .55:
            boxes.append((int(x), int(y), int(w), int(h)))
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    contours, _ = cv.findContours(cv.Canny(gray, 60, 160), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if cv.contourArea(contour) >= .80 * w * h:
            boxes.append((x, y, w, h))
    accepted = []
    for box in sorted(set(boxes), key=lambda b: (-b[2] * b[3], b)):
        x, y, w, h = box
        area = w * h / (width * height)
        if w < .30 * width or h < .08 * height or not .035 <= area <= .93:
            continue
        if any(max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
               >= .8 * min(w * h, c * d) for a, b, c, d in accepted):
            continue
        accepted.append(box)
        if len(accepted) == 4:
            break
    return accepted, background


@dataclass
class _LocalTrack:
    index: int
    pixels: np.ndarray
    origin: int
    width_ratio: float
    kind: str
    last_index: int
    observations: list = field(default_factory=list)
    bins: list | None = None
    color: np.ndarray | None = None
    gray: np.ndarray | None = None
    reliable: bool = True
    search_data: object = None


@dataclass
class _LocalAnalysis:
    candidates: list
    representatives: list = field(default_factory=list)
    replacements: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    descriptors: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)


class _LocalSearchData:
    """一份不再改写的灰度图；纹理条及精确摘要按需计算。"""
    def __init__(self, gray):
        self.gray = gray
        self._templates = None
        self._digest = None
        self._lock = threading.Lock()

    def templates(self, work):
        with self._lock:
            if self._templates is not None:
                work.add('preparation_reuses')
                return self._templates
            sh = len(self.gray)
            height = min(24, max(8, sh // 3))
            energy = np.mean(np.abs(np.diff(self.gray.astype(np.float32), axis=1)), axis=1)
            textured = int(np.argmax(np.convolve(energy, np.ones(height), mode='valid')))
            self._templates = [(y, self.gray[y:y + height, 2:-2])
                               for y in sorted({0, max(0, (sh - height) // 2), sh - height, textured})
                               if not float(self.gray[y:y + height, 2:-2].std()) < 3]
            work.add('preparations')
            return self._templates

    def digest(self):
        with self._lock:
            if self._digest is None:
                self._digest = (self.gray.shape, hashlib.sha256(self.gray.tobytes()).digest())
            return self._digest


class _LocalWork:
    """仅限一次局部分析的计数、精确搜索缓存和可选线程池。"""
    def __init__(self, config):
        self.config = config
        self.counts = Counter()
        self.peaks = OrderedDict()
        self.lock = threading.Lock()
        self.pool = None

    def add(self, key, value=1):
        with self.lock:
            self.counts[key] += value

    def peak(self, source, target, y, patch):
        cv = _local_cv()
        key = (source.digest(), target.digest(), y, patch.shape) if self.config.local_search_reuse else None
        if key is not None:
            with self.lock:
                if key in self.peaks:
                    self.peaks.move_to_end(key)
                    self.counts['template_cache_hits'] += 1
                    return self.peaks[key]
        started = time.perf_counter()
        matches = cv.matchTemplate(target.gray, patch, cv.TM_CCOEFF_NORMED)
        _, score, _, (x, match_y) = cv.minMaxLoc(matches)
        self.add('template_search_s', time.perf_counter() - started)
        self.add('template_calls')
        result = (score, x, match_y)
        if key is not None:
            with self.lock:
                self.peaks[key] = result
                self.peaks.move_to_end(key)
                if len(self.peaks) > 512:
                    self.peaks.popitem(last=False)
        return result

    def comparisons(self, tracks, crop, crop_gray, crop_data):
        """原顺序裁决。最近参考先串行，其余小批并行，整批结束才允许改写参考图。"""
        def locate(track):
            return _locate_local_content(track, crop, self.config, crop_gray, self, crop_data)
        if not tracks:
            return
        yield tracks[0], locate(tracks[0])
        workers = self.config.local_match_workers
        for start in range(1, len(tracks), workers):
            wave = tracks[start:start + workers]
            if workers == 1:
                results = [locate(wave[0])]
            else:
                if self.pool is None:
                    self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='local-match')
                # 等待整批任务结束；若一个失败，也在 wrapper 的 finally 中等待并清理。
                results = list(self.pool.map(locate, wave))
            yield from zip(wave, results)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.peaks.clear()


class _LocalLocation(tuple):
    def __new__(cls, position, confidence, method):
        result = super().__new__(cls, (position, confidence))
        result.method = method
        return result


def _locate_local_content(track, crop, config, crop_gray=None, work=None, crop_data=None):
    """用有纹理的小条匹配位移，再严格核对整个重叠区；不靠匹配分数合并文字。"""
    cv = _local_cv()
    work = work if work is not None else _LocalWork(config)
    work.add('pair_attempts')
    reference = track.pixels
    if reference.shape[1] != crop.shape[1]:
        work.add('failed_pairs')
        return None
    rh, ch = len(reference), len(crop)
    if config.local_fast_alignment and rh == ch and rh >= 8:
        work.add('original_position_checks')
        left, right = reference[:, 2:-2], crop[:, 2:-2]
        if (_strict_region_equal(left, right, config)
                and not (left.std() < 2 and (reference.std() >= 3 or crop.std() >= 3))):
            # 等高时，原位通过完整严格检查就是唯一最大重叠范围。
            # 不伪造未执行的模板搜索置信分数。
            work.add('original_position_hits')
            return _LocalLocation(track.origin, 0.0, 'original_position')
    work.add('full_search_attempts')
    possible = {track.origin, track.origin + rh - ch}
    scores = {position: 0.0 for position in possible}
    if track.gray is None:
        track.gray = cv.cvtColor(reference, cv.COLOR_RGB2GRAY)
    rg = track.gray
    cg = crop_gray if crop_gray is not None else cv.cvtColor(crop, cv.COLOR_RGB2GRAY)
    if config.local_search_reuse:
        if track.search_data is None:
            track.search_data = _LocalSearchData(rg)
        reference_data = track.search_data
        crop_data = crop_data if crop_data is not None else _LocalSearchData(cg)
    else:
        reference_data, crop_data = _LocalSearchData(rg), _LocalSearchData(cg)
    for source_data, target_data, reverse in ((crop_data, reference_data, False), (reference_data, crop_data, True)):
        source, target = source_data.gray, target_data.gray
        sh, th = len(source), len(target)
        strip_height = min(24, max(8, sh // 3))
        if strip_height > th:
            continue
        for y, patch in source_data.templates(work):
            score, x, match_y = work.peak(source_data, target_data, y, patch)
            if score < config.local_template_threshold or abs(x - 2) > 1:
                continue
            position = track.origin + (y - match_y if reverse else match_y - y)
            for adjustment in (-1, 0, 1):
                possible.add(position + adjustment)
                scores[position + adjustment] = score
    best = None
    for position in sorted(possible):
        top, bottom = max(track.origin, position), min(track.origin + rh, position + ch)
        overlap = bottom - top
        if overlap < min(rh, ch) * .40 or overlap < 8:
            continue
        left = reference[top - track.origin:bottom - track.origin, 2:-2]
        right = crop[top - position:bottom - position, 2:-2]
        if not _strict_region_equal(left, right, config):
            continue
        # 两幅有内容的图不能仅靠一段白底建立关系。
        if left.std() < 2 and (reference.std() >= 3 or crop.std() >= 3):
            continue
        value = (overlap, scores.get(position, 0), -abs(position - track.origin))
        if best is None or value > best[0]:
            best = (value, position, scores.get(position, 0))
    if best is None:
        work.add('failed_pairs')
        return None
    return _LocalLocation(best[1], best[2], 'template_search')


def _extend_local_track(track, crop, position):
    """只扩展观察到的新范围；既有参考像素不被后续近似图覆盖，避免漂移。"""
    top = min(track.origin, position)
    bottom = max(track.origin + len(track.pixels), position + len(crop))
    if bottom - top > 4096:
        return False
    if top == track.origin and bottom == track.origin + len(track.pixels):
        return True
    combined = np.empty((bottom - top, track.pixels.shape[1], 3), dtype=np.uint8)
    combined[position - top:position - top + len(crop)] = crop
    combined[track.origin - top:track.origin - top + len(track.pixels)] = track.pixels
    track.origin, track.pixels = top, combined
    track.gray = None
    track.search_data = None
    track.color = np.median(combined, axis=(0, 1))
    return True


def _selection_anchors(candidates, limit):
    if not candidates:
        return []
    endings = sorted([c for c in candidates[1:] if c.keep_reason in {
        'task-end-explicit', 'task-end-auto', 'final-frame'}],
        key=lambda c: (c.keep_reason != 'final-frame', c.time), reverse=True)
    return [candidates[0], *endings[:max(0, limit - 1)]]


def _local_track_bins(track, config):
    if track.bins is not None:
        return track.bins
    cv = _local_cv()
    pixels = track.pixels
    edges = cv.Canny(cv.cvtColor(pixels, cv.COLOR_RGB2GRAY), 50, 150)
    base = np.median(pixels.reshape(-1, 3), axis=0)
    colored = np.max(np.abs(pixels.astype(np.float32) - base), axis=2) >= 16
    signal = (np.count_nonzero(edges, axis=1) >= 2) | (colored.mean(axis=1) >= .02)
    bins, size = [], config.local_content_bin_height
    for number in range(math.floor(track.origin / size), math.ceil((track.origin + len(pixels)) / size)):
        top, bottom = max(track.origin, number * size), min(track.origin + len(pixels), (number + 1) * size)
        rows = np.flatnonzero(signal[top - track.origin:bottom - track.origin]) + top
        if len(rows):
            bins.append((number, rows))
    return bins or [(0, np.array([track.origin + len(pixels) // 2]))]


def _local_views(candidates, config, fingerprints):
    """最多预读八张，顺序消费结果，避免同时持有整段视频的原尺寸图。"""
    from PIL import Image
    cv = _local_cv()
    identities = dict(_FEATURE_KEYS.get() or {})
    def load(candidate):
        digest = identities.get(str(candidate.path))
        payload = None
        if not isinstance(digest, bytes):
            payload = candidate.path.read_bytes()
            digest = hashlib.sha256(payload).digest()
        if digest in fingerprints:
            return candidate, digest, None, None, None, None
        with Image.open(io.BytesIO(payload) if payload is not None else candidate.path) as image:
            source = np.asarray(_masked_analysis_image(image.convert('RGB'), config.analysis_mask_box))
        height, width = source.shape[:2]
        scale = min(1.0, config.local_analysis_max_edge / max(height, width))
        small = cv.resize(source, (max(8, round(width * scale)), max(8, round(height * scale))),
                          interpolation=cv.INTER_AREA)
        boxes, background = _local_boxes(small)
        return candidate, digest, source, small, boxes, background
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='frame-local-read') as workers:
        iterator, pending = iter(candidates), deque()
        for _ in range(8):
            candidate = next(iterator, None)
            if candidate is None:
                break
            pending.append(workers.submit(load, candidate))
        while pending:
            yield pending.popleft().result()
            candidate = next(iterator, None)
            if candidate is not None:
                pending.append(workers.submit(load, candidate))


def _analyze_local_content(candidates, config):
    work = _LocalWork(config)
    started = time.perf_counter()
    try:
        analysis = _analyze_local_content_impl(candidates, config, work)
        return analysis
    finally:
        work.close()
        performance = {key: int(work.counts[key]) for key in (
            'pair_attempts', 'original_position_checks', 'original_position_hits',
            'full_search_attempts', 'template_calls', 'template_cache_hits',
            'preparations', 'preparation_reuses', 'failed_pairs', 'accepted_matches')}
        performance.update(input_frames=len(candidates),
                           elapsed_s=round(time.perf_counter() - started, 4),
                           template_search_work_s=round(work.counts['template_search_s'], 4),
                           match_workers=config.local_match_workers)
        if 'analysis' in locals():
            analysis.performance = performance
        recorder = _PROCESS.get()
        if recorder is not None:
            recorder.local_analysis = performance


def _analyze_local_content_impl(candidates, config, work):
    """仅分析名额分配前的候选；计算过程与实验台记录完全独立。"""
    cv = _local_cv()
    analysis = _LocalAnalysis(list(candidates))
    tracks, backgrounds, fingerprints, background_keys = [], [], {}, {}
    background_descriptors = np.empty((len(candidates), 48), dtype=np.float32)
    for index, (candidate, digest, source, small, boxes, background) in enumerate(_local_views(candidates, config, fingerprints)):
        identity = id(candidate)
        candidate.evidence_ids = ()
        analysis.coverage[identity] = set()
        if digest in fingerprints:
            previous = fingerprints[digest]
            analysis.details[identity] = {**analysis.details[id(previous)], 'exact_repeat_of': previous.time}
            analysis.quality[identity] = analysis.quality[id(previous)]
            analysis.descriptors[identity] = analysis.descriptors[id(previous)]
            for track in tracks:
                old = next((o for o in track.observations if o['candidate'] is previous), None)
                if old is not None:
                    track.observations.append({**old, 'candidate': candidate})
            background_id = analysis.details[identity]['background_id']
            analysis.coverage[identity].add(background_id)
            continue
        fingerprints[digest] = candidate
        height, width = source.shape[:2]
        scale = min(1.0, config.local_analysis_max_edge / max(height, width))
        fallback = not boxes
        if fallback:
            boxes = [(0, 0, small.shape[1], small.shape[0])]
        masked = small.copy()
        regions, exposures, clipped_edges, sharpness, motion = [], [], [], [], []
        for x, y, w, h in boxes:
            x0, y0 = round(x / scale), round(y / scale)
            x1, y1 = min(width, round((x + w) / scale)), min(height, round((y + h) / scale))
            source_crop = source[y0:y1, x0:x1]
            # 局部卡片保留原图采样尺度。分别按可见高度缩放会改变字形，
            # 使同一卡片的半张/全张无法通过严格重叠检查。
            if fallback:
                crop = source_crop[2:-2].copy() if source_crop.shape[1] <= 320 else small[2:-2].copy()
            else:
                crop = source_crop[2:-2].copy()
            kind = 'scene' if fallback else 'local'
            ratio = w / small.shape[1]
            matched = None
            crop_color = np.median(crop, axis=(0, 1))
            crop_gray = cv.cvtColor(crop, cv.COLOR_RGB2GRAY)
            crop_data = _LocalSearchData(crop_gray)
            active = [track for track in tracks if track.pixels is not None and track.kind == kind]
            # 位移跟踪只检查时间上邻近的内容；跨时间的完全重复由文件标记复用。
            # 整页逐一匹配几十个旧页面既不能解释操作，还会使动态视频耗时失控。
            recent = sorted(active, key=lambda t: (-t.last_index, t.index))[:1 if fallback else 8]
            eligible = [track for track in recent if .85 <= ratio / track.width_ratio <= 1.18
                        and not (fallback and np.abs(crop_color - track.color).max() > 40)]
            for track, location in work.comparisons(eligible, crop, crop_gray, crop_data):
                if location is not None and _extend_local_track(track, crop, location[0]):
                    matched = track, location
                    work.add('accepted_matches')
                    break
            if matched is None:
                track = _LocalTrack(len(tracks), crop.copy(), 0, ratio, kind, index)
                track.color, track.gray = crop_color, crop_gray
                if fallback and float(crop_gray.std()) < 2:
                    track.reliable = not any(old.gray is not None and float(old.gray.std()) < 2
                                             and np.abs(old.color - crop_color).max() <= 8 for old in active)
                tracks.append(track)
                position, confidence = 0, 0.0
                match_method = 'new_reference'
            else:
                track, location = matched
                position, confidence = location
                match_method = location.method
            track.last_index = index
            track.observations.append({'candidate': candidate, 'top': position, 'bottom': position + len(crop)})
            clipped = 0 if fallback else sum((x <= 1, y <= 1, x + w >= small.shape[1] - 1,
                                              y + h >= small.shape[0] - 1))
            clarity = float(cv.Laplacian(crop_gray, cv.CV_32F).var())
            exposures.append(len(crop))
            clipped_edges.append(clipped)
            sharpness.append(clarity)
            motion.append(abs(position))
            regions.append({'track': track.index, 'kind': kind, 'box': [x0, y0, x1 - x0, y1 - y0],
                            'possible_clipping': bool(clipped),
                            'match_confidence': None if match_method == 'original_position' else round(float(confidence), 4),
                            'match_method': match_method,
                            'visible_range': [position, position + len(crop)]})
            masked[y:y + h, x:x + w] = background
        signature = cv.resize(masked, (96, 160), interpolation=cv.INTER_AREA)
        background_key = hashlib.sha256(signature.tobytes()).digest()
        background_index = background_keys.get(background_key)
        descriptor = cv.resize(signature, (4, 4), interpolation=cv.INTER_AREA).astype(np.float32).ravel() / 255
        if background_index is None and backgrounds:
            distances = np.mean((background_descriptors[:len(backgrounds)] - descriptor) ** 2, axis=1)
            possible = np.flatnonzero(distances <= (config.global_dedup_rms_threshold + 2 / 255) ** 2)
            background_index = next((int(i) for i in possible
                                     if _strict_region_equal(signature, backgrounds[i], config)), None)
        if background_index is None:
            background_index = len(backgrounds)
            uniform = bool(np.all(signature == signature[0, 0]))
            weak_background = uniform and any(ref.shape[:2] == (1, 1)
                                               and np.abs(ref[0, 0].astype(np.int16) - signature[0, 0]).max() <= 8
                                               for ref in backgrounds)
            backgrounds.append(signature[:1, :1].copy() if np.all(signature == signature[0, 0]) else signature)
            background_descriptors[background_index] = descriptor
            key = f'b{background_index:04d}'
            analysis.units[key] = {'id': key, 'kind': 'background', 'weight': 0.0 if weak_background else .15}
        background_keys[background_key] = background_index
        background_id = f'b{background_index:04d}'
        analysis.units.setdefault(background_id, {'id': background_id, 'kind': 'background', 'weight': .15})
        analysis.coverage[identity].add(background_id)
        analysis.quality[identity] = (sum(exposures), -sum(clipped_edges),
                                      round(sum(sharpness) / max(1, len(sharpness)), 4), -sum(motion))
        analysis.descriptors[identity] = cv.resize(small, (12, 20), interpolation=cv.INTER_AREA).astype(np.float32) / 255
        analysis.details[identity] = {'regions': regions, 'background_id': background_id,
                                      'note': '局部区域和裁切线索，不代表已识别弹窗用途或确认完整性'}
        # 活跃跟踪最多保存 32 份参考图；更早的内容只留覆盖区间和记录。
        active = sorted((track for track in tracks if track.pixels is not None),
                        key=lambda t: (t.last_index, t.index))
        for old in active[:-32]:
            old.bins = _local_track_bins(old, config)
            old.pixels = None
            old.gray = None
            old.search_data = None
    for track in tracks:
        bins = _local_track_bins(track, config)
        for number, rows in bins:
            key = f'r{track.index:04d}:{number}'
            analysis.units[key] = {'id': key, 'kind': track.kind, 'track': track.index,
                                   'weight': (1.5 if track.kind == 'local' else 1.0) / len(bins) if track.reliable else 0.0,
                                   'range': [int(rows.min()), int(rows.max()) + 1]}
            for observation in track.observations:
                if float(np.mean((rows >= observation['top']) & (rows < observation['bottom']))) >= .8:
                    analysis.coverage[id(observation['candidate'])].add(key)
    anchors = {id(c) for c in _selection_anchors(candidates, 15)}
    def rank(candidate):
        coverage = analysis.coverage[id(candidate)]
        return (sum(analysis.units[key]['weight'] for key in coverage), len(coverage),
                *analysis.quality[id(candidate)], candidate.support_s, candidate.time)
    ordered = sorted(candidates, key=rank, reverse=True)
    # 固定参考的直接覆盖关系，不沿被删除图片串联合并；首尾始终保留候选资格。
    for candidate in ordered:
        identity = id(candidate)
        covering = next((rep for rep in analysis.representatives
                         if analysis.coverage[identity] <= analysis.coverage[id(rep)]), None)
        if identity not in anchors and covering is not None:
            analysis.replacements[identity] = covering
        else:
            analysis.representatives.append(candidate)
    analysis.representatives.sort(key=lambda c: c.time)
    for candidate in candidates:
        identity = id(candidate)
        candidate.evidence_ids = tuple(sorted(analysis.coverage[identity]))
        details = {**analysis.details[identity], 'coverage_ids': list(candidate.evidence_ids)}
        if identity in analysis.replacements:
            representative = analysis.replacements[identity]
            _note(candidate, '同一内容换用展示更充分或更清楚的代表图', [representative], **details)
        else:
            _note(candidate, '保留独立内容或互补范围的候选，尚未占用最终名额', **details)
    return analysis


def _limit_candidates_without_scroll(candidates, limit, *, config=None, analysis=None):
    """按尚未覆盖的内容选图，内容覆盖完即停；上限不是目标张数。"""
    config = config or KeyframeConfig()
    limit = min(15, limit)
    analysis = analysis or _analyze_local_content(candidates, config)
    selected = _selection_anchors(list(candidates), limit)
    selected_ids = {id(c) for c in selected}
    covered = set().union(*(analysis.coverage[id(c)] for c in selected)) if selected else set()
    eligible = set().union(*(analysis.coverage[id(c)] for c in analysis.representatives))
    required = {key for key in eligible if analysis.units[key]['weight'] > 0}
    rounds = {}
    while len(selected) < limit:
        remaining = [c for c in analysis.representatives if id(c) not in selected_ids]
        def gain(candidate):
            return sum(analysis.units[key]['weight'] for key in analysis.coverage[id(candidate)] - covered)
        useful = [c for c in remaining if gain(c) > 1e-9]
        if not useful:
            break
        def rank(candidate):
            descriptor = analysis.descriptors[id(candidate)]
            novelty = min((float(np.mean((descriptor - analysis.descriptors[id(c)]) ** 2))
                           for c in selected), default=1.0)
            return (gain(candidate), *analysis.quality[id(candidate)], novelty,
                    candidate.support_s, candidate.time)
        best = max(useful, key=rank)
        rounds[id(best)] = {'round': len(selected) + 1, 'new_content_weight': round(gain(best), 6)}
        selected.append(best)
        selected_ids.add(id(best))
        covered.update(analysis.coverage[id(best)])
    uncovered = required - covered
    uncertain = {key for key in eligible if analysis.units[key]['weight'] == 0} - covered
    hit = bool(uncovered) and len(selected) >= limit
    reason = 'limit_reached' if hit else ('covered' if not uncovered and not uncertain else 'no_reliable_addition')
    analysis.summary = {'policy': 'adaptive-local-v1', 'limit': limit, 'stop_reason': reason, 'hit_limit': hit,
                        'input_count': len(analysis.candidates), 'representative_count': len(analysis.representatives),
                        'identified_content_count': len(required), 'covered_content_count': len(required & covered),
                        'uncovered_count': len(uncovered),
                        'uncertain_content_count': len(uncertain),
                        'uncovered_content': [analysis.units[key] for key in sorted(uncovered)],
                        'selected_count': len(selected)}
    for candidate in candidates:
        identity = id(candidate)
        missing = analysis.coverage[identity] - covered
        common = [c for c in selected if c is not candidate
                  and analysis.coverage[identity] & analysis.coverage[id(c)]]
        details = {'frame_limit': limit, 'coverage_ids': list(candidate.evidence_ids),
                   'uncovered_ids': sorted(missing), 'stop_reason': reason}
        if identity in selected_ids:
            _note(candidate, '保留任务首帧或结束位置' if identity not in rounds
                  else '覆盖尚未展示的局部内容或整体状态', **details, **rounds.get(identity, {}))
        elif identity in analysis.replacements:
            _note(candidate, '同一内容已换用展示更充分的代表图', [analysis.replacements[identity]], **details)
        elif not missing:
            _note(candidate, '此帧识别到的内容已由所选图片覆盖，不补图凑数', common, **details)
        elif hit:
            _note(candidate, f'达到 {limit} 张上限，此帧仍有内容未覆盖', selected[-1:],
                  last_selected_comparison=rounds.get(id(selected[-1]), {}),
                  new_content_weight=round(sum(analysis.units[key]['weight'] for key in missing), 6), **details)
        else:
            _note(candidate, '未发现可靠新增内容，停止增加图片', common, **details)
    return sorted(selected, key=lambda c: c.time)


@dataclass(frozen=True)
class _ScrollSettings:
    analysis_edge: int = 800
    min_width: int = 400
    min_height: int = 800
    border_width_fraction: float = 0.025
    status_height_fraction: float = 0.04
    bottom_edge_fraction: float = 0.012
    orb_features: int = 600
    match_ratio: float = 0.75
    motion_tolerance: float = 2.0
    minimum_motion_points: int = 8
    pixel_difference: int = 28
    contrast_threshold: int = 20
    residual_component_limit: int = 12
    residual_pixel_limit: int = 90
    local_structure_correlation: float = 0.94
    analysis_mask_box: tuple = (0.63, 0.005, 0.74, 0.04)
    agent_indicator_box: tuple = (0.32, 0.955, 0.7, 0.995)

def _scroll_touch_mask(rgb, neighbors):
    """Only a filled achromatic disc absent in BOTH temporal neighbors, on blank background."""
    cv = _local_cv()
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    (h, w) = gray.shape
    bright = ((gray >= 155) & (_scroll_channel_range(rgb) < 14)).astype(np.uint8)
    (n, labels, stats, centers) = cv.connectedComponentsWithStats(bright, 8)
    result = np.zeros_like(gray, np.uint8)
    circles = []
    for i in range(1, n):
        (x, y, bw, bh, area) = map(int, stats[i])
        (cx, cy) = centers[i]
        if not (40 <= area <= 1800 and 0.8 <= bw / max(1, bh) <= 1.25 and (area / (bw * bh) > 0.62)):
            continue
        radius = max(bw, bh) / 2
        if not 4 <= radius <= 25:
            continue
        if x < round(0.04 * w) or x + bw > round(0.96 * w) or y < round(0.1 * h) or (y + bh > round(0.9 * h)):
            continue
        margin = round(radius * 0.9) + 4
        (ax0, ay0) = (max(0, x - margin), max(0, y - margin))
        (ax1, ay1) = (min(w, x + bw + margin), min(h, y + bh + margin))
        (yy, xx) = np.mgrid[ay0:ay1, ax0:ax1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + 2) ** 2
        ring = (xx - cx) ** 2 + (yy - cy) ** 2 >= (radius + 3) ** 2
        patch = gray[ay0:ay1, ax0:ax1]
        if patch[ring].std() > 12 or patch[disk].mean() - patch[ring].mean() < 65:
            continue
        absent = []
        for other in neighbors:
            og = cv.cvtColor(other[ay0:ay1, ax0:ax1], cv.COLOR_RGB2GRAY)
            absent.append(float(og[disk].mean() - og[ring].mean()) < 25 and float(og.std()) < 16)
        if len(absent) != 2 or not all(absent):
            continue
        result[ay0:ay1, ax0:ax1][disk] = 255
        circles.append({'center': [round(float(cx), 1), round(float(cy), 1)], 'radius': round(radius, 1)})
    return (result, circles)

def _scroll_motions(a, b):
    cv = _local_cv()
    c = a.config
    if a.descriptors is None or b.descriptors is None:
        return ([], 0)
    pairs = cv.BFMatcher(cv.NORM_HAMMING).knnMatch(a.descriptors, b.descriptors, k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < c.match_ratio * p[1].distance]
    if not matches:
        return ([], 0)
    aa = np.float32([a.points[m.queryIdx].pt for m in matches])
    bb = np.float32([b.points[m.trainIdx].pt for m in matches])
    delta = bb - aa
    remaining = np.ones(len(delta), bool)
    models = []
    for _ in range(3):
        if remaining.sum() < c.minimum_motion_points:
            break
        options = np.flatnonzero(remaining)
        support = np.max(np.abs(delta[options, None, :] - delta[None, :, :]), axis=2) <= c.motion_tolerance
        chosen = options[int(np.argmax(support.sum(axis=1)))]
        members = np.max(np.abs(delta - delta[chosen]), axis=1) <= c.motion_tolerance
        shift = np.median(delta[members], axis=0)
        members = np.max(np.abs(delta - shift), axis=1) <= c.motion_tolerance
        remaining[members] = False
        if members.sum() < c.minimum_motion_points:
            continue
        span = np.ptp(aa[members], axis=0)
        if span.max() < 35:
            continue
        (dx, dy) = np.rint(shift / a.scale).astype(int)
        if max(abs(dx), abs(dy)) <= 3:
            continue
        if min(abs(dx), abs(dy)) > max(abs(dx), abs(dy)) * 0.15:
            continue
        models.append({'shift': [int(dx), int(dy)], 'points': int(members.sum()), 'span': span.tolist()})
    return (models, len(matches))

def _scroll_channel_range(rgb):
    cv = _local_cv()
    (r, g, b) = cv.split(rgb)
    return cv.subtract(cv.max(cv.max(r, g), b), cv.min(cv.min(r, g), b))

class _ScrollView:

    def __init__(self, engine, fid):
        cv = _local_cv()
        self.engine = engine
        self.fid = fid
        self.config = engine.config
        rgb = engine.raw(fid).copy()
        (h, w) = rgb.shape[:2]
        for box in [self.config.analysis_mask_box, self.config.agent_indicator_box]:
            if box is None:
                continue
            rgb[round(box[1] * h):round(box[3] * h), round(box[0] * w):round(box[2] * w)] = 0
        self.rgb = cv.GaussianBlur(rgb, (3, 3), 0.55)
        self.gray = cv.cvtColor(self.rgb, cv.COLOR_RGB2GRAY)
        self.scale = min(1.0, self.config.analysis_edge / max(h, w))

    def bounds(self):
        (h, w) = self.gray.shape
        c = self.config
        return (round(c.border_width_fraction * w), round(c.status_height_fraction * h), round(w * (1 - c.border_width_fraction)), round(h * (1 - c.bottom_edge_fraction)))

    def quality(self):
        if self.fid not in self.engine.qualities:
            (x0, y0, x1, y1) = self.bounds()
            self.engine.qualities[self.fid] = int(self.foreground[y0:y1, x0:x1].sum())
        return self.engine.qualities[self.fid]

    def __getattr__(self, key):
        cv = _local_cv()
        if key == 'foreground':
            self.engine.counts['foreground_build'] += 1
            value = cv.absdiff(self.gray, cv.GaussianBlur(self.gray, (0, 0), 2)) >= self.config.contrast_threshold
        elif key == 'lower':
            self.engine.counts['range_build'] += 1
            value = cv.erode(self.rgb, np.ones((3, 3), np.uint8))
        elif key == 'upper':
            self.engine.counts['range_build'] += 1
            value = cv.dilate(self.rgb, np.ones((3, 3), np.uint8))
        elif key == 'touch_mask':
            value = self.engine.touch(self.fid)
        elif key in ('points', 'descriptors'):
            if self.fid not in self.engine.features:
                small = cv.resize(self.gray, None, fx=self.scale, fy=self.scale, interpolation=cv.INTER_AREA)
                valid = np.zeros_like(small)
                (x0, y0, x1, y1) = self.bounds()
                valid[round(y0 * self.scale):round(y1 * self.scale), round(x0 * self.scale):round(x1 * self.scale)] = 255
                value = cv.ORB_create(nfeatures=self.config.orb_features, fastThreshold=12).detectAndCompute(small, valid)
                self.engine.features[self.fid] = value
                self.engine.counts['feature_build'] += 1
                if len(self.engine.features) > 64:
                    self.engine.features.popitem(last=False)
            self.engine.features.move_to_end(self.fid)
            (self.points, self.descriptors) = self.engine.features[self.fid]
            return getattr(self, key)
        else:
            raise AttributeError(key)
        setattr(self, key, value)
        return value

class _ScrollEngine:

    def __init__(self, frames, root, view_capacity=4, raw_capacity=6, config=None):
        self.frames = frames
        self.root = root
        self.config = config if config is not None else _ScrollSettings()
        self.lookup = {f['id']: f for f in frames}
        self.indices = {f['id']: i for (i, f) in enumerate(frames)}
        self.raw_cache = OrderedDict()
        self.views = OrderedDict()
        self.features = OrderedDict()
        self.touches = OrderedDict()
        self.qualities = {}
        self.pairs = {}
        self.counts = Counter()
        self.logs = []
        self.view_capacity = view_capacity
        self.raw_capacity = raw_capacity

    def raw(self, fid):
        from PIL import Image
        if fid not in self.raw_cache:
            with Image.open(self.root / self.lookup[fid]['image']) as im:
                self.raw_cache[fid] = np.array(im.convert('RGB'))
            self.counts['decode'] += 1
            if len(self.raw_cache) > self.raw_capacity:
                self.raw_cache.popitem(last=False)
        self.raw_cache.move_to_end(fid)
        return self.raw_cache[fid]

    def touch(self, fid):
        if fid not in self.touches:
            i = self.indices[fid]
            neighbors = [self.raw(self.frames[j]['id']) for j in [i - 1, i + 1] if 0 <= j < len(self.frames)]
            (mask, _) = _scroll_touch_mask(self.raw(fid), neighbors)
            self.touches[fid] = np.packbits(mask.ravel() > 0)
            self.counts['touch_build'] += 1
            if len(self.touches) > 64:
                self.touches.popitem(last=False)
        self.touches.move_to_end(fid)
        (h, w) = self.raw(fid).shape[:2]
        return np.unpackbits(self.touches[fid], count=h * w).reshape(h, w) * np.uint8(255)

    def view(self, fid):
        if fid not in self.views:
            self.views[fid] = _ScrollView(self, fid)
            self.counts['view_build'] += 1
            if len(self.views) > self.view_capacity:
                self.views.popitem(last=False)
        self.views.move_to_end(fid)
        return self.views[fid]

    def compare(self, a, b):
        key = (a, b)
        self.counts['compare_requests'] += 1
        if key not in self.pairs:
            self.counts['judge'] += 1
            self.pairs[key] = {'a': a, 'b': b, **_scroll_judge(self.view(a), self.view(b))}
            self.logs.append(self.pairs[key])
            arrays = []
            arrays.extend(self.raw_cache.values())
            arrays.extend(self.touches.values())
            arrays.extend((v for view in self.views.values() for v in view.__dict__.values() if isinstance(v, np.ndarray)))
            arrays.extend((value[1] for value in self.features.values() if value[1] is not None))
            unique = {id(a): a for a in arrays}
            self.counts['max_cached_array_bytes'] = max(self.counts['max_cached_array_bytes'], sum((a.nbytes for a in unique.values())))
        return self.pairs[key]


    def close(self):
        self.views.clear()
        self.raw_cache.clear()
        self.features.clear()
        self.touches.clear()
        self.qualities.clear()
        self.pairs.clear()
        self.logs.clear()

def _scroll_judge(a, b):
    cv = _local_cv()
    if a.rgb.shape != b.rgb.shape:
        return {'decision': 'keep', 'reason': '尺寸变化'}
    # Native-pixel tolerances were validated on phone recordings. Do not
    # erase small controls when a lower-resolution input has thinner strokes.
    height, width = a.rgb.shape[:2]
    if width < a.config.min_width or height < a.config.min_height:
        return {'decision': 'uncertain', 'reason': '候选图尺寸较小，未确认滑动重复'}
    c = a.config
    (x0, y0, x1, y1) = a.bounds()
    (h, w) = a.gray.shape
    (motion_models, match_count) = _scroll_motions(a, b)
    if match_count < 8:
        return {'decision': 'uncertain', 'reason': '缺少可靠的共同图案', 'feature_matches': match_count, 'terminated_before_native': True}
    moving = sorted(motion_models, key=lambda m: -m['points'])[:1]
    shifts = [(0, 0)] + [tuple(m['shift']) for m in moving]
    error = np.full((h, w), 255, np.uint8)
    foreground = np.zeros((h, w), bool)
    for (dx, dy) in shifts:
        (ax0, ax1) = (max(x0, x0 - dx), min(x1, x1 - dx))
        (ay0, ay1) = (max(y0, y0 - dy), min(y1, y1 - dy))
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        sa = np.s_[ay0:ay1, ax0:ax1]
        sb = np.s_[ay0 + dy:ay1 + dy, ax0 + dx:ax1 + dx]
        ar = a.rgb[sa]
        br = b.rgb[sb]
        delta = cv.max(cv.max(cv.subtract(a.lower[sa], br), cv.subtract(br, a.upper[sa])), cv.max(cv.subtract(b.lower[sb], ar), cv.subtract(ar, b.upper[sb])))
        channels = cv.split(delta)
        delta = cv.max(cv.max(channels[0], channels[1]), channels[2])
        delta[(a.touch_mask[sa] > 0) | (b.touch_mask[sb] > 0)] = 0
        error[sa] = np.minimum(error[sa], delta)
        foreground[sa] |= a.foreground[sa] | b.foreground[sb]
    valid = np.zeros((h, w), bool)
    valid[y0:y1, x0:x1] = True
    if moving:
        (dx, dy) = moving[0]['shift']
        if dy > 0:
            valid[max(y0, y1 - dy):y1] = False
        elif dy < 0:
            valid[y0:min(y1, y0 - dy)] = False
        if dx > 0:
            valid[:, max(x0, x1 - dx):x1] = False
        elif dx < 0:
            valid[:, x0:min(x1, x0 - dx)] = False
    changed = ((error >= c.pixel_difference) & foreground & valid).astype(np.uint8)
    (n, labels, stats, _) = cv.connectedComponentsWithStats(changed, 8)
    initial_pixels = int(changed.sum())
    explained_components = 0
    for i in sorted(range(1, n), key=lambda i: -stats[i, 4]):
        (x, y, bw, bh, area) = map(int, stats[i])
        if area < 3 or bw > 160 or bh > 100:
            if area >= c.residual_component_limit:
                return _scroll_early_failure(match_count, moving, area)
            continue
        (ax0, ay0) = (max(x0, x - 5), max(y0, y - 5))
        (ax1, ay1) = (min(x1, x + bw + 5), min(y1, y + bh + 5))
        patch = a.gray[ay0:ay1, ax0:ax1]
        if patch.size < 25 or patch.std() < 8:
            if area >= c.residual_component_limit:
                return _scroll_early_failure(match_count, moving, area)
            continue
        explained = False
        for (dx, dy) in shifts:
            (bx0, by0, bx1, by1) = (ax0 + dx, ay0 + dy, ax1 + dx, ay1 + dy)
            if bx0 < 0 or by0 < 0 or bx1 > w or (by1 > h):
                continue
            search = b.gray[max(0, by0 - 2):min(h, by1 + 2), max(0, bx0 - 2):min(w, bx1 + 2)]
            if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
                continue
            (_, score, _, loc) = cv.minMaxLoc(cv.matchTemplate(search, patch, cv.TM_CCOEFF_NORMED))
            if score < c.local_structure_correlation:
                continue
            xx = max(0, bx0 - 2) + loc[0]
            yy = max(0, by0 - 2) + loc[1]
            other = b.rgb[yy:yy + patch.shape[0], xx:xx + patch.shape[1]]
            original = a.rgb[ay0:ay1, ax0:ax1]
            if np.max(np.abs(original.mean(axis=(0, 1)) - other.mean(axis=(0, 1)))) > 18:
                continue
            explained = True
            break
        if explained:
            region = changed[y:y + bh, x:x + bw]
            region[labels[y:y + bh, x:x + bw] == i] = 0
            explained_components += 1
        if not explained and area >= c.residual_component_limit:
            return _scroll_early_failure(match_count, moving, area)
    (n, _, stats, _) = cv.connectedComponentsWithStats(changed, 8)
    largest = int(stats[1:n, cv.CC_STAT_AREA].max()) if n > 1 else 0
    pixels = int(changed.sum())
    detail = {'motion_models': moving, 'feature_matches': match_count, 'residual_pixels': pixels, 'largest_residual': largest, 'foreground_a': a.quality(), 'foreground_b': b.quality(), 'raw_residual_pixels': initial_pixels, 'structure_matched_components': explained_components, 'largest_boxes': [list(map(int, s)) for s in sorted(stats[1:], key=lambda s: -s[4])[:5]]}
    if match_count < 8:
        return {**detail, 'decision': 'uncertain', 'reason': '缺少可靠的共同图案'}
    if largest >= c.residual_component_limit or pixels > c.residual_pixel_limit:
        return {**detail, 'decision': 'keep', 'reason': '对齐后仍有前景变化'}
    return {**detail, 'decision': 'merge', 'reason': '主要图案对应，剩余变化低于试验门槛', 'representative': 'b' if b.quality() > a.quality() else 'a'}

def _scroll_early_failure(match_count, moving, area):
    return {'decision': 'keep' if match_count >= 8 else 'uncertain', 'reason': '对齐后仍有前景变化' if match_count >= 8 else '缺少可靠的共同图案', 'motion_models': moving, 'feature_matches': match_count, 'certified_residual_lower_bound': int(area), 'terminated_early': True}

def _choose_with_scroll(candidates, analysis, anchors, frame_ids, compare=None, *, limit=15, max_related=3, max_checks=None, routing='mixed'):
    limit = min(15, limit)
    order = {id(c): i for (i, c) in enumerate(candidates)}
    lookup = {id(c): c for c in candidates}
    selected = list(anchors[:limit])
    protected = {id(c) for c in selected}
    groups = {id(c): [id(c)] for c in selected}
    coverage = {key: set(value) for (key, value) in analysis.coverage.items()}
    weights = {key: unit['weight'] for (key, unit) in analysis.units.items()}
    blocked = set()
    events = []
    checks = 0
    replacement_checks = 0
    visited = 0
    max_checks = 3 * len(candidates) if max_checks is None else max_checks

    def total(keys):
        return round(math.fsum((weights[k] for k in sorted(keys))), 12)

    def covered():
        return set().union(*(coverage[id(c)] for c in selected)) if selected else set()

    def related(candidate, rep):
        (ca, cb) = (coverage[id(candidate)], coverage[id(rep)])
        shared = total((k for k in ca & cb if analysis.units[k]['kind'] != 'background'))
        distance = float(np.mean((analysis.descriptors[id(candidate)] - analysis.descriptors[id(rep)]) ** 2))
        return (shared, distance)

    def _scroll_judge(rep, candidate):
        nonlocal checks
        if checks >= max_checks:
            return None
        checks += 1
        return compare(frame_ids[id(rep)], frame_ids[id(candidate)])
    while len(selected) < limit:
        present = {id(c) for c in selected}
        seen = covered()
        remaining = [c for c in analysis.representatives if id(c) not in present | blocked]
        useful = [c for c in remaining if total(coverage[id(c)] - seen) > 1e-09]
        if not useful:
            break

        def rank(candidate):
            ident = id(candidate)
            novelty = min((float(np.mean((analysis.descriptors[ident] - analysis.descriptors[id(c)]) ** 2)) for c in selected), default=1.0)
            return (total(coverage[ident] - seen), *analysis.quality[ident], novelty, candidate.support_s, candidate.time, order[ident])
        candidate = max(useful, key=rank)
        cid = id(candidate)
        visited += 1
        event = {'candidate': frame_ids[cid], 'gain_before_check': total(coverage[cid] - seen), 'checks': []}
        matched = False
        if compare is not None and checks < max_checks:
            possible = []
            for rep in selected:
                (shared, distance) = related(candidate, rep)
                if shared > 0 or distance <= 0.04:
                    possible.append((rep, shared, distance))
            possible.sort(key=lambda item: (-item[1], item[2], abs(item[0].time - candidate.time), order[id(item[0])]))
            if routing == 'mixed':
                before = sorted((p for p in possible if p[0].time <= candidate.time), key=lambda p: (candidate.time - p[0].time, order[id(p[0])]))
                after = sorted((p for p in possible if p[0].time > candidate.time), key=lambda p: (p[0].time - candidate.time, order[id(p[0])]))
                priority = before[:1] + after[:1] + possible
                unique = []
                used = set()
                for p in priority:
                    if id(p[0]) not in used:
                        used.add(id(p[0]))
                        unique.append(p)
                possible = unique
            for (rep, shared, distance) in possible[:max_related]:
                result = _scroll_judge(rep, candidate)
                if result is None:
                    break
                rid = id(rep)
                event['checks'].append({'representative': frame_ids[rid], 'decision': result['decision']})
                if result['decision'] != 'merge':
                    continue
                winner = cid if result['representative'] == 'b' else rid
                if winner == cid and rid in protected:
                    event.setdefault('notes', []).append('首尾代表不可替换，保留本候选的参选机会')
                    continue
                if winner == cid:
                    verified = True
                    for member in groups[rid]:
                        if member == rid:
                            continue
                        check = _scroll_judge(lookup[member], candidate)
                        replacement_checks += int(check is not None)
                        if check is None or check['decision'] != 'merge':
                            verified = False
                            break
                    if not verified:
                        event.setdefault('notes', []).append('未完成直接替换核验，保留本候选的参选机会')
                        continue
                    selected[selected.index(rep)] = candidate
                    groups[cid] = groups.pop(rid) + [cid]
                    blocked.add(rid)
                    coverage[cid] |= coverage[rid]
                    event.update({'action': 'replace', 'previous': frame_ids[rid], 'representative': frame_ids[cid]})
                else:
                    blocked.add(cid)
                    groups[rid].append(cid)
                    coverage[rid] |= coverage[cid]
                    event.update({'action': 'merge', 'representative': frame_ids[rid]})
                matched = True
                break
        if not matched:
            selected.append(candidate)
            groups[cid] = [cid]
            event['action'] = 'select'
            if compare is not None and checks >= max_checks:
                event.setdefault('notes', []).append('检查预算用完，按原选图规则保留，未确认重复不删除')
        events.append(event)
    selected.sort(key=lambda c: (c.time, order[id(c)]))
    all_eligible = set().union(*(analysis.coverage[id(c)] for c in analysis.representatives))
    missing = {k for k in all_eligible - covered() if weights[k] > 0}
    reason = 'limit_reached' if len(selected) >= limit and missing else 'recorded_units_covered' if not missing else 'no_reliable_addition'
    result = {'selected_ids': [frame_ids[id(c)] for c in selected], 'selected_count': len(selected), 'visited_candidates': visited, 'checks': checks, 'replacement_checks': replacement_checks, 'check_budget': max_checks, 'budget_exhausted': checks >= max_checks, 'stop_reason': reason, 'uncovered_units': len(missing), 'events': events, 'groups': [{'representative': frame_ids[id(c)], 'members': [frame_ids[m] for m in groups[id(c)]]} for c in selected]}
    assert len(selected) <= limit and len(events) <= len(candidates)
    return result

def _limit_candidates(candidates, limit, *, config=None, analysis=None):
    """Allocate a slot only after the previous, lightweight scroll check."""
    config = config or KeyframeConfig()
    if not config.scroll_dedup_enabled:
        return _limit_candidates_without_scroll(candidates, limit, config=config, analysis=analysis)
    analysis = analysis or _analyze_local_content(candidates, config)
    pool = list(analysis.candidates)
    recorder = _PROCESS.get()
    frame_ids = {id(c): recorder.identify(c) if recorder is not None else f'c{i:05d}'
                 for i, c in enumerate(pool)}
    lookup = {frame_ids[id(c)]: c for c in pool}
    frames = [{'id': frame_ids[id(c)], 'image': c.path} for c in pool]
    settings = _ScrollSettings(analysis_edge=config.scroll_analysis_max_edge,
                               min_width=config.scroll_min_width, min_height=config.scroll_min_height,
                               analysis_mask_box=config.analysis_mask_box,
                               agent_indicator_box=config.scroll_agent_indicator_box)
    engine = _ScrollEngine(frames, Path(), view_capacity=16, raw_capacity=48, config=settings)
    try:
        result = _choose_with_scroll(pool, analysis, _selection_anchors(list(candidates), limit),
                                     frame_ids, engine.compare, limit=limit,
                                     max_related=config.scroll_max_related,
                                     max_checks=config.scroll_checks_per_candidate * len(pool))
        comparisons = list(engine.logs)
        counters = dict(engine.counts)
    finally:
        engine.close()
    selected = [lookup[fid] for fid in result['selected_ids']]
    eligible = set().union(*(analysis.coverage[id(c)] for c in analysis.representatives))
    required = {key for key in eligible if analysis.units[key]['weight'] > 0}
    representatives = {}
    for group in result['groups']:
        rep = lookup[group['representative']]
        analysis.coverage[id(rep)] = set().union(*(analysis.coverage[id(lookup[fid])]
                                                   for fid in group['members']))
        for fid in group['members']:
            representatives[fid] = rep
            if lookup[fid] is not rep:
                analysis.replacements[id(lookup[fid])] = rep
    covered = set().union(*(analysis.coverage[id(c)] for c in selected)) if selected else set()
    uncovered = required - covered
    uncertain = {key for key in eligible if analysis.units[key]['weight'] == 0} - covered
    hit = bool(uncovered) and len(selected) >= min(limit, 15)
    reason = 'limit_reached' if hit else 'covered' if not uncovered and not uncertain else 'no_reliable_addition'
    merged_count = sum(len(group['members']) - 1 for group in result['groups'])
    checked = {r[key] for r in comparisons for key in ('a', 'b')}
    analysis.summary = {
        'policy': 'adaptive-scroll-v1', 'limit': min(limit, 15), 'stop_reason': reason, 'hit_limit': hit,
        'input_count': len(pool), 'representative_count': len(analysis.representatives),
        'identified_content_count': len(required), 'covered_content_count': len(required & covered),
        'uncovered_count': len(uncovered), 'uncertain_content_count': len(uncertain),
        'uncovered_content': [analysis.units[key] for key in sorted(uncovered)], 'selected_count': len(selected),
        'scroll_dedup': {'enabled': True, 'checked_frames': len(checked),
                         'comparisons': counters.get('judge', 0), 'merged_frames': merged_count,
                         'representative_changes': sum(e['action'] == 'replace' for e in result['events']),
                         'check_budget': result['check_budget'], 'budget_exhausted': result['budget_exhausted'],
                         'cache_counts': counters},
    }
    events = {e['candidate']: e for e in result['events']}
    per_frame = {fid: [] for fid in lookup}
    for comparison in comparisons:
        per_frame[comparison['a']].append(comparison)
        per_frame[comparison['b']].append(comparison)
    selected_ids = {id(c) for c in selected}
    anchor_ids = {id(c) for c in _selection_anchors(list(candidates), limit)}
    for candidate in candidates:
        fid, identity = frame_ids[id(candidate)], id(candidate)
        candidate.evidence_ids = tuple(sorted(analysis.coverage[identity]))
        missing = analysis.coverage[identity] - covered
        event = events.get(fid)
        details = {'frame_limit': min(limit, 15), 'coverage_ids': list(candidate.evidence_ids),
                   'uncovered_ids': sorted(missing), 'stop_reason': reason,
                   'scroll_checked': fid in checked, 'scroll_comparisons': per_frame[fid]}
        if event:
            details['selection_attempt'] = event
        representative = representatives.get(fid)
        if identity in selected_ids:
            label = '保留任务首帧或结束位置' if identity in anchor_ids else '覆盖尚未展示的内容；入选前未确认滑动重复'
            if event and event['action'] == 'replace':
                label = '入选前确认重复，改用本帧作为代表并继续选图'
            _note(candidate, label, **details)
        elif representative is not None and representative is not candidate:
            _note(candidate, '入选前确认滑动或静止重复，由代表图覆盖，不占新名额', [representative], **details)
        elif identity in analysis.replacements:
            rep = analysis.replacements[identity]
            rep = representatives.get(frame_ids[id(rep)], rep)
            _note(candidate, '同一内容已有代表图', [rep], **details)
        elif not missing:
            _note(candidate, '识别到的内容已被覆盖，不补图凑数', **details)
        elif hit:
            _note(candidate, f'达到 {min(limit, 15)} 张上限，此帧仍有内容未覆盖', selected[-1:],
                  last_selected_comparison={'new_content_weight': result['events'][-1]['gain_before_check']}
                  if result['events'] else {},
                  new_content_weight=round(math.fsum(analysis.units[key]['weight'] for key in sorted(missing)), 6),
                  **details)
        else:
            _note(candidate, '未发现可靠新增内容，停止增加图片', **details)
    return selected


@_with_feature_cache
def _hybrid_keyframes(
    video: Path,
    work_dir: Path,
    config: KeyframeConfig,
    duration: float,
) -> tuple[list[_Candidate], float | None, float | None, dict]:
    effective_task_end, raw = _extract_candidates(
        video,
        work_dir,
        config,
        duration,
    )
    _prewarm_features(raw, _content_signature)
    _stage("collected", "最初收集", [], raw)
    terminal_candidates = [
        candidate
        for candidate in raw
        if candidate.source.startswith("terminal")
    ]
    algorithm_candidates = [
        candidate
        for candidate in raw
        if not candidate.source.startswith("terminal")
    ]
    all_timestamp_unique = _canonicalize_timestamps(
        algorithm_candidates,
        config.timestamp_merge_gap_s,
        config=config,
    )
    for candidate in terminal_candidates:
        _note(candidate, "视频末帧单独保留，暂不参与相近时刻合并")
    _stage("timestamps", "合并近时刻的重复画面", raw,
           all_timestamp_unique + terminal_candidates)

    task_end_candidate: _Candidate | None = None
    task_end_confidence: float | None = None
    if effective_task_end is not None:
        task_end_candidate = next(
            (
                candidate
                for candidate in all_timestamp_unique
                if candidate.source == "task-end"
            ),
            None,
        )
    else:
        task_end_candidate, task_end_confidence = _infer_task_end_candidate(
            all_timestamp_unique,
            terminal_candidates[-1] if terminal_candidates else None,
            config,
        )
        if task_end_candidate is not None:
            effective_task_end = task_end_candidate.time

    timestamp_unique = (
        [
            candidate
            for candidate in all_timestamp_unique
            if candidate.time <= effective_task_end + 0.001
        ]
        if effective_task_end is not None
        else all_timestamp_unique
    )
    for candidate in all_timestamp_unique:
        if candidate not in timestamp_unique:
            _note(candidate, "超过判断出的任务结束位置",
                  [task_end_candidate] if task_end_candidate else [],
                  effective_task_end=effective_task_end, confidence=task_end_confidence)
    for candidate in terminal_candidates:
        _note(candidate, "视频末帧作为最后状态兜底，暂不裁剪")
    _stage("range", "确定任务范围", all_timestamp_unique + terminal_candidates,
           timestamp_unique + terminal_candidates)
    state_unique = _deduplicate_states(timestamp_unique, config)
    for candidate in terminal_candidates:
        _note(candidate, "视频末帧单独保留，暂不参与页面合并")
    _stage("states", "合并重复页面", timestamp_unique + terminal_candidates,
           state_unique + terminal_candidates)
    before_forced = state_unique + terminal_candidates

    forced: list[_Candidate] = []
    if task_end_candidate is not None:
        task_end_candidate.keep_reason = (
            "task-end-explicit"
            if config.task_end_time is not None
            else "task-end-auto"
        )
        forced.append(task_end_candidate)
    if terminal_candidates:
        terminal = terminal_candidates[-1]
        terminal.keep_reason = "final-frame"
        forced.append(terminal)
        state_unique = [
            candidate
            for candidate in state_unique
            if candidate is task_end_candidate
            or abs(candidate.time - terminal.time) > 0.5
        ]

    combined = {
        id(candidate): candidate for candidate in state_unique + forced
    }
    combined_ordered = sorted(combined.values(), key=lambda candidate: candidate.time)
    for candidate in before_forced:
        if id(candidate) not in combined:
            _note(candidate, "距离视频末帧太近，改由末帧代表", terminal_candidates, gap_s=0.5)
    for candidate in forced:
        _note(candidate, "补回任务结束帧" if candidate is task_end_candidate else "保留视频末帧")
    _stage("endpoints", "保留任务首尾", before_forced, combined_ordered)
    content_unique = _merge_repeated_content(combined_ordered, config)
    _stage("content", "合并重复内容", combined_ordered, content_unique)
    analysis = _analyze_local_content(content_unique, config)
    _stage("local", "检查局部内容与代表图", content_unique, analysis.representatives)
    limited = _limit_candidates(analysis.representatives, config.max_frames, config=config, analysis=analysis)
    _stage("budget", "按新增内容自适应选图", analysis.representatives, limited)
    selected = limited
    recorder = _PROCESS.get()
    if recorder is not None:
        recorder.selection = analysis.summary
    return selected, effective_task_end, task_end_confidence, analysis.summary


def extract_scene_keyframes(
    video: Path | str,
    out_dir: Path | str,
    max_frames: int = 15,
    min_frames: int = 4,
    threshold: float = 0.02,
    min_gap_s: float = 0.15,
    max_edge: int = 720,
    *,
    task_start_time: float = DEFAULT_TASK_START_TIME,
    task_end_time: float | None = None,
    sample_fps: float = 3.0,
    config: KeyframeConfig | None = None,
    algorithm_version: str = KEYFRAME_ALGORITHM_VERSION,
) -> list[Path]:
    """抽取操作录屏关键帧并返回有序图片路径。

    旧参数 ``max_frames/min_frames/threshold/min_gap_s/max_edge`` 保持兼容；
    ``min_frames`` 在新算法中不再强制均匀补帧。高级参数可通过
    :class:`KeyframeConfig` 一次性传入。
    """
    del min_frames
    video = Path(video)
    out_dir = Path(out_dir)
    if config is None:
        config = KeyframeConfig(
            task_start_time=task_start_time,
            task_end_time=task_end_time,
            max_frames=max_frames,
            sample_fps=sample_fps,
            scene_threshold=threshold,
            scene_min_gap_s=min_gap_s,
            max_edge=max_edge,
        )
    elif (
        task_start_time != DEFAULT_TASK_START_TIME
        or task_end_time is not None
    ):
        raise ValueError("传入 config 时不能再单独传任务起止时间")
    duration = probe_duration(video)
    if duration <= 0:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("kf_*.jpg"):
        stale.unlink(missing_ok=True)
    metadata_path = out_dir / "keyframes.json"
    metadata_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="ae_kf_candidates_") as temp_dir:
        selected, effective_task_end, confidence, selection = _hybrid_keyframes(
            video,
            Path(temp_dir),
            config,
            duration,
        )
        frames: list[Path] = []
        records: list[dict] = []
        for index, candidate in enumerate(selected, start=1):
            target = out_dir / f"kf_{index:03d}.jpg"
            shutil.copy2(candidate.path, target)
            frames.append(target)
            records.append(
                {
                    "index": index,
                    "time": round(candidate.time, 3),
                    "source": candidate.source,
                    "keep_reason": candidate.keep_reason,
                    "evidence_ids": list(candidate.evidence_ids),
                }
            )

        metadata = {
            "algorithm_version": algorithm_version,
            "video": str(video),
            "duration": round(duration, 3),
            "config": asdict(config),
            "effective_task_end_time": (
                round(effective_task_end, 3)
                if effective_task_end is not None
                else None
            ),
            "task_end_confidence": (
                round(confidence, 5) if confidence is not None else None
            ),
            "selected_count": len(records),
            "selection": selection,
            "selected": records,

        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return frames


def encode_frame(
    path: Path | str,
    max_edge: int = 768,
    quality: int = 70,
) -> str:
    """等比缩放并 JPEG 压缩，返回 data:image/jpeg;base64 URL。"""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = max_edge / max(width, height)
    if scale < 1:
        image = image.resize(
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def video_to_frame_urls(
    video: Path | str,
    out_dir: Path | str | None = None,
    max_frames: int = 15,
    min_frames: int = 4,
    threshold: float = 0.02,
    min_gap_s: float = 0.15,
    max_edge: int = 768,
    quality: int = 70,
    *,
    task_start_time: float = DEFAULT_TASK_START_TIME,
    task_end_time: float | None = None,
    sample_fps: float = 3.0,
    config: KeyframeConfig | None = None,
) -> list[str]:
    """抽帧并编码为裁判可直接使用的 image_url 列表。"""
    video = Path(video)
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="ae_kf_"))
    frames = extract_scene_keyframes(
        video,
        out_dir,
        max_frames,
        min_frames,
        threshold,
        min_gap_s,
        max_edge,
        task_start_time=task_start_time,
        task_end_time=task_end_time,
        sample_fps=sample_fps,
        config=config,
    )
    return [encode_frame(frame, max_edge, quality) for frame in frames]
