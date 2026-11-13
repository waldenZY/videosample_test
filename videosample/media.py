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
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import lru_cache, wraps
from numbers import Integral, Real
from pathlib import Path

import numpy as np


KEYFRAME_ALGORITHM_VERSION = "hybrid-state-exp-v4.3"
DEFAULT_TASK_START_TIME = 7.0

# 仅供实验台记录过程；不参与任何选帧决策。定稿迁回时可整体移除。
_PROCESS = ContextVar("keyframe_process", default=None)
_FEATURES = ContextVar("keyframe_features", default=None)
_FEATURE_KEYS = ContextVar("keyframe_feature_keys", default=None)
_SMALL_FEATURES = ContextVar("keyframe_small_features", default=None)


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
        try:
            return function(*args, **kwargs)
        finally:
            _FEATURES.reset(token)
            _FEATURE_KEYS.reset(identities_token)
            _SMALL_FEATURES.reset(small_token)
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
    def load(path):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).digest()
        value = function.__wrapped__(io.BytesIO(payload))
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
    sample_fps: float = 4.0
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
    local_recording_timer_mask: bool = True
    local_recording_timer_box: tuple = (.63, .005, .74, .040)
    local_scroll_alignment: bool = True
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
        if (len(self.local_recording_timer_box) != 4
                or not all(isinstance(v, Real) and math.isfinite(v) for v in self.local_recording_timer_box)
                or not 0 <= self.local_recording_timer_box[0] < self.local_recording_timer_box[2] <= 1
                or not 0 <= self.local_recording_timer_box[1] < self.local_recording_timer_box[3] <= 1):
            raise ValueError("录屏计时区域必须是 0 到 1 的左、上、右、下坐标")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps 必须大于 0")
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
    command.extend(
        [
            "-an",
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
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


@_cached_feature
def _signature(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("L").resize((96, 160))
    array = np.asarray(image, dtype=np.float32)
    return array[12:154, :]


@_cached_feature
def _layout_signature(path: Path) -> np.ndarray:
    from PIL import Image, ImageFilter

    image = Image.open(path).convert("L")
    image = image.filter(ImageFilter.GaussianBlur(radius=12)).resize((16, 27))
    array = np.asarray(image, dtype=np.float32)
    return array[2:-1, :]


@_cached_feature
def _assistant_shell_signature(path: Path) -> np.ndarray:
    from PIL import Image, ImageFilter

    image = Image.open(path).convert("L")
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
        return np.asarray(image.convert("RGB").resize((96, 160)), dtype=np.uint8)


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


def _strict_local_equal(left, right, config):
    if not _strict_region_equal(left, right, config):
        return False
    cv = _local_cv()
    mask = (np.max(np.abs(left.astype(np.int16) - right.astype(np.int16)), axis=2) >= 12).astype(np.uint8)
    count, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    for x, y, w, h, area in stats[1:count]:
        if area < 4:
            continue
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(left.shape[1], x + w + 2), min(left.shape[0], y + h + 2)
        if not _strict_region_equal(left[y0:y1, x0:x1], right[y0:y1, x0:x1], config):
            return False
    return True


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


def _locate_local_content(track, crop, config, crop_gray=None):
    """用有纹理的小条匹配位移，再严格核对整个重叠区；不靠匹配分数合并文字。"""
    cv = _local_cv()
    reference = track.pixels
    if reference.shape[1] != crop.shape[1]:
        return None
    rh, ch = len(reference), len(crop)
    possible = {track.origin, track.origin + rh - ch}
    scores = {position: 0.0 for position in possible}
    if track.gray is None:
        track.gray = cv.cvtColor(reference, cv.COLOR_RGB2GRAY)
    rg = track.gray
    cg = crop_gray if crop_gray is not None else cv.cvtColor(crop, cv.COLOR_RGB2GRAY)
    for source, target, reverse in ((cg, rg, False), (rg, cg, True)):
        sh, th = len(source), len(target)
        strip_height = min(24, max(8, sh // 3))
        if strip_height > th:
            continue
        energy = np.mean(np.abs(np.diff(source.astype(np.float32), axis=1)), axis=1)
        textured = int(np.argmax(np.convolve(energy, np.ones(strip_height), mode='valid')))
        for y in sorted({0, max(0, (sh - strip_height) // 2), sh - strip_height, textured}):
            patch = source[y:y + strip_height, 2:-2]
            if float(patch.std()) < 3:
                continue
            matches = cv.matchTemplate(target, patch, cv.TM_CCOEFF_NORMED)
            _, score, _, (x, match_y) = cv.minMaxLoc(matches)
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
        # 多个独立边缘才追加小字检查；单个加载圆点的重绘沿用原移动规则。
        edge_count, _ = cv.connectedComponents(cv.Canny(cv.cvtColor(left, cv.COLOR_RGB2GRAY), 60, 160))
        if edge_count >= 5 and not _strict_local_equal(left, right, config):
            continue
        # 两幅有内容的图不能仅靠一段白底建立关系。
        if left.std() < 2 and (reference.std() >= 3 or crop.std() >= 3):
            continue
        value = (overlap, scores.get(position, 0), -abs(position - track.origin))
        if best is None or value > best[0]:
            best = (value, position, scores.get(position, 0))
    return None if best is None else (best[1], best[2])


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


def _analysis_pixels(source, config):
    """只处理分析副本；限定录屏布局的计时框，不屏蔽整个状态栏。"""
    if not config.local_recording_timer_mask:
        return source, []
    cv = _local_cv()
    height, width = source.shape[:2]
    # 已知录屏布局：左上方红色录屏按钮，以及同一行的右侧计时。
    # 未找到红色按钮时不套用这一位置假设，避免修改普通页面顶部内容。
    strip = source[:max(1, round(height * .05)), :round(width * .4)]
    red = ((strip[..., 0].astype(np.int16) - strip[..., 1] > 70)
           & (strip[..., 0].astype(np.int16) - strip[..., 2] > 70)
           & (strip[..., 0] > 150)).astype(np.uint8)
    count, _, stats, _ = cv.connectedComponentsWithStats(red, connectivity=8)
    if not any(.008 * height <= h <= .05 * height and .7 <= w / max(1, h) <= 1.4
               and area >= w * h * .25 and .10 * width <= x <= .36 * width
               for x, y, w, h, area in stats[1:count]):
        return source, []
    x0, y0, x1, y1 = config.local_recording_timer_box
    box = [round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)]
    result = source.copy()
    result[box[1]:box[3], box[0]:box[2]] = 0
    return result, [box]


def _screen_bands(pixels):
    """提取有边缘的横向图案；保留原尺寸，移动前后不缩放字形。"""
    cv = _local_cv()
    gray = cv.cvtColor(pixels, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 60, 160)
    # 彩色但亮度接近的按钮也必须留下比较线索。
    for channel in (0, 2):
        edges |= cv.Canny(pixels[..., channel], 60, 160)
    active = np.count_nonzero(edges, axis=1) >= 4
    runs, start = [], None
    for row, on in enumerate([*active, False]):
        if on and start is None:
            start = row
        elif not on and start is not None:
            if runs and start - runs[-1][1] <= 4:
                runs[-1] = (runs[-1][0], row)
            else:
                runs.append((start, row))
            start = None
    bands = []
    for top, bottom in runs:
        ys, xs = np.nonzero(edges[top:bottom])
        if len(xs) >= 8:
            bands.append((max(0, top - 2), min(len(pixels), bottom + 2),
                          max(0, int(xs.min()) - 2), min(pixels.shape[1], int(xs.max()) + 3)))
    return gray, bands


def _aligned_band_equal(left, right):
    """容忍高清画面移动重绘的两像素边缘偏差，检查新增笔画及内部颜色变化。"""
    cv = _local_cv()
    lg, rg = (cv.cvtColor(a, cv.COLOR_RGB2GRAY) for a in (left, right))
    # 例如 3→8 的小数字变化可能只有一两个像素宽，但字内闭合区域变了；
    # 不能让位移重绘容忍吞掉这种结构变化。
    def holes(gray, threshold):
        contours, hierarchy = cv.findContours((gray >= threshold).astype(np.uint8),
                                             cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        return sum(h[3] >= 0 and cv.contourArea(contour) >= 4
                   for contour, h in zip(contours, hierarchy[0])) if hierarchy is not None else 0
    if all(holes(lg, threshold) != holes(rg, threshold) for threshold in (80, 175)):
        return False
    le, re = cv.Canny(lg, 100, 220), cv.Canny(rg, 100, 220)
    kernel = np.ones((5, 5), np.uint8)
    unmatched = ((le > 0) & (cv.dilate(re, kernel) == 0)) | ((re > 0) & (cv.dilate(le, kernel) == 0))
    near_edge = cv.dilate(cv.Canny(lg, 60, 160) | cv.Canny(rg, 60, 160), kernel) > 0
    delta = np.max(np.abs(left.astype(np.int16) - right.astype(np.int16)), axis=2)
    changed = unmatched | ((delta > 24) & ~near_edge)
    count, _, stats, _ = cv.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
    return not any(area >= 4 for x, y, w, h, area in stats[1:count])


def _screen_matches(reference, candidate, config):
    """找共同滚动的多行图案；未匹配的按钮、文字仍是独立内容。"""
    cv = _local_cv()
    rp, rg, rb = reference
    cp, cg, cb = candidate
    if rp.shape != cp.shape or not cb or not rb:
        return None
    if np.max(np.abs(np.median(rp[::8, ::8], axis=(0, 1)) - np.median(cp[::8, ::8], axis=(0, 1)))) > 16:
        return None
    used, shifts, matches = [], [], []
    for number, (top, bottom, left, right) in enumerate(cb):
        patch = cp[top:bottom, left:right]
        # 同位置先核对，固定标题、输入栏不需要模板搜索。
        exact = _strict_region_equal(rp[top:bottom, left:right], patch, config)
        position = top if exact or _aligned_band_equal(rp[top:bottom, left:right], patch) else None
        band_approximate = not exact
        if position is None:
            result = cv.matchTemplate(rg[:, left:right], cg[top:bottom, left:right], cv.TM_CCOEFF_NORMED)
            _, score, _, (_, y) = cv.minMaxLoc(result)
            if score < .85 or not _aligned_band_equal(rp[y:y + len(patch), left:right], patch):
                continue
            position = y
            band_approximate = not _strict_region_equal(rp[y:y + len(patch), left:right], patch, config)
        if any(max(a, position) < min(b, position + len(patch)) - 2 for a, b in used):
            continue  # 不允许重复出现的多行内容共用同一参考行。
        used.append((position, position + len(patch)))
        if abs(position - top) > 2:
            shifts.append(position - top)
        ref_number = max(range(len(rb)), key=lambda i: min(rb[i][1], position + len(patch)) - max(rb[i][0], position))
        overlap = min(rb[ref_number][1], position + len(patch)) - max(rb[ref_number][0], position)
        if overlap >= len(patch) * .7:
            matches.append((number, ref_number, position - top, band_approximate))
    if not shifts and len(matches) == len(cb) and all(not approximate for n, r, delta, approximate in matches):
        return {'vertical_shift': 0, 'matches': [(n, r) for n, r, delta, approximate in matches]}
    if len(shifts) < 3 or cp.shape[1] < 480:
        return None
    # 至少三条图案共同移动；共享的每一条仍须通过颜色与结构检查。
    shift = max(shifts, key=lambda v: sum(abs(v - other) <= 2 for other in shifts))
    matches = [(n, r) for n, r, delta, approximate in matches
               if abs(delta) <= 2 or abs(delta - shift) <= 2]
    moving = sum(abs(cb[n][0] - rb[r][0]) > 2 for n, r in matches)
    if moving < 3 or len(matches) < min(len(cb), len(rb)) * .5:
        return None
    return {'vertical_shift': int(shift), 'matches': matches}


def _scroll_coverage(analysis, config, candidate, pixels, references, overrides, index):
    """复用本轮已解码图片，固定参考逐条登记正文与独立变化。"""
    if not config.local_scroll_alignment or pixels.shape[1] < 480:
        return
    cv = _local_cv()
    preview = cv.resize(pixels, (180, max(1, round(len(pixels) * 180 / pixels.shape[1]))),
                        interpolation=cv.INTER_AREA)
    colors = preview.astype(np.int32) // 16
    color_keys = colors[..., 0] * 256 + colors[..., 1] * 16 + colors[..., 2]
    if np.bincount(color_keys.ravel(), minlength=4096).max() < color_keys.size * .25:
        return  # 没有占主要面积的界面底色时，不假定它是可滚动的文字页面。
    preview_edges = cv.Canny(cv.cvtColor(preview, cv.COLOR_RGB2GRAY), 60, 160)
    if np.mean(np.count_nonzero(preview_edges, axis=1) >= 4) > .8:
        return  # 持续密集运动画面没有可分离的正文行，沿用原局部检查。
    screen = (pixels, *_screen_bands(pixels))
    if not 4 <= len(screen[2]) <= 64:
        return
    matched = None
    for reference in references:
        if len(reference['patches']) + len(screen[2]) > 256:
            continue
        details = _screen_matches(reference['screen'], screen, config)
        if details is not None:
            matched = reference, details
            break
    if matched is not None:
        reference, details = matched
        root = reference['candidate']
        prefix = f's{reference["index"]:04d}'
        bands = reference['screen'][2]
        if not reference['active']:
            reference['active'] = True
            overrides[id(root)] = {f'{prefix}:{i}' for i in range(len(bands))}
            for i, (top, bottom, left, right) in enumerate(bands):
                reference['patches'].append((reference['screen'][0][top:bottom, left:right],
                                             (top, bottom, left, right), (top, bottom, left, right)))
                analysis.units[f'{prefix}:{i}'] = {'id': f'{prefix}:{i}', 'kind': 'scroll-band',
                                                  'weight': 1.5 / len(bands)}
        mapping = dict(details['matches'])
        shift = details['vertical_shift']
        moving_rows = [screen[2][n][0] for n, r in details['matches']
                       if abs(screen[2][n][0] - bands[r][0]) > 2]
        covered = set()
        for number, box in enumerate(screen[2]):
            found = mapping.get(number)
            if found is None:
                top, bottom, left, right = box
                moving = moving_rows and min(moving_rows) - 60 <= top <= max(moving_rows) + 120
                normalized = (top + shift, bottom + shift, left, right) if moving else box
                patch = pixels[top:bottom, left:right]
                # 新状态只与固定的首次像素核对，不能沿后续近似图片串联。
                for i, (old_patch, old_box, old_normalized) in enumerate(reference['patches']):
                    ot, ob, ol, oright = old_box
                    if (max(abs(left - ol), abs(right - oright)) > 2 or abs((bottom - top) - (ob - ot)) > 2
                            or abs(normalized[0] - old_normalized[0]) > 3):
                        continue
                    compared = old_patch if old_patch.shape == patch.shape else _local_cv().resize(
                        old_patch, (patch.shape[1], patch.shape[0]), interpolation=_local_cv().INTER_AREA)
                    if (_strict_region_equal(compared, patch, config)
                            or _aligned_band_equal(compared, patch)):
                        found = i
                        break
            if found is None:
                found = len(reference['patches'])
                reference['patches'].append((patch.copy(), box, normalized))
                key = f'{prefix}:{found}'
                analysis.units[key] = {'id': key, 'kind': 'scroll-band', 'weight': 1.5 / len(bands)}
            covered.add(f'{prefix}:{found}')
        overrides[id(candidate)] = covered
        analysis.details[id(candidate)]['scroll_comparison'] = {
            'reference_time': root.time, 'vertical_shift': details['vertical_shift'],
            'shared_bands': len(details['matches']), 'independent_bands': len(covered) - len(set(mapping.values()))}
    else:
        references.insert(0, {'candidate': candidate, 'screen': screen, 'active': False,
                              'patches': [], 'index': index})
        references[:] = [ref for ref in references if len(ref['patches']) < 256][:4]

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
            return candidate, digest, None, None, None, None, None
        with Image.open(io.BytesIO(payload) if payload is not None else candidate.path) as image:
            source = np.asarray(image.convert('RGB'))
        source, ignored = _analysis_pixels(source, config)
        height, width = source.shape[:2]
        scale = min(1.0, config.local_analysis_max_edge / max(height, width))
        small = cv.resize(source, (max(8, round(width * scale)), max(8, round(height * scale))),
                          interpolation=cv.INTER_AREA)
        boxes, background = _local_boxes(small)
        return candidate, digest, source, small, boxes, background, ignored
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
    """仅分析名额分配前的候选；计算过程与实验台记录完全独立。"""
    cv = _local_cv()
    analysis = _LocalAnalysis(list(candidates))
    tracks, backgrounds, fingerprints, background_keys = [], [], {}, {}
    scroll_references, scroll_overrides, exact_copies = [], {}, {}
    background_descriptors = np.empty((len(candidates), 48), dtype=np.float32)
    for index, (candidate, digest, source, small, boxes, background, ignored) in enumerate(_local_views(candidates, config, fingerprints)):
        identity = id(candidate)
        candidate.evidence_ids = ()
        analysis.coverage[identity] = set()
        if digest in fingerprints:
            previous = fingerprints[digest]
            exact_copies[identity] = id(previous)
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
            active = [track for track in tracks if track.pixels is not None and track.kind == kind]
            # 位移跟踪只检查时间上邻近的内容；跨时间的完全重复由文件标记复用。
            # 整页逐一匹配几十个旧页面既不能解释操作，还会使动态视频耗时失控。
            recent = sorted(active, key=lambda t: (-t.last_index, t.index))[:1 if fallback else 8]
            for track in recent:
                if not .85 <= ratio / track.width_ratio <= 1.18:
                    continue
                if fallback and np.abs(crop_color - track.color).max() > 40:
                    continue
                location = _locate_local_content(track, crop, config, crop_gray)
                if location is not None and _extend_local_track(track, crop, location[0]):
                    matched = track, location
                    break
            if matched is None:
                track = _LocalTrack(len(tracks), crop.copy(), 0, ratio, kind, index)
                track.color, track.gray = crop_color, crop_gray
                if fallback and float(crop_gray.std()) < 2:
                    track.reliable = not any(old.gray is not None and float(old.gray.std()) < 2
                                             and np.abs(old.color - crop_color).max() <= 8 for old in active)
                tracks.append(track)
                position, confidence = 0, 0.0
            else:
                track, (position, confidence) = matched
            track.last_index = index
            track.observations.append({'candidate': candidate, 'top': position, 'bottom': position + len(crop)})
            clipped = 0 if fallback else sum((x <= 1, y <= 1, x + w >= small.shape[1] - 1,
                                              y + h >= small.shape[0] - 1))
            gray = cv.cvtColor(crop, cv.COLOR_RGB2GRAY)
            clarity = float(cv.Laplacian(gray, cv.CV_32F).var())
            exposures.append(len(crop))
            clipped_edges.append(clipped)
            sharpness.append(clarity)
            motion.append(abs(position))
            regions.append({'track': track.index, 'kind': kind, 'box': [x0, y0, x1 - x0, y1 - y0],
                            'possible_clipping': bool(clipped), 'match_confidence': round(float(confidence), 4),
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
        analysis.details[identity] = {'regions': regions, 'background_id': background_id, 'ignored_recording_timer': ignored,
                                      'note': '局部区域和裁切线索，不代表已识别弹窗用途或确认完整性'}
        _scroll_coverage(analysis, config, candidate, source, scroll_references, scroll_overrides, index)
        # 活跃跟踪最多保存 32 份参考图；更早的内容只留覆盖区间和记录。
        active = sorted((track for track in tracks if track.pixels is not None),
                        key=lambda t: (t.last_index, t.index))
        for old in active[:-32]:
            old.bins = _local_track_bins(old, config)
            old.pixels = None
            old.gray = None
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
    analysis.coverage.update(scroll_overrides)
    for identity, previous in exact_copies.items():
        analysis.coverage[identity] = set(analysis.coverage[previous])
        if 'scroll_comparison' in analysis.details[previous]:
            analysis.details[identity]['scroll_comparison'] = analysis.details[previous]['scroll_comparison']
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
            _note(candidate, '正文对齐后换用展示更充分的代表图' if 'scroll_comparison' in details
                  else '同一内容换用展示更充分或更清楚的代表图', [representative], **details)
        else:
            _note(candidate, '保留独立内容或互补范围的候选，尚未占用最终名额', **details)
    return analysis


def _limit_candidates(candidates, limit, *, config=None, analysis=None):
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
    sample_fps: float = 4.0,
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
    sample_fps: float = 4.0,
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
