"""任务类（录屏）的混合关键帧抽取与帧编码。

流程：固定频率采样 + scene 候选 → UI 状态聚类 → 短暂弹窗保护 →
任务结束点判断 → 强制首帧/任务结束帧（无显式结束点时保留视频末帧）→ 最终严格去重。

纯 ffmpeg + Pillow + NumPy，不依赖 OpenCV。
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path

import numpy as np


KEYFRAME_ALGORITHM_VERSION = "hybrid-state-exp-v3"
DEFAULT_TASK_START_TIME = 7.0

# 仅供实验台记录过程；不参与任何选帧决策。定稿迁回时可整体移除。
_PROCESS = ContextVar("keyframe_process", default=None)


class _ProcessRecord:
    def __init__(self, directory: Path):
        self.directory = directory
        (directory / "candidates").mkdir(parents=True, exist_ok=True)
        self.identities = {}
        self.frames = []
        self.stages = []
        self.notes = {}

    def identify(self, candidate):
        identity = self.identities.get(id(candidate))
        if identity is None:
            identity = f"f{len(self.frames) + 1:05d}"
            self.identities[id(candidate)] = identity
            target = self.directory / "candidates" / f"{identity}.jpg"
            shutil.copy2(candidate.path, target)
            self.frames.append({"id": identity, "time": round(candidate.time, 3),
                                "source": candidate.source,
                                "image": f"candidates/{identity}.jpg", "history": []})
        return identity

    def stage(self, key, title, before, after, fallback):
        before_ids = [self.identify(c) for c in before]
        after_ids = [self.identify(c) for c in after]
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
        self.stages.append({"id": key, "title": title, "input_ids": before_ids,
                            "output_ids": after_ids,
                            "removed_ids": [i for i in before_ids if i not in selected],
                            "events": events})
        self.notes.clear()
        self.save()

    def save(self):
        payload = {"algorithm_version": KEYFRAME_ALGORITHM_VERSION,
                   "frames": self.frames, "stages": self.stages}
        target = self.directory / "trace.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)


@contextmanager
def record_process(directory: Path | str):
    """实验时保留候选图与每步决定；不改变抽帧入口及返回值。"""
    recorder = _ProcessRecord(Path(directory))
    token = _PROCESS.set(recorder)
    try:
        yield recorder
    finally:
        recorder.save()
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
    max_frames: int = 25
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
        if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, Integral) or self.max_frames < 2:
            raise ValueError("max_frames 必须为不小于 2 的整数")
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
    max_frames: int = 25,
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


def _signature(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("L").resize((96, 160))
    array = np.asarray(image, dtype=np.float32)
    return array[12:154, :]


def _layout_signature(path: Path) -> np.ndarray:
    from PIL import Image, ImageFilter

    image = Image.open(path).convert("L")
    image = image.filter(ImageFilter.GaussianBlur(radius=12)).resize((16, 27))
    array = np.asarray(image, dtype=np.float32)
    return array[2:-1, :]


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
    for index, timestamp in enumerate(scene_times, start=1):
        if not effective_start <= timestamp <= algorithm_end:
            continue
        path = output_dir / "scenes" / f"scene_{index:04d}_{timestamp:.3f}.jpg"
        if _extract_at(video, timestamp, path, max_edge=config.max_edge):
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
    for candidate in candidates:
        image = pixels(str(candidate.path))
        descriptor = _descriptor(image)
        found = None
        for group in reversed(groups):
            if float(np.sqrt(np.mean((descriptor - group["descriptor"]) ** 2))) > config.content_rms_threshold:
                continue
            other = pixels(str(group["reference"].path))
            rms, fraction = _content_difference(image, other)
            if (rms < config.content_rms_threshold and fraction < config.content_changed_fraction
                    and _tile_difference(image, other) < config.content_tile_threshold):
                found = group
                break
        if found is None:
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
    selected = []
    interval = 1.0 / config.sample_fps
    for index, candidate in enumerate(candidates):
        candidate.support_s = min(interval, max(0.0, candidates[index + 1].time - candidate.time)) if index + 1 < len(candidates) else interval
    for run in runs:
        layout = layouts[run[0]]
        page = next((i for i, known in enumerate(pages)
                     if _layout_distance(layout, known) < config.state_layout_threshold), None)
        if page is None:
            page = len(pages)
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


def _limit_candidates(candidates, limit, *, config=None):
    """内容先归组；优先覆盖不同页面，再分配剩余内容名额。"""
    if not candidates:
        return []
    config = config or KeyframeConfig()
    if len(candidates) <= limit:
        for candidate in candidates:
            _note(candidate, "内容代表数量未超过上限，保留", frame_limit=limit)
        return candidates
    # 只有时间语义上的首尾为固定名额，短页面与定时保护采样不再无限豁免。
    anchors = [candidates[0]]
    endings = sorted([c for c in candidates[1:] if c.keep_reason in {
        "task-end-explicit", "task-end-auto", "final-frame"}],
        key=lambda c: (c.keep_reason != "final-frame", c.time), reverse=True)
    anchors.extend(endings[:max(0, limit - 1)])
    selected = list(dict((id(c), c) for c in anchors).values())
    pixels = {id(c): _content_signature(c.path) for c in candidates}

    def distance(left, right):
        rms, _ = _content_difference(pixels[id(left)], pixels[id(right)])
        return rms + 0.5 * _tile_difference(pixels[id(left)], pixels[id(right)])

    nearest = {id(c): min(distance(c, other) for other in selected) for c in candidates if c not in selected}
    decisions = {id(c): "保留任务首帧或结束位置" for c in selected}
    while len(selected) < min(limit, len(candidates)):
        remaining = [c for c in candidates if c not in selected]
        covered = {c.page_id for c in selected}
        uncovered = [c for c in remaining if c.page_id not in covered]
        pool = uncovered or remaining

        def score(candidate):
            same_page = sum(c.page_id == candidate.page_id for c in selected)
            stability = 0.8 + 0.2 * min(1.0, candidate.support_s / max(0.001, config.stable_min_duration_s))
            return nearest[id(candidate)] * stability / (1.0 + 0.35 * same_page)

        best = max(pool, key=lambda c: (score(c), c.support_s, c.time))
        decisions[id(best)] = "为尚未覆盖的页面保留内容代表" if uncovered else "优先保留新增内容较多的画面"
        selected.append(best)
        nearest.pop(id(best))
        for candidate in remaining:
            if candidate is not best:
                nearest[id(candidate)] = min(nearest[id(candidate)], distance(candidate, best))
    selected_ids = {id(c) for c in selected}
    for candidate in candidates:
        if id(candidate) in selected_ids:
            _note(candidate, decisions[id(candidate)], frame_limit=limit, page=candidate.page_id + 1,
                  support_s=candidate.support_s)
        else:
            other = min(selected, key=lambda c: distance(candidate, c))
            _note(candidate, "内容名额已满，优先覆盖不同页面及差异更大的内容", [other],
                  frame_limit=limit, page=candidate.page_id + 1,
                  distance_to_retained=distance(candidate, other),
                  note="相关图是最相近的已选画面，不表示全部内容都相同或已完整覆盖")
    return sorted(selected, key=lambda c: c.time)


def _final_deduplicate(
    candidates: list[_Candidate],
    config: KeyframeConfig,
) -> tuple[list[_Candidate], list[_Candidate]]:
    """跨全部最终候选检查高度重复，不受相邻关系或时间间隔限制。"""
    if not candidates:
        return [], []
    signatures = {id(c): _content_signature(c.path) for c in candidates}
    first = min(candidates, key=lambda c: c.time)
    endpoints = {id(c) for c in candidates if c is first or c.keep_reason in {
        "task-end-explicit", "task-end-auto", "final-frame"}}
    # 首尾先确定；其他图片优先选择停留更充分的代表，不按时间顺序截断。
    ordered = sorted([c for c in candidates if id(c) in endpoints], key=lambda c: c.time)
    ordered += sorted([c for c in candidates if id(c) not in endpoints],
                      key=lambda c: (c.support_s, c.novelty, c.time), reverse=True)
    kept, removed = [], []
    for candidate in ordered:
        image = signatures[id(candidate)]
        matches = []
        checks = []
        comparisons = []
        for representative in kept:
            other = signatures[id(representative)]
            rms, fraction = _content_difference(image, other)
            tile = _tile_difference(image, other)
            delta = np.max(np.abs(image.astype(np.float32) - other.astype(np.float32)), axis=-1) / 255.0
            patch = float(delta.reshape(40, 4, 24, 4).mean(axis=(1, 3)).max())
            similar = (rms < config.global_dedup_rms_threshold
                       and fraction < config.global_dedup_changed_fraction
                       and tile < config.global_dedup_tile_threshold
                       and patch < config.global_dedup_patch_threshold)
            failed = [name for name, value, threshold in [
                ("整体差异", rms, config.global_dedup_rms_threshold),
                ("变化像素比例", fraction, config.global_dedup_changed_fraction),
                ("局部区域差异", tile, config.global_dedup_tile_threshold),
                ("细小区域差异", patch, config.global_dedup_patch_threshold)] if value >= threshold]
            check = {"time": representative.time, "rms": rms,
                           "changed_fraction": fraction, "max_tile_difference": tile,
                           "max_patch_difference": patch, "highly_similar": bool(similar),
                           "not_merged_because": failed}
            checks.append(check)
            comparisons.append((representative, check))
            if similar:
                matches.append(representative)
        details = {"scope": "all_final_candidates", "comparisons": checks,
                   "rms_threshold": config.global_dedup_rms_threshold,
                   "changed_fraction_threshold": config.global_dedup_changed_fraction,
                   "tile_threshold": config.global_dedup_tile_threshold,
                   "patch_threshold": config.global_dedup_patch_threshold}
        if matches and id(candidate) not in endpoints:
            representative = min(matches, key=lambda c: _content_difference(image, signatures[id(c)])[0])
            removed.append(candidate)
            _note(candidate, "全局检查发现高度重复，保留了另一时刻的代表图", [representative],
                  time_gap_s=abs(candidate.time - representative.time), **details)
        else:
            kept.append(candidate)
            if matches:
                _note(candidate, "画面高度相似，但保留任务首尾位置", matches, **details)
            elif comparisons:
                representative, check = min(comparisons, key=lambda pair: max(
                    pair[1]["rms"] / max(1e-9, config.global_dedup_rms_threshold),
                    pair[1]["changed_fraction"] / max(1e-9, config.global_dedup_changed_fraction),
                    pair[1]["max_tile_difference"] / max(1e-9, config.global_dedup_tile_threshold),
                    pair[1]["max_patch_difference"] / max(1e-9, config.global_dedup_patch_threshold)))
                _note(candidate, "与最接近画面比较后仍保留：" + "、".join(check["not_merged_because"]),
                      [representative], closest_time=representative.time,
                      not_merged_because=check["not_merged_because"], **details)
            else:
                _note(candidate, "保留第一张代表画面，后续图片会与它比较", **details)
    # 只与最终确实留下的代表比较，不把 A≈B、B≈C 串联成 A≈C；去重后不补图凑数。
    return sorted(kept, key=lambda c: c.time), sorted(removed, key=lambda c: c.time)


def _hybrid_keyframes(
    video: Path,
    work_dir: Path,
    config: KeyframeConfig,
    duration: float,
) -> tuple[list[_Candidate], list[_Candidate], float | None, float | None]:
    effective_task_end, raw = _extract_candidates(
        video,
        work_dir,
        config,
        duration,
    )
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
    limited = _limit_candidates(content_unique, config.max_frames, config=config)
    _stage("budget", "按内容分配图片名额", content_unique, limited)
    selected, removed = _final_deduplicate(limited, config)
    _stage("final", "全局去除高度重复 · 最终结果", limited, selected)
    return selected, removed, effective_task_end, task_end_confidence


def extract_scene_keyframes(
    video: Path | str,
    out_dir: Path | str,
    max_frames: int = 25,
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
        selected, removed, effective_task_end, confidence = _hybrid_keyframes(
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
            "selected": records,
            "final_dedup_removed_count": len(removed),
            "final_dedup_removed": [
                {
                    "time": round(candidate.time, 3),
                    "source": candidate.source,
                    "keep_reason": candidate.keep_reason,
                }
                for candidate in removed
            ],
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
    max_frames: int = 25,
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
