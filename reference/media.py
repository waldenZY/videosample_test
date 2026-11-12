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
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import numpy as np


KEYFRAME_ALGORITHM_VERSION = "hybrid-state-v3.2.0"
DEFAULT_TASK_START_TIME = 7.0


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
    max_frames: int = 20
    sample_fps: float = 1.0
    scene_threshold: float = 0.06
    scene_min_gap_s: float = 0.8
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
        if self.max_frames < 2:
            raise ValueError("max_frames 不能小于 2")
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
    threshold: float = 0.06,
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
    max_frames: int = 20,
    min_frames: int = 4,
    threshold: float = 0.06,
    min_gap_s: float = 0.8,
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
    ps_interval = round(config.protected_sample_interval)
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
                if round(c.time - effective_start) % ps_interval == 0:
                    protected = True
            # 结尾窗口：[algorithm_end - protected_end_window, algorithm_end]
            # 帧时间相对终点的偏移能被 interval 整除则保留
            if not protected and config.protected_end_window > 0 and protected_end_start <= c.time <= algorithm_end:
                if round(algorithm_end - c.time) % ps_interval == 0:
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
) -> list[_Candidate]:
    if not candidates:
        return []
    groups: list[list[_Candidate]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate.time - groups[-1][-1].time <= gap:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    def priority(candidate: _Candidate) -> tuple[int, float]:
        if candidate.source.startswith("start"):
            return (0, candidate.time)
        if candidate.source.startswith("protected"):
            return (1, candidate.time)
        if candidate.source == "task-end":
            return (2, candidate.time)
        if candidate.source.startswith("scene"):
            return (3, candidate.time)
        return (4, candidate.time)

    return [min(group, key=priority) for group in groups]


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


def _deduplicate_states(
    candidates: list[_Candidate],
    config: KeyframeConfig,
) -> list[_Candidate]:
    if not candidates:
        return []
    layouts, runs = _build_layout_runs(
        candidates,
        config.state_layout_threshold,
    )
    stable_runs = {
        run_index
        for run_index, run in enumerate(runs)
        if run_index in {0, len(runs) - 1}
        or candidates[run[-1]].time - candidates[run[0]].time
        >= config.stable_min_duration_s
    }

    transient_runs: set[int] = set()
    for run_index in range(1, len(runs) - 1):
        if run_index in stable_runs:
            continue
        run = runs[run_index]
        run_duration = candidates[run[-1]].time - candidates[run[0]].time
        surrounding_distance = _layout_distance(
            layouts[runs[run_index - 1][-1]],
            layouts[runs[run_index + 1][0]],
        )
        if (
            run_duration <= config.transient_max_duration_s
            and surrounding_distance < config.transient_return_threshold
        ):
            transient_runs.add(run_index)

    selected_indices: set[int] = set()
    # 受保护帧跳过状态聚类去重，始终保留
    protected_indices = {
        i for i, c in enumerate(candidates) if c.source.startswith("protected")
    }
    selected_indices.update(protected_indices)
    for run_index, run in enumerate(runs):
        first, last = run[0], run[-1]
        if run_index in transient_runs:
            before = layouts[runs[run_index - 1][-1]]
            after = layouts[runs[run_index + 1][0]]
            peak = max(
                run,
                key=lambda index: min(
                    _layout_distance(layouts[index], before),
                    _layout_distance(layouts[index], after),
                ),
            )
            peak_pixels = _signature(candidates[peak].path)
            before_pixels = _signature(
                candidates[runs[run_index - 1][-1]].path
            )
            after_pixels = _signature(
                candidates[runs[run_index + 1][0]].path
            )
            changed_against_both = (
                (np.abs(peak_pixels - before_pixels) >= 12.0)
                & (np.abs(peak_pixels - after_pixels) >= 12.0)
            )
            changed_fraction = float(np.mean(changed_against_both))
            if (
                config.transient_min_changed_fraction
                <= changed_fraction
                <= config.transient_max_changed_fraction
            ):
                selected_indices.add(peak)
                candidates[peak].keep_reason = "transient-local-A-B-A"
            continue
        if run_index not in stable_runs:
            continue

        selected_indices.add(first)
        candidates[first].keep_reason = "stable-state-start"
        if last != first:
            rms, changed_fraction = _visual_difference(
                _signature(candidates[first].path),
                _signature(candidates[last].path),
            )
            if (
                run_index == len(runs) - 1
                or rms >= 0.020
                or changed_fraction >= 0.025
            ):
                selected_indices.add(last)
                candidates[last].keep_reason = "stable-state-end"

    return [candidates[index] for index in sorted(selected_indices)]


def _limit_candidates(
    candidates: list[_Candidate],
    limit: int,
) -> list[_Candidate]:
    if len(candidates) <= limit:
        return candidates
    mandatory_indices = {0, len(candidates) - 1}
    mandatory_indices.update(
        index
        for index, candidate in enumerate(candidates)
        if candidate.keep_reason
        in {
            "stable-state-start",
            "transient-local-A-B-A",
            "task-end-explicit",
            "task-end-auto",
            "final-frame",
        }
    )
    # 受保护帧强制保留
    mandatory_indices.update(
        index
        for index, candidate in enumerate(candidates)
        if candidate.source.startswith("protected")
    )
    if len(mandatory_indices) > limit:
        protected = {
            index
            for index in mandatory_indices
            if index in {0, len(candidates) - 1}
            or candidates[index].keep_reason
            in {
                "transient-local-A-B-A",
                "task-end-explicit",
                "task-end-auto",
                "final-frame",
            }
            or candidates[index].source.startswith("protected")
        }
        remaining = sorted(mandatory_indices - protected)
        slots = max(0, limit - len(protected))
        if slots and remaining:
            positions = np.linspace(0, len(remaining) - 1, slots).round().astype(int)
            protected.update(remaining[position] for position in positions)
        mandatory_indices = protected

    selected = set(mandatory_indices)
    while len(selected) < limit:
        remaining = [
            index for index in range(len(candidates)) if index not in selected
        ]
        best = max(
            remaining,
            key=lambda index: (
                min(
                    abs(candidates[index].time - candidates[kept].time)
                    for kept in selected
                ),
                candidates[index].novelty,
            ),
        )
        selected.add(best)
    return [candidates[index] for index in sorted(selected)]


def _final_deduplicate(
    candidates: list[_Candidate],
    config: KeyframeConfig,
) -> tuple[list[_Candidate], list[_Candidate]]:
    if not candidates:
        return [], []
    signatures = {
        id(candidate): _signature(candidate.path) for candidate in candidates
    }
    groups: list[list[_Candidate]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        previous = groups[-1][-1]
        rms, changed_fraction = _visual_difference(
            signatures[id(previous)],
            signatures[id(candidate)],
        )
        # 受保护帧使用更严格的去重阈值（仅高度一致时才合并）
        is_any_protected = (
            previous.source.startswith("protected")
            or candidate.source.startswith("protected")
        )
        dedup_rms = (
            config.protected_sample_dedup_rms
            if is_any_protected
            else config.final_dedup_rms_threshold
        )
        dedup_cf = (
            config.protected_sample_dedup_cf
            if is_any_protected
            else config.final_dedup_changed_fraction_threshold
        )
        if rms < dedup_rms and changed_fraction < dedup_cf:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    protected_reasons = {
        "task-end-explicit",
        "task-end-auto",
        "final-frame",
    }
    ordinary_priority = {
        "transient-local-A-B-A": 3,
        "stable-state-end": 2,
        "stable-state-start": 1,
    }
    first_candidate = candidates[0]
    kept: list[_Candidate] = []
    removed: list[_Candidate] = []
    for group in groups:
        protected = [
            candidate
            for candidate in group
            if candidate is first_candidate
            or candidate.keep_reason in protected_reasons
        ]
        if protected:
            group_kept = protected
        else:
            group_kept = [
                max(
                    group,
                    key=lambda candidate: (
                        ordinary_priority.get(candidate.keep_reason, 0),
                        candidate.novelty,
                        candidate.time,
                    ),
                )
            ]
        kept_ids = {id(candidate) for candidate in group_kept}
        kept.extend(group_kept)
        removed.extend(
            candidate for candidate in group if id(candidate) not in kept_ids
        )
    return (
        sorted(kept, key=lambda candidate: candidate.time),
        sorted(removed, key=lambda candidate: candidate.time),
    )


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
    )

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
    state_unique = _deduplicate_states(timestamp_unique, config)

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
    limited = _limit_candidates(
        sorted(combined.values(), key=lambda candidate: candidate.time),
        config.max_frames,
    )
    selected, removed = _final_deduplicate(limited, config)
    return selected, removed, effective_task_end, task_end_confidence


def extract_scene_keyframes(
    video: Path | str,
    out_dir: Path | str,
    max_frames: int = 20,
    min_frames: int = 4,
    threshold: float = 0.06,
    min_gap_s: float = 0.8,
    max_edge: int = 720,
    *,
    task_start_time: float = DEFAULT_TASK_START_TIME,
    task_end_time: float | None = None,
    sample_fps: float = 1.0,
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
    max_frames: int = 20,
    min_frames: int = 4,
    threshold: float = 0.06,
    min_gap_s: float = 0.8,
    max_edge: int = 768,
    quality: int = 70,
    *,
    task_start_time: float = DEFAULT_TASK_START_TIME,
    task_end_time: float | None = None,
    sample_fps: float = 1.0,
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
