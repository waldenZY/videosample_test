from pathlib import Path
import subprocess

import numpy as np
import pytest
from PIL import Image

from videosample import media


@pytest.mark.parametrize("rate,variable", [("30", False), ("30000/1001", False), ("30", True)])
def test_batched_capture_matches_individual_seeks_including_rounded_times(tmp_path, rate, variable):
    video = tmp_path / "moving.mp4"
    filters = ["-vf", "select='not(mod(n,3))'", "-vsync", "0"] if variable else []
    subprocess.run([media.resolve_ffmpeg_executable(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=120x200:rate={rate}:duration=2", *filters,
                    "-c:v", "mpeg4", "-q:v", "3", str(video)],
                   check=True, timeout=30)
    times = [0, 0.033333, 0.066667, 0.199, 0.201, 0.566667, 1.5, 1.999]
    batch = media._extract_many_at(video, times, tmp_path / "batch", max_edge=120)
    for index, timestamp in enumerate(times):
        individual = tmp_path / f"individual-{index}.jpg"
        exists = media._extract_at(video, timestamp, individual, max_edge=120)
        assert (timestamp in batch) == exists
        if exists:
            assert batch[timestamp].read_bytes() == individual.read_bytes()


def test_failed_batch_capture_falls_back_without_losing_scene_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(media.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(media, "scene_change_times", lambda *args, **kwargs: [8.1, 9.2])
    monkeypatch.setattr(media, "_extract_many_at", lambda *args, **kwargs: {})
    calls = []

    def capture(video, timestamp, output, **kwargs):
        calls.append(timestamp)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (96, 160), "white").save(output)
        return True

    monkeypatch.setattr(media, "_extract_at", capture)
    _, candidates = media._extract_candidates(tmp_path / "video.mp4", tmp_path / "work",
                                             media.KeyframeConfig(task_end_time=10), 12)
    assert [candidate.time for candidate in candidates if candidate.source.startswith("scene")] == [8.1, 9.2]
    assert 8.1 in calls and 9.2 in calls


def test_image_features_are_reused_only_within_one_extraction(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (96, 160), "white").save(path)

    @media._with_feature_cache
    def read():
        first = media._content_signature(path)
        assert media._content_signature(path) is first
        return first.copy()

    original = read()
    Image.new("RGB", (96, 160), "black").save(path)
    changed = read()
    # 固定布局 mask 会填黑顶部小框，框外的白色仍须更新为新文件的黑色。
    assert original[80, 48].mean() == 255 and changed.mean() == 0
    assert media._FEATURES.get() is None
    assert media._FEATURE_KEYS.get() is None


def test_identical_files_share_features_but_different_pixels_do_not(tmp_path):
    first, same, different = [tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg")]
    Image.new("RGB", (96, 160), "white").save(first)
    same.write_bytes(first.read_bytes())
    Image.new("RGB", (96, 160), "black").save(different)

    @media._with_feature_cache
    def read():
        assert media._layout_signature(first) is media._layout_signature(same)
        assert media._content_signature(first) is media._content_signature(same)
        assert media._content_signature(first) is not media._content_signature(different)
    read()


def test_feature_cache_is_bounded_and_cleared_on_failure():
    loads = []

    @media._cached_feature
    def feature(path):
        loads.append(str(path))
        return object()

    @media._with_feature_cache
    def run():
        for i in range(600):
            feature(Path(str(i)))
        assert len(media._FEATURES.get()) == 512
        assert len(media._FEATURE_KEYS.get()) == 512
        feature(Path("0"))
        assert len(loads) == 601
        raise ValueError("expected")

    try:
        run()
    except ValueError:
        pass
    assert media._FEATURES.get() is None
    assert media._FEATURE_KEYS.get() is None


def test_local_analysis_reads_each_candidate_once(tmp_path, monkeypatch):
    candidates = []
    random = np.random.default_rng(42)
    for index in range(40):
        path = tmp_path / f"{index}.png"
        Image.fromarray(random.integers(0, 256, (160, 96, 3), dtype=np.uint8)).save(path)
        candidates.append(media._Candidate(index, path, "4fps", page_id=index % 6))
    calls = []
    original = Image.open

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(Image, "open", counted)
    selected = media._limit_candidates(candidates, 10, config=media.KeyframeConfig(scroll_dedup_enabled=False))
    assert len(selected) == 10
    assert len(calls) == len(candidates)


def test_small_feature_cache_has_its_own_bound_and_is_released():
    @media._cached_feature
    def _layout_signature(path):
        return object()

    @media._with_feature_cache
    def run():
        for index in range(4200):
            _layout_signature(Path(str(index)))
        assert len(media._SMALL_FEATURES.get()) == 4096
        assert not media._FEATURES.get()
    run()
    assert media._SMALL_FEATURES.get() is None


def test_parallel_features_match_scalar_pixels_and_share_identical_files(tmp_path):
    random = np.random.default_rng(7)
    candidates = []
    for index in range(12):
        path = tmp_path / f'{index}.png'
        Image.fromarray(random.integers(0, 256, (160, 96, 3), dtype=np.uint8)).save(path)
        candidates.append(media._Candidate(index, path, '4fps'))
    candidates[3].path.write_bytes(candidates[0].path.read_bytes())

    @media._with_feature_cache
    def run():
        for function in (media._content_signature, media._layout_signature):
            scalar = [function.__wrapped__(candidate.path) for candidate in candidates]
            media._prewarm_features(candidates, function)
            for candidate, expected in zip(candidates, scalar):
                np.testing.assert_array_equal(function(candidate.path), expected)
            assert function(candidates[0].path) is function(candidates[3].path)
    run()
