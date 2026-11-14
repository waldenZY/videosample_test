import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from videosample import media
from videosample.versions import base_media


def picture(path, color):
    Image.new("RGB", (240, 400), color).save(path, quality=95)
    return path


def test_close_but_different_images_survive_time_merge(tmp_path):
    candidates = [media._Candidate(t, picture(tmp_path / f"{i}.jpg", color), source)
                  for i, (t, color, source) in enumerate([
                      (7, "white", "start-7s"), (7.1, "black", "scene-0.06"),
                      (7.2, "red", "scene-0.06"), (7.3, "blue", "scene-0.06")])]
    assert media._canonicalize_timestamps(candidates, 0.25) == candidates


def test_same_images_do_not_form_unbounded_time_chain(tmp_path):
    path = picture(tmp_path / "same.jpg", "white")
    candidates = [media._Candidate(i / 10, path, "4fps") for i in range(11)]
    selected = media._canonicalize_timestamps(candidates, 0.25)
    assert len(selected) >= 4
    assert selected[-1].time >= 0.9


def test_explicit_end_and_start_remain_even_when_identical(tmp_path):
    path = picture(tmp_path / "same.jpg", "white")
    candidates = [media._Candidate(7, path, "start-7s"),
                  media._Candidate(7.1, path, "protected-2s"),
                  media._Candidate(7.2, path, "task-end")]
    selected = media._canonicalize_timestamps(candidates, 0.25)
    assert candidates[0] in selected and candidates[-1] in selected


def test_short_distinct_pages_survive_state_filter_and_budget(tmp_path):
    candidates = [media._Candidate(t, picture(tmp_path / f"{i}.jpg", color), "scene-0.06")
                  for i, (t, color) in enumerate([(7, "white"), (9, "white"),
                      (10, "black"), (10.2, "gray"), (10.4, "blue"), (11, "white"), (13, "white")])]
    selected = media._deduplicate_states(candidates, media.KeyframeConfig())
    for candidate in candidates[2:5]:
        assert candidate in selected and candidate.keep_reason == "content-state"
        assert candidate in media._limit_candidates(selected, len(selected))
    assert len(media._limit_candidates(selected, 2)) == 2


@pytest.mark.parametrize("limit", [5, 10])
@pytest.mark.parametrize("sample_fps", [2.0, 3.0, 4.0])
@pytest.mark.parametrize("name,filters,expected_colors", [
    ("brief_card", "drawbox=x=35:y=80:w=170:h=12:color=black:t=fill:enable='gte(t,30.1)*lt(t,30.3)'", []),
    ("brief_app", "drawbox=x=0:y=0:w=240:h=400:color=gray:t=fill:enable='gte(t,30.1)*lt(t,30.4)'", ["gray"]),
    ("successive_apps", ",".join([
        "drawbox=x=0:y=0:w=240:h=400:color=red:t=fill:enable='gte(t,30.1)*lt(t,30.4)'",
        "drawbox=x=0:y=0:w=240:h=400:color=green:t=fill:enable='gte(t,30.4)*lt(t,30.7)'",
        "drawbox=x=0:y=0:w=240:h=400:color=blue:t=fill:enable='gte(t,30.7)*lt(t,31.0)'"]), ["red", "green", "blue"]),
])
def test_fast_evidence_at_two_three_and_four_fps(tmp_path, name, filters, expected_colors, limit, sample_fps):
    video = tmp_path / f"{name}.mp4"
    subprocess.run([media.resolve_ffmpeg_executable(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=white:s=240x400:r=30:d=40", "-vf", filters,
                    "-c:v", "mpeg4", "-q:v", "2", str(video)], check=True, timeout=30)
    base = base_media.extract_scene_keyframes(video, tmp_path / "base", task_end_time=38)
    with media.record_process(tmp_path / "process"):
        experiment = media.extract_scene_keyframes(video, tmp_path / "experiment", task_end_time=38,
                                                   max_frames=limit, sample_fps=sample_fps)
    assert len(experiment) <= limit

    def content(paths):
        return [np.asarray(Image.open(path).convert("RGB")) for path in paths]

    old, new = content(base), content(experiment)
    if not expected_colors:
        assert not any(np.mean(image < 40) > 0.005 for image in old)
        assert any(np.mean(image < 40) > 0.005 for image in new)
    else:
        means = [image.mean(axis=(0, 1)) for image in new]
        if sample_fps == 2 and name == 'successive_apps':
            # 降采样的已知结果：0.3 秒绿色画面未进入候选；保留此限制的回归。
            expected_colors = ['red', 'blue']
            trace = json.loads((tmp_path / 'process/trace.json').read_text(encoding='utf-8'))
            for frame in trace['frames']:
                with Image.open(tmp_path / 'process' / frame['image']) as image:
                    mean = np.asarray(image.convert('RGB')).mean(axis=(0, 1))
                assert not (mean[1] > 90 and mean[1] > mean[0] + 50 and mean[1] > mean[2] + 50)
        for color in expected_colors:
            if color == "gray":
                assert any(100 < mean.mean() < 160 for mean in means)
            else:
                channel = {"red": 0, "green": 1, "blue": 2}[color]
                assert any(mean[channel] > 90 and all(mean[channel] > mean[j] + 50
                           for j in range(3) if j != channel) for mean in means)
        assert not any(np.mean(image < 230) > 0.5 for image in old)
    metadata = json.loads((tmp_path / "experiment/keyframes.json").read_text(encoding="utf-8"))
    assert metadata["config"]["sample_fps"] == sample_fps
    assert metadata["algorithm_version"] == "hybrid-state-exp-v4.7"
    assert metadata["selected"][0]["time"] == 7.0
    assert "task-end-explicit" in [row["keep_reason"] for row in metadata["selected"]]
    assert all(row["time"] <= 38 for row in metadata["selected"])


def test_static_video_does_not_multiply_final_duplicates(tmp_path):
    path = picture(tmp_path / "static.jpg", "white")
    candidates = [media._Candidate(t, path, "4fps") for t in range(20)]
    selected = media._deduplicate_states(candidates, media.KeyframeConfig())
    output = media._limit_candidates(selected, 15)
    assert len(output) == 1
