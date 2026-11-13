import json
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw

from videosample import media
from videosample.versions import base_media


def card(path, y, variant=0, jitter=0):
    image = Image.new("RGB", (240, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, y, 220, y + 70), fill="#d8e1eb")
    draw.rectangle((30, y + 10, 90 + variant * 30, y + 30), fill="#162e50")
    draw.rectangle((230, 3, 232, 5), fill=(jitter, jitter, jitter))
    image.save(path, quality=95)
    return path


def test_same_page_keeps_different_local_cards_and_merges_jitter(tmp_path):
    candidates = [media._Candidate(i / 4, card(tmp_path / f"{i}.jpg", 100, variant=i // 20,
                                             jitter=(i % 20) * 10), "protected-2s")
                  for i in range(60)]
    config = media.KeyframeConfig(state_layout_threshold=1.0)
    selected = media._deduplicate_states(candidates, config)
    assert 3 <= len(selected) <= 5
    assert any(5 <= c.time < 10 for c in selected)
    assert any(10 <= c.time < 15 for c in selected)
    assert len({c.page_id for c in selected}) == 1


def test_repeated_content_is_reduced_before_budget(tmp_path):
    candidates = [media._Candidate(i, card(tmp_path / f"{i}.jpg", 80, variant=i % 3), "protected-2s",
                                   keep_reason="content-state", support_s=0.25, page_id=i)
                  for i in range(120)]
    config = media.KeyframeConfig()
    representatives = media._merge_repeated_content(candidates, config)
    assert len(representatives) == 3
    assert len(media._limit_candidates(representatives, 25, config=config)) == 3


def test_incremental_content_changes_do_not_all_drift_into_one_group(tmp_path):
    candidates = [media._Candidate(i / 4, card(tmp_path / f"growing-{i}.jpg", 80, variant=i / 30), "4fps")
                  for i in range(90)]
    groups = media._content_groups(candidates, media.KeyframeConfig())
    assert len(groups) > 3


def test_number_budget_is_hard_and_does_not_exempt_protected_frames(tmp_path):
    random = np.random.default_rng(40)
    candidates = []
    for i in range(100):
        path = tmp_path / f"{i}.png"
        Image.fromarray(random.integers(0, 256, (160, 96, 3), dtype=np.uint8)).save(path)
        candidates.append(media._Candidate(i, path, "protected-2s", keep_reason="short-state",
                                           support_s=0.25, page_id=i % 10))
    config = media.KeyframeConfig()
    selected = media._limit_candidates(candidates, config.max_frames, config=config)
    assert len(selected) == 15
    assert candidates[0] in selected


def test_short_card_is_not_displaced_by_a_long_wait_on_same_page(tmp_path):
    first = media._Candidate(0, card(tmp_path / "start.jpg", 80), "start-0s", page_id=0)
    result = media._Candidate(100, card(tmp_path / "end.jpg", 80), "task-end",
                              keep_reason="task-end-explicit", page_id=0)
    pool = [first]
    for i in range(8):
        pool.append(media._Candidate(i + 1, card(tmp_path / f"wait-{i}.jpg", 80, variant=0),
                                     "protected-2s", support_s=5, page_id=0))
    needed = media._Candidate(2.1, card(tmp_path / "needed.jpg", 180, variant=2),
                              "scene-0.02", support_s=0.1, page_id=1)
    pool.extend([needed, result])
    assert needed in media._limit_candidates(sorted(pool, key=lambda c: c.time), 3)


@pytest.mark.parametrize("limit", [2, 3, 15])
def test_end_positions_share_the_same_budget(tmp_path, limit):
    candidates = [media._Candidate(i, card(tmp_path / f"{i}.jpg", 30 + i * 20), "4fps", page_id=i)
                  for i in range(8)]
    candidates[-2].keep_reason = "task-end-auto"
    candidates[-1].keep_reason = "final-frame"
    selected = media._limit_candidates(candidates, limit)
    assert len(selected) <= limit
    assert candidates[0] in selected and candidates[-2] in selected


@pytest.mark.parametrize("limit", [5, 10, 15])
def test_long_animated_video_stays_within_limit_and_trace_explains_selection(tmp_path, limit):
    video = tmp_path / "animated.mp4"
    subprocess.run([media.resolve_ffmpeg_executable(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=120x200:rate=12:duration=40", "-c:v", "mpeg4", "-q:v", "3", str(video)],
                   check=True, timeout=30)
    with media.record_process(tmp_path / "process"):
        frames = media.extract_scene_keyframes(video, tmp_path / "final", task_start_time=0, task_end_time=38,
                                              max_frames=limit)
    assert 2 <= len(frames) <= limit
    trace = json.loads((tmp_path / "process/trace.json").read_text(encoding="utf-8"))
    assert len(trace["frames"]) > 100
    assert any(stage["id"] == "content" for stage in trace["stages"])
    budget = next(stage for stage in trace["stages"] if stage["id"] == "budget")
    assert len(budget["output_ids"]) <= limit
    assert any(event["status"] == "dropped" and event["related"] for event in budget["events"])
    metadata = json.loads((tmp_path / "final/keyframes.json").read_text(encoding="utf-8"))
    assert metadata["algorithm_version"] == "hybrid-state-exp-v4.3"
    assert metadata["config"]["max_frames"] == limit
    assert base_media.KeyframeConfig().max_frames == 20
