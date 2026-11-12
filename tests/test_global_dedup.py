import json

from PIL import Image, ImageDraw

from videosample import media


def frame(tmp_path, name, time, color, support=0, reason="content-state"):
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (96, 160), color).save(path)
    return media._Candidate(time, path, "4fps", keep_reason=reason, support_s=support)


def test_repeat_is_removed_across_other_pages_and_large_time_gap(tmp_path):
    start = frame(tmp_path, "start", 0, "white")
    card = frame(tmp_path, "card", 10, "black", support=3)
    app = frame(tmp_path, "app", 30, "red")
    repeated = frame(tmp_path, "repeated", 200, "black", support=0.25)
    frames = [start, card, app, repeated]
    with media.record_process(tmp_path / "process"):
        media._stage("collected", "测试候选", [], frames)
        kept, removed = media._final_deduplicate(frames, media.KeyframeConfig())
        media._stage("final", "全局去重", frames, kept)
    assert kept == [start, card, app] and removed == [repeated]
    trace = json.loads((tmp_path / "process/trace.json").read_text(encoding="utf-8"))
    event = trace["frames"][3]["history"][-1]
    assert event["related"] == [trace["frames"][1]["id"]]
    assert event["details"]["time_gap_s"] == 190


def test_tiny_encoding_difference_can_merge(tmp_path):
    frames = [frame(tmp_path, "start", 0, "white"), frame(tmp_path, "a", 10, (40, 40, 40), 2),
              frame(tmp_path, "other", 20, "blue"), frame(tmp_path, "b", 80, (41, 41, 41))]
    kept, removed = media._final_deduplicate(frames, media.KeyframeConfig())
    assert frames[1] in kept and frames[3] in removed


def test_local_text_difference_is_not_merged(tmp_path):
    start = frame(tmp_path, "start", 0, "white")
    candidates = [start]
    for index, text in enumerate(["123", "128"]):
        image = Image.new("RGB", (96, 160), "white")
        ImageDraw.Draw(image).text((20, 60), text, fill="black")
        path = tmp_path / f"text-{index}.png"
        image.save(path)
        candidates.append(media._Candidate(10 + index * 100, path, "4fps"))
    kept, removed = media._final_deduplicate(candidates, media.KeyframeConfig())
    assert kept == candidates and not removed


def test_similarity_does_not_chain_through_removed_frame(tmp_path):
    # 相邻亮度仅差 1，但两个端点差 2；只能保留高度相似关系成立的代表。
    config = media.KeyframeConfig(global_dedup_rms_threshold=1.5 / 255)
    start = frame(tmp_path, "start", 0, "white")
    a = frame(tmp_path, "a", 10, (40, 40, 40), 3)
    b = frame(tmp_path, "b", 50, (41, 41, 41), 2)
    c = frame(tmp_path, "c", 100, (42, 42, 42), 1)
    kept, removed = media._final_deduplicate([start, a, b, c], config)
    assert a in kept and c in kept and b in removed


def test_identical_endpoints_are_retained_and_explained(tmp_path):
    start = frame(tmp_path, "start", 0, "white")
    ordinary = frame(tmp_path, "ordinary", 20, "white")
    end = frame(tmp_path, "end", 40, "white", reason="task-end-explicit")
    with media.record_process(tmp_path / "process"):
        media._stage("collected", "测试候选", [], [start, ordinary, end])
        kept, removed = media._final_deduplicate([start, ordinary, end], media.KeyframeConfig())
        media._stage("final", "全局去重", [start, ordinary, end], kept)
    assert kept == [start, end] and removed == [ordinary]
    trace = json.loads((tmp_path / "process/trace.json").read_text(encoding="utf-8"))
    assert "保留任务首尾位置" in trace["frames"][2]["history"][-1]["reason"]
