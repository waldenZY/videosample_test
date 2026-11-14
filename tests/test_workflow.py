import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from videosample import algorithms, media, server
from videosample.versions import base_media
from videosample.preparation import _operation_timing, resolve_operation_video_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline():
    spec = importlib.util.spec_from_file_location("original_media", ROOT / "reference" / "media.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "flow.mp4"
    subprocess.run([media.resolve_ffmpeg_executable(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=white:s=120x200:r=10:d=16", "-vf",
                    "drawbox=x=15:y=50:w=90:h=60:color=black:t=fill:enable='gte(t,8)*lt(t,10)',"
                    "drawbox=x=0:y=0:w=120:h=200:color=gray:t=fill:enable='gte(t,14)'",
                    "-c:v", "mpeg4", "-q:v", "2", str(path)], check=True, timeout=30)
    return path


@pytest.mark.parametrize("kwargs", [{"task_end_time": 13}, {}, {"task_end_time": 13, "sample_fps": 10}])
def test_base_recording_preserves_original_results(tmp_path, video, baseline, kwargs):
    original_dir, experiment_dir = tmp_path / "original", tmp_path / "experiment"
    original = baseline.extract_scene_keyframes(video, original_dir, max_edge=120, **kwargs)
    with base_media.record_process(tmp_path / "process"):
        experiment = base_media.extract_scene_keyframes(video, experiment_dir, max_edge=120, **kwargs)
    assert len(experiment) == len(original)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in experiment] == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in original]
    assert json.loads((original_dir / "keyframes.json").read_text(encoding="utf-8")) == json.loads(
        (experiment_dir / "keyframes.json").read_text(encoding="utf-8"))
    trace = json.loads((tmp_path / "process" / "trace.json").read_text(encoding="utf-8"))
    assert len(trace["stages"]) == 7
    assert len(trace["stages"][-1]["output_ids"]) == len(experiment)
    known = {frame["id"] for frame in trace["frames"]}
    for frame in trace["frames"]:
        assert (tmp_path / "process" / frame["image"]).is_file()
        assert frame["history"]
        for event in frame["history"]:
            assert event["reason"]
            assert set(event["related"]) <= known
    assert any(stage["removed_ids"] for stage in trace["stages"])


def test_plain_extraction_has_no_process_outputs(tmp_path, video):
    out = tmp_path / "plain"
    media.extract_scene_keyframes(video, out, task_end_time=13)
    assert not (out / "trace.json").exists()
    assert not (out / "candidates").exists()


def test_adaptive_selection_is_output_without_another_filter(tmp_path, video):
    with media.record_process(tmp_path / 'process') as recorder:
        frames = media.extract_scene_keyframes(video, tmp_path / 'final', task_end_time=13)
    assert recorder.stages[-1]['id'] == 'budget'
    assert len(recorder.stages[-1]['output_ids']) == len(frames)
    assert all(stage['id'] != 'final' for stage in recorder.stages)
    metadata = json.loads((tmp_path / 'final/keyframes.json').read_text(encoding='utf-8'))
    assert metadata['selection']['selected_count'] == len(frames)
    assert 'final_dedup_removed' not in metadata
    assert metadata['config']['analysis_mask_box'] == list(media.DEFAULT_ANALYSIS_MASK_BOX)
    for path, identity in zip(frames, recorder.stages[-1]['output_ids']):
        assert path.read_bytes() == (tmp_path / 'process/candidates' / f'{identity}.jpg').read_bytes()


def test_merged_candidates_are_not_restored_by_later_stages(tmp_path, video, monkeypatch):
    # 故意让内容合并只留下首尾：后面不能重新引入中间的不同画面。
    monkeypatch.setattr(media, '_merge_repeated_content', lambda candidates, config: [candidates[0], candidates[-1]])
    with media.record_process(tmp_path / 'process') as recorder:
        media.extract_scene_keyframes(video, tmp_path / 'final', task_end_time=13)
    stages = {stage['id']: stage for stage in recorder.stages}
    assert 'review' not in stages
    assert 'merged_content_review' not in recorder.selection
    expected = stages['content']['output_ids']
    assert len(expected) == 2
    assert stages['local']['input_ids'] == expected
    assert set(stages['local']['output_ids']) <= set(expected)
    assert stages['budget']['input_ids'] == stages['local']['output_ids']
    assert set(stages['budget']['output_ids']) <= set(expected)


def test_experiment_default_sampling_has_third_second_spacing(tmp_path, video):
    config = media.KeyframeConfig(task_start_time=0, task_end_time=13, protected_sample_interval=0)
    assert config.sample_fps == 3
    _, candidates = media._extract_candidates(video, tmp_path / 'candidates', config, 16)
    samples = [c for c in candidates if c.source == '3fps']
    assert len(samples) > 10
    assert samples[0].time == pytest.approx(1 / 3)
    assert all(abs(b.time - a.time - 1 / 3) < 1e-9 for a, b in zip(samples, samples[1:]))


def test_base_defaults_and_output_match_original(tmp_path, video, baseline):
    assert asdict(base_media.KeyframeConfig()) == asdict(baseline.KeyframeConfig())
    original = baseline.extract_scene_keyframes(video, tmp_path / "old", task_end_time=13)
    with base_media.record_process(tmp_path / "base-process"):
        frozen = base_media.extract_scene_keyframes(video, tmp_path / "base", task_end_time=13)
    assert [p.read_bytes() for p in original] == [p.read_bytes() for p in frozen]
    assert json.loads((tmp_path / "old/keyframes.json").read_text(encoding="utf-8")) == json.loads(
        (tmp_path / "base/keyframes.json").read_text(encoding="utf-8"))


def test_base_stays_separate_from_experiment(monkeypatch):
    expected = algorithms.snapshot("base")
    monkeypatch.setattr(media, "KEYFRAME_ALGORITHM_VERSION", "experiment-test")
    assert algorithms.snapshot("base") == expected
    assert algorithms.snapshot("current")["algorithm_version"] == "experiment-test"


def test_out_of_range_times_keep_auto_eval_fallback():
    supplied, _, warnings, ignored = _operation_timing({"task_end_time": 50}, 16)
    assert supplied == {} and ignored == {"task_end_time"} and warnings


@pytest.mark.parametrize("relative", [False, True])
def test_server_video_outside_project_needs_no_allowlist(tmp_path, monkeypatch, relative):
    project = tmp_path / "project"
    project.mkdir()
    video = tmp_path / "videos" / "outside.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    monkeypatch.delenv("OPERATION_VIDEO_ROOTS", raising=False)
    raw = "../videos/outside.mp4" if relative else str(video)
    assert resolve_operation_video_path(raw, base_dir=project) == video.resolve()


def test_relative_video_roots_still_find_external_files(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    video = tmp_path / "videos" / "outside.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    monkeypatch.setenv("OPERATION_VIDEO_ROOTS", str(video.parent))
    assert resolve_operation_video_path("outside.mp4", base_dir=project) == video.resolve()


def test_remote_video_download_and_batch_record(tmp_path, monkeypatch, video):
    payload = video.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/video.mp4")
                self.end_headers()
                return
            body = b"<html>not a video</html>" if self.path == "/bad" else payload
            self.send_response(200)
            self.send_header("Content-Type", "text/html" if self.path == "/bad" else "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
    identity = "1" * 32
    url = f"http://127.0.0.1:{httpd.server_port}"
    run = {"id": identity, "status": "running", "rows": [
        {"status": "queued", "item": {"video_path": url + "/redirect", "task_end_time": 13}},
        {"status": "queued", "item": {"video_path": url + "/bad"}}]}
    try:
        server.save_run(run)
        server.process_row(identity, 0)
        server.process_row(identity, 1)
        result = server.read_run(identity)
        assert [row["status"] for row in result["rows"]] == ["done", "error"]
        assert result["rows"][0]["video_sha256"] == hashlib.sha256(payload).hexdigest()
        assert (tmp_path / "runs" / identity / "inputs/00000/video.mp4").read_bytes() == payload
        assert "不是视频" in result["rows"][1]["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_xlsx_batch_api(tmp_path, monkeypatch, video):
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
    monkeypatch.delenv("OPERATION_VIDEO_ROOTS", raising=False)
    table = io.BytesIO()
    with pd.ExcelWriter(table, engine="openpyxl") as writer:
        pd.DataFrame([{"id": "other", "query": "其他页", "video_path": "wrong.mp4"}]).to_excel(writer, sheet_name="其他", index=False)
        pd.DataFrame([{"id": "ok", "query": "设置闹钟", "video_path": str(video), "结束时间": 13},
                      {"id": "missing", "query": "坏视频单题失败", "video_path": str(tmp_path / "missing.mp4")}]).to_excel(writer, sheet_name="合并数据", index=False)
    with TestClient(server.app) as client:
        response = client.post("/api/import", files={"file": ("cases.xlsx", table.getvalue())})
        assert response.status_code == 200, response.text
        run = response.json()
        assert run["sheet"] == "合并数据" and len(run["rows"]) == 2
        assert run["rows"][0]["item"]["task_end_time"] == 13
        identity = run["id"]
        assert client.post(f"/api/runs/{identity}/start", json={}).status_code == 200
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            run = client.get(f"/api/runs/{identity}").json()
            if run["status"] == "done" and all(c["status"] == "done" for c in run["chains"]):
                break
            time.sleep(0.1)
        assert run["status"] == "done", run
        assert [row["status"] for row in run["rows"]] == ["done", "error"]
        assert run["rows"][0]["timings_s"]["extraction"] >= 0
        assert run["rows"][1]["timings_s"]["video_prepare"] >= 0
        detail = client.get(f"/api/runs/{identity}/items/0/trace").json()
        assert len(detail["stages"]) == 7
        frame_id = detail["frames"][0]["id"]
        assert client.get(f"/api/runs/{identity}/items/0/frames/{frame_id}?thumb=true").status_code == 200
        assert client.get(f"/api/runs/{identity}/items/0/video").status_code == 200
        assert client.get(f"/api/runs/{identity}/items/0/frames/bad").status_code == 404
        assert client.get("/api/runs/not-a-run").status_code == 404
        assert client.post(f"/api/runs/{identity}/start", json={}).status_code == 409
        copied = client.post(f"/api/runs/{identity}/copy").json()
        assert copied["id"] != identity and copied["status"] == "imported"
    run["status"] = "running"
    run["rows"][0]["status"] = "running"
    server.save_run(run)
    with TestClient(server.app) as client:
        recovered = client.get(f"/api/runs/{identity}").json()
        assert recovered["status"] == "interrupted"
        assert recovered["rows"][0]["status"] == "interrupted"


@pytest.mark.parametrize("limit", [5, 10, 15])
def test_paired_chains_have_independent_steps(tmp_path, monkeypatch, video, limit):
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")

    class InlinePool:
        def submit(self, fn, *args):
            fn(*args)

    monkeypatch.setattr(server, "POOL", InlinePool())
    extract = media.extract_scene_keyframes

    def experiment(*args, **kwargs):
        frames = extract(*args, **kwargs)
        recorder = media._PROCESS.get()
        final = recorder.stages[-1]
        final["id"], final["title"] = "experiment_custom", "实验自定义步骤"
        recorder.stages[:] = [final]
        return frames

    monkeypatch.setattr(media, "extract_scene_keyframes", experiment)
    payload = json.dumps({"id": "same", "query": "同一题两条链路", "video_path": str(video),
                          "task_end_time": 13}, ensure_ascii=False).encode("utf-8")
    with TestClient(server.app) as client:
        base = client.post("/api/import", files={"file": ("input.jsonl", payload)}).json()
        base = client.post(f"/api/runs/{base['id']}/start", json={"experiment_max_frames": limit}).json()
        assert len(base["chains"]) == 2
        other_id = base["chain_ids"]["current"]
        other = client.get(f"/api/runs/{other_id}").json()
        assert base["rows"][0]["status"] == other["rows"][0]["status"] == "done"
        assert base["algorithm"]["key"] == "base" and other["algorithm"]["key"] == "current"
        assert base["rows"][0]["effective_config"]["max_frames"] == 20
        assert other["rows"][0]["effective_config"]["max_frames"] == limit
        assert other['rows'][0]['selection']['policy'] == 'adaptive-scroll-v1'
        assert other['rows'][0]['selection']['selected_count'] == other['rows'][0]['frame_count']
        trace = client.get(f"/api/runs/{other_id}/items/0/trace").json()
        assert trace['selection'] == other['rows'][0]['selection']
        assert base["rows"][0]["stages"][-1]["id"] == "final"
        assert other["rows"][0]["stages"] == [{"id": "experiment_custom", "title": "实验自定义步骤"}]
        assert (tmp_path / "runs" / base["id"] / "algorithm.py").is_file()
        assert (tmp_path / "runs" / other_id / "algorithm.py").is_file()
        assert len(client.get("/api/runs").json()) == 1
        assert client.get("/api/compare?left=a&right=b").status_code == 404
        base_trace = client.get(f"/api/runs/{base['id']}/items/0/trace").json()
        other_trace = client.get(f"/api/runs/{other_id}/items/0/trace").json()
        assert len(base_trace["stages"]) == 7 and len(other_trace["stages"]) == 1
        assert base["rows"][0]["video_sha256"] == other["rows"][0]["video_sha256"]
        assert base["rows"][0]["initial_frame_count"] == base["rows"][0]["stage_counts"]["collected"]
        assert other["rows"][0]["initial_frame_count"] >= other["rows"][0]["frame_count"]
        assert "stages" not in base["rows"][0]["timings_s"]
        assert "stages" in other["rows"][0]["timings_s"]
        assert "process_io" in other["rows"][0]["timings_s"]["stages"]
        copied = client.post(f"/api/runs/{other_id}/copy").json()
        assert "chain_ids" not in copied and "group_id" not in copied
        copied = client.post(f"/api/runs/{copied['id']}/start", json={}).json()
        assert len(copied["chains"]) == 2 and copied["chain_ids"] != base["chain_ids"]


def test_both_chains_start_and_experiment_failure_is_isolated(tmp_path, monkeypatch, video):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(server, "POOL", pool)
    barrier = threading.Barrier(2, timeout=5)
    original_base = base_media.extract_scene_keyframes

    def base(*args, **kwargs):
        barrier.wait()
        return original_base(*args, **kwargs)

    def experiment(*args, **kwargs):
        barrier.wait()
        raise ValueError("实验链路测试失败")

    monkeypatch.setattr(base_media, "extract_scene_keyframes", base)
    monkeypatch.setattr(media, "extract_scene_keyframes", experiment)
    downloads = []

    def download(url, directory):
        downloads.append(url)
        time.sleep(0.03)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "video.mp4"
        path.write_bytes(video.read_bytes())
        return path

    monkeypatch.setattr(server, "download_remote_video", download)
    payload = json.dumps({"id": "same", "query": "同时运行", "video_path": "https://example.test/video.mp4",
                          "task_end_time": 13}, ensure_ascii=False).encode("utf-8")
    try:
        with TestClient(server.app) as client:
            run = client.post("/api/import", files={"file": ("input.jsonl", payload)}).json()
            run = client.post(f"/api/runs/{run['id']}/start", json={}).json()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                run = client.get(f"/api/runs/{run['id']}").json()
                if all(c["status"] == "done" for c in run["chains"]):
                    break
                time.sleep(0.05)
            other = client.get(f"/api/runs/{run['chain_ids']['current']}").json()
            assert run["rows"][0]["status"] == "done"
            assert other["rows"][0]["status"] == "error"
            assert "实验链路测试失败" in other["rows"][0]["error"]
            assert downloads == ["https://example.test/video.mp4"]
            assert run["rows"][0]["video_sha256"] == other["rows"][0]["video_sha256"]
    finally:
        pool.shutdown(wait=True)
