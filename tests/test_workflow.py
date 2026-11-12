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

from videosample import media, server
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
def test_recording_preserves_original_results(tmp_path, video, baseline, kwargs):
    original_dir, experiment_dir = tmp_path / "original", tmp_path / "experiment"
    original = baseline.extract_scene_keyframes(video, original_dir, max_edge=120, **kwargs)
    with media.record_process(tmp_path / "process"):
        experiment = media.extract_scene_keyframes(video, experiment_dir, max_edge=120, **kwargs)
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
        assert (tmp_path / "runs" / identity / "items/00000/video/video.mp4").read_bytes() == payload
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
            if run["status"] == "done":
                break
            time.sleep(0.1)
        assert run["status"] == "done", run
        assert [row["status"] for row in run["rows"]] == ["done", "error"]
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
