import json

import pytest
from fastapi.testclient import TestClient

from videosample import server


@pytest.mark.parametrize("mode,keys", [("base", ["base"]), ("current", ["current"]),
                                       ("both", ["base", "current"])])
def test_only_selected_chains_are_submitted_and_mode_survives_copy(tmp_path, monkeypatch, mode, keys):
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
    jobs = []

    class RecordingPool:
        def submit(self, fn, *args):
            jobs.append(args)

    monkeypatch.setattr(server, "POOL", RecordingPool())
    payload = json.dumps({"id": "q1", "query": "测试选择", "video_path": "video.mp4"}).encode()
    with TestClient(server.app) as client:
        run = client.post("/api/import", files={"file": ("input.jsonl", payload)}).json()
        identity = run["id"]
        assert client.post(f"/api/runs/{identity}/start", json={"run_mode": "invalid"}).status_code == 422
        assert server.read_run(identity)["status"] == "imported"
        started = client.post(f"/api/runs/{identity}/start", json={"run_mode": mode}).json()
        assert list(started["chain_ids"]) == keys
        assert len(jobs) == len(keys)
        assert [server.read_run(job[0])["algorithm"]["key"] for job in jobs] == keys
        assert started["algorithm"]["key"] == keys[0]
        assert started["run_mode"] == mode
        assert len(client.get("/api/runs").json()) == 1
        copied = client.post(f"/api/runs/{identity}/copy").json()
        assert copied["run_mode"] == mode and "chain_ids" not in copied
        assert copied["algorithm"]["key"] == keys[0]


@pytest.mark.parametrize("limit", [5, 10])
def test_experiment_limit_is_saved_only_on_experiment_and_survives_rerun(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(server.POOL, "submit", lambda *args: None)
    payload = json.dumps({"id": "q1", "query": "数量测试", "video_path": "video.mp4"}).encode()
    with TestClient(server.app) as client:
        imported = client.post("/api/import", files={"file": ("input.jsonl", payload)}).json()
        assert imported["experiment_max_frames"] == 10
        identity = imported["id"]
        assert client.post(f"/api/runs/{identity}/start",
                           json={"experiment_max_frames": 25}).status_code == 422
        started = client.post(f"/api/runs/{identity}/start", json={"run_mode": "both",
                              "experiment_max_frames": limit}).json()
        base = server.read_run(started["chain_ids"]["base"])
        experiment = server.read_run(started["chain_ids"]["current"])
        assert "overrides" not in base["algorithm"]
        assert base["algorithm"]["defaults"]["max_frames"] == 20
        assert experiment["algorithm"]["overrides"] == {"max_frames": limit}
        assert started["chains"][1]["frame_limit"] == limit
        copied = client.post(f"/api/runs/{experiment['id']}/copy").json()
        assert copied["experiment_max_frames"] == limit
