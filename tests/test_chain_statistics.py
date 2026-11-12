from fastapi.testclient import TestClient

from videosample import server


def test_statistics_exclude_unsuccessful_rows_and_do_not_treat_missing_as_zero():
    run = {"rows": [
        {"status": "done", "elapsed_s": 10, "frame_count": 20, "stage_counts": {"collected": 100},
         "timings_s": {"extraction": 8}},
        {"status": "done", "elapsed_s": 20, "frame_count": 10, "initial_frame_count": 80,
         "timings_s": {"extraction": 15}},
        {"status": "done"},
        {"status": "error", "elapsed_s": 100, "frame_count": 500, "initial_frame_count": 1000},
        {"status": "interrupted", "elapsed_s": 80},
        {"status": "skipped"}, {"status": "running", "elapsed_s": 30}]}
    stats = server.chain_statistics(run)
    assert stats["selected_samples"] == 6
    assert stats["successful_samples"] == 3 and stats["failed_samples"] == 2
    assert stats["sample_time_s"] == {"samples": 2, "total": 30, "mean": 15}
    assert stats["extraction_time_s"] == {"samples": 2, "total": 23, "mean": 11.5}
    assert stats["initial_frames"] == {"samples": 2, "total": 180, "mean": 90}
    assert stats["final_frames"] == {"samples": 2, "total": 30, "mean": 15}


def test_empty_and_zero_statistics():
    assert server.chain_statistics({"rows": []})["final_frames"] == {"samples": 0, "total": None, "mean": None}
    stats = server.chain_statistics({"rows": [{"status": "done", "elapsed_s": 0, "frame_count": 0,
                                              "initial_frame_count": 0}]})
    assert stats["initial_frames"] == {"samples": 1, "total": 0, "mean": 0}


def test_batch_api_returns_independent_statistics_for_each_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
    ids = {"base": "a" * 32, "current": "b" * 32}
    for key, identity in ids.items():
        run = {"id": identity, "group_id": ids["base"], "chain_ids": ids, "status": "done", "name": "test",
               "version": key, "rows": [{"status": "done", "elapsed_s": 30 if key == "base" else 8,
                                          "frame_count": 20 if key == "base" else 25,
                                          "stage_counts": {"collected": 100 if key == "base" else 400}}]}
        server.save_run(run)
    with TestClient(server.app) as client:
        run = client.get(f"/api/runs/{ids['base']}").json()
        base, experiment = [chain["statistics"] for chain in run["chains"]]
        assert run["statistics"] == base
        assert base["sample_time_s"]["mean"] == 30
        assert experiment["sample_time_s"]["mean"] == 8
        assert base["initial_frames"]["total"] == 100
        assert experiment["initial_frames"]["total"] == 400
        assert experiment["final_frames"]["total"] == 25
