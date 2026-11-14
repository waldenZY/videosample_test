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


def test_local_search_statistics_exclude_old_missing_and_failed_samples():
    stats = server.chain_statistics({'rows': [
        {'status': 'done', 'local_analysis': {'elapsed_s': 2, 'input_frames': 20,
             'original_position_hits': 4, 'full_search_attempts': 10, 'template_calls': 25}},
        {'status': 'done'},  # 历史记录不按 0 秒计入。
        {'status': 'done', 'local_analysis': {'elapsed_s': 6, 'input_frames': 80,
             'original_position_hits': 8, 'full_search_attempts': 30, 'template_calls': 100}},
        {'status': 'error', 'local_analysis': {'elapsed_s': 99}}]})['local_analysis']
    assert stats['samples'] == 2
    assert stats['elapsed_s'] == {'samples': 2, 'total': 8, 'mean': 4}
    assert stats['full_search_attempts']['mean'] == 20
    assert server.chain_statistics({'rows': [{'status': 'done'}]})['local_analysis']['elapsed_s']['mean'] is None


def test_output_distribution_and_hit_limit_use_actual_stop_reason():
    rows = []
    for count, reason, seconds in [(4, 'covered', 3), (15, 'limit_reached', 26),
                                   (14, 'limit_reached', 12), (8, 'no_reliable_addition', 5)]:
        rows.append({'status': 'done', 'frame_count': count, 'selection': {'stop_reason': reason},
                     'timings_s': {'extraction': seconds}})
    rows += [{'status': 'done', 'frame_count': 10},
             {'status': 'error', 'frame_count': 15, 'selection': {'stop_reason': 'limit_reached'}}]
    stats = server.chain_statistics({'rows': rows})
    assert stats['output_distribution'] == {'samples': 5, 'counts':
            {'1-5': 1, '6-10': 2, '11-14': 1, '15': 1, 'other': 0}}
    assert stats['adaptive_selection']['samples'] == 4
    assert stats['adaptive_selection']['hit_limit_count'] == 2
    assert stats['adaptive_selection']['hit_limit_rate'] == .5
    assert stats['extraction_max_s'] == 26


def test_exactly_15_with_all_content_covered_is_not_limit_stopped():
    stats = server.chain_statistics({'rows': [{'status': 'done', 'frame_count': 15,
                                              'selection': {'stop_reason': 'covered', 'hit_limit': False}}]})
    assert stats['output_distribution']['counts']['15'] == 1
    assert stats['adaptive_selection']['hit_limit_rate'] == 0


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
