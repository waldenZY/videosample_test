import io
import json
import math
from dataclasses import asdict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from videosample import media, server, algorithms


def imported(client):
    payload = json.dumps({'id': 'same', 'query': '=1+1', 'video_path': 'video.mp4'}).encode()
    return client.post('/api/import', files={'file': ('测试.jsonl', payload)}).json()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(server.POOL, 'submit', lambda *a: None)
    with TestClient(server.app) as c:
        yield c


@pytest.mark.parametrize('field,value', [('experiment_max_frames', 1), ('experiment_max_frames', 2.5),
    ('experiment_max_frames', True), ('experiment_max_frames', None), ('experiment_max_frames', '20'),
    ('experiment_sample_fps', 0), ('experiment_sample_fps', -1), ('experiment_sample_fps', True),
    ('experiment_sample_fps', None), ('experiment_sample_fps', '3'), ('experiment_sample_fps', float('nan')),
    ('experiment_sample_fps', float('inf')), ('experiment_selection_scoring', 'content_weight')])
def test_invalid_parameters_rejected_before_scheduling(client, field, value):
    run = imported(client)
    response = client.post(f"/api/runs/{run['id']}/start", content=json.dumps({field: value}), headers={'Content-Type': 'application/json'})
    assert response.status_code == 422
    assert isinstance(response.json()['detail'], str)
    assert server.read_run(run['id'])['status'] == 'imported'


def test_custom_parameters_names_snapshots_and_worker(client, monkeypatch, tmp_path):
    run = imported(client)
    assert run['name'] == server.task_name('测试.jsonl', run['created_at'])
    assert run['created_at'].endswith('+08:00')
    started = client.post(f"/api/runs/{run['id']}/start", json={'run_mode': 'both', 'experiment_max_frames': 25, 'experiment_sample_fps': .5}).json()
    base = server.read_run(started['chain_ids']['base'])
    current = server.read_run(started['chain_ids']['current'])
    assert 'overrides' not in base['algorithm']
    assert current['algorithm']['overrides'] == {'max_frames': 25, 'sample_fps': .5}
    listed = client.get('/api/runs').json()[0]
    assert listed['chains'][1]['effective_config']['sample_fps'] == .5
    video = tmp_path / 'input.mp4'
    video.write_bytes(b'fixture')
    monkeypatch.setattr(server, 'prepared_video', lambda *a: video)
    monkeypatch.setattr(media, 'probe_duration', lambda *a: 30)
    calls = []
    monkeypatch.setattr(media, 'extract_scene_keyframes', lambda *a, **kw: calls.append(kw) or [])
    server.process_row(current['id'], 0)
    assert calls[0]['max_frames'] == 25 and calls[0]['sample_fps'] == .5
    assert calls[0]['selection_scoring'] == 'new_content_ratio'
    copied = client.post(f"/api/runs/{current['id']}/copy").json()
    assert copied['name'] == server.task_name('测试.jsonl', copied['created_at'])
    assert copied['experiment_max_frames'] == 15 and copied['experiment_sample_fps'] == 3
    assert server.read_run(current['id'])['algorithm']['overrides']['sample_fps'] == .5


@pytest.mark.parametrize('limit', [2, 10, 15, 25, 40])
def test_selector_respects_actual_cap_without_internal_15_clamp(tmp_path, limit):
    candidates = [media._Candidate(i, tmp_path / f'{i}.jpg', '3fps') for i in range(45)]
    candidates[-1].keep_reason = 'task-end-explicit'
    analysis = media._LocalAnalysis(candidates, representatives=list(candidates))
    for i, c in enumerate(candidates):
        key = f'r{i}:0'
        analysis.units[key] = {'weight': 1., 'kind': 'local', 'track': i}
        analysis.coverage[id(c)] = {key}
        analysis.quality[id(c)] = (100, 0, 1., 0)
        analysis.descriptors[id(c)] = np.full((2, 2), i / 45)
    selected = media._limit_candidates(candidates, limit, analysis=analysis,
        config=media.KeyframeConfig(max_frames=limit, scroll_dedup_enabled=False))
    assert len(selected) == limit
    assert candidates[0] in selected and candidates[-1] in selected
    assert analysis.summary['limit'] == limit and analysis.summary['stop_reason'] == 'limit_reached'


@pytest.mark.parametrize('value', [0, -1, True, float('nan'), float('inf'), '3'])
def test_core_fps_validation(value):
    with pytest.raises(ValueError):
        media.KeyframeConfig(sample_fps=value)


def test_single_xlsx_contains_sample_diagnostics_and_no_per_frame_sheet(client):
    run = imported(client)
    run['status'] = 'done'
    run['algorithm'] = algorithms.snapshot('current', max_frames=25, sample_fps=.5)
    run['rows'][0].update(status='done', frame_count=15, initial_frame_count=80,
        effective_config={**asdict(media.KeyframeConfig()), 'max_frames': 25, 'sample_fps': .5},
        timings_s={'extraction': 0, 'stages': {'local': 1.23}},
        selection={'stop_reason': 'covered', 'uncovered_count': 0, 'scroll_dedup': {'comparisons': 4, 'merged_frames': 2}},
        local_analysis={'input_frames': 30, 'elapsed_s': 1.23, 'original_position_hits': 0,
            'template_calls': 9, 'template_search_work_s': 1.1,
            'timing_breakdown': {'comparison_wall_s': .7, 'position_search_work_s': 1.2},
            'reference_filter': {'reference_filter_rejections': 12}, 'coarse_search': {'fallback_rate': .25},
            'slowest_frames': [{'frame_id': 'f00009', 'time': 4.5, 'elapsed_s': .3}]})
    run['rows'] += [{'index': 1, 'item': {'id': 'same', 'query': '坏样本'}, 'status': 'error', 'error': '取图失败'},
                    {'index': 2, 'item': {'id': 'same', 'query': '进行中'}, 'status': 'running'}]
    server.save_run(run)
    directory = server.run_directory(run['id'])
    (directory / 'items/00000/final').mkdir(parents=True)
    proc = directory / 'items/00000/process'
    proc.mkdir()
    (proc / 'trace.json').write_text(json.dumps({'stages': [{'id': 'local', 'title': '局部分析', 'input_ids': ['1', '2'], 'output_ids': ['1']}]}), 'utf8')
    before = (directory / 'run.json').read_bytes()
    response = client.get(f"/api/runs/{run['id']}/export.xlsx")
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('application/vnd.openxmlformats')
    assert "filename*=UTF-8''" in response.headers['content-disposition']
    wb = load_workbook(io.BytesIO(response.content))
    assert wb.sheetnames == ['批次概览', '样本统计']
    sheet = wb['样本统计']
    headers = [c.value for c in sheet[1]]
    values = [dict(zip(headers, row)) for row in sheet.iter_rows(min_row=2, values_only=True)]
    assert len(values) == 3
    assert values[0]['题目'] == '=1+1' and sheet.cell(2, headers.index('题目') + 1).data_type == 's'
    assert values[0]['张数上限'] == 25 and values[0]['每秒采样帧数'] == .5
    assert values[0]['抽帧合计（秒）'] == 0
    assert values[0]['局部·原位匹配次数'] == 0
    assert values[0]['参考快速排除·排除次数'] == 12
    assert values[0]['局部分析·输入张数'] == 2 and values[0]['局部分析·输出张数'] == 1
    assert values[0]['因上限停止'] is False
    assert values[0]['最终图片'] == str((directory / 'items/00000/final').resolve())
    assert 'f00009' in values[0]['最慢候选摘要（最多3张）']
    assert values[1]['错误'] == '取图失败' and values[1]['局部·原位匹配次数'] == '未记录'
    assert values[2]['状态'] == '运行中' and values[2]['最终图片'] == '未完成'
    assert not sheet._images and sheet.freeze_panes == 'D2'
    assert (directory / 'run.json').read_bytes() == before


def test_legacy_export_does_not_invent_current_configuration(client):
    run = imported(client)
    run.pop('algorithm')
    run['rows'][0].update(status='done', frame_count=3)
    server.save_run(run)
    response = client.get(f"/api/runs/{run['id']}/export.xlsx")
    wb = load_workbook(io.BytesIO(response.content))
    values = dict(zip([c.value for c in wb['样本统计'][1]], [c.value for c in wb['样本统计'][2]]))
    assert values['张数上限'] == '未记录' and values['每秒采样帧数'] == '未记录'
    assert values['评分策略'] == '未记录'
    assert client.get('/api/runs/invalid/export.xlsx').status_code == 404
