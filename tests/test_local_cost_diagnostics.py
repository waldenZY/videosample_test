"""定位记录不改变选图；并行累计时间与实际经过时间分别展示。"""
import json

import numpy as np
import pytest

from videosample import media


def test_comparison_timer_excludes_parent_work_between_results(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(media.time, 'perf_counter', lambda: clock[0])

    def locate(*args):
        clock[0] += 2
        return None

    monkeypatch.setattr(media, '_locate_local_content', locate)
    work = media._LocalWork(media.KeyframeConfig(local_match_workers=1))
    results = work.comparisons([object(), object()], None, None, None)
    next(results)
    clock[0] += 50  # 主线程在两个结果之间处理其他事情，不能算作比较时间。
    next(results)
    assert work.counts['comparison_wall_s'] == 4
    work.close()


def test_width_rejection_does_not_count_as_search_or_strict_check():
    pixels = np.zeros((80, 40, 3), dtype=np.uint8)
    reference = media._LocalTrack(0, pixels, 0, 1., 'local', 0)
    work = media._LocalWork(media.KeyframeConfig())
    assert media._locate_local_content(reference, pixels[:, :-1], work.config, work=work) is None
    assert work.counts['width_mismatches'] == work.counts['failed_pairs'] == 1
    assert work.counts['strict_checks'] == work.counts['full_search_attempts'] == 0


def test_top_frames_are_bounded_linked_and_keep_counters_separate(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(media.time, 'perf_counter', lambda: clock[0])
    pixels = np.zeros((100, 60, 3), dtype=np.uint8)
    work = media._LocalWork(media.KeyframeConfig())
    with media.record_process(tmp_path / 'process') as recorder:
        for index, elapsed in enumerate([2, 5, 1, 3, 4]):
            candidate = media._Candidate(index, tmp_path / 'not-read.png', 'test')
            identity = recorder.identify(candidate, copy_image=False)
            with work.frame(candidate, index, pixels):
                clock[0] += elapsed
                work.add('pair_attempts', index + 1)
                work.frame_info['regions'].append([40, 80])
                work.reference_size(np.zeros((150 + index, 30, 3), dtype=np.uint8))
            if index == 1:
                second_identity = identity
        assert [entry['candidate_index'] for entry in work.slowest_frames] == [1, 4, 3]
        assert work.slowest_frames[0]['frame_id'] == second_identity
        assert work.slowest_frames[0]['pair_attempts'] == 2
        assert work.slowest_frames[0]['regions'] == [[40, 80]]
        assert work.slowest_frames[0]['source_size'] == [60, 100]
        assert work.slowest_frames[0]['largest_reference_size'] == [30, 151]
    assert work.frame_counts is work.frame_info is None
    json.dumps(work.slowest_frames)


def test_failed_frame_retains_partial_diagnostics_and_resets_active_frame(tmp_path):
    work = media._LocalWork(media.KeyframeConfig())
    candidate = media._Candidate(12.3, tmp_path / 'not-read.png', 'test')
    with pytest.raises(ValueError, match='expected'):
        with work.frame(candidate, 3, None):
            work.add('full_search_attempts', 2)
            raise ValueError('expected')
    assert work.slowest_frames[0]['completed'] is False
    assert work.slowest_frames[0]['full_search_attempts'] == 2
    assert work.frame_counts is work.frame_info is None


def test_breakdown_has_nonoverlapping_wall_parts_and_keeps_worker_time_separate(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(media.time, 'perf_counter', lambda: clock[0])

    def analyze(candidates, config, work):
        work.add('comparison_wall_s', 4.)
        work.add('background_strict_wall_s', 1.)
        work.add('position_search_work_s', 5.)
        work.add('strict_check_work_s', 3.)
        clock[0] = 10.
        return media._LocalAnalysis(candidates)

    monkeypatch.setattr(media, '_analyze_local_content_impl', analyze)
    performance = media._analyze_local_content([], media.KeyframeConfig()).performance
    parts = performance['timing_breakdown']
    assert parts == {'comparison_wall_s': 4., 'background_strict_wall_s': 1.,
                     'other_preparation_wall_s': 5., 'position_search_work_s': 5., 'strict_check_work_s': 3.}
    assert sum(parts[key] for key in ('comparison_wall_s', 'background_strict_wall_s',
                                     'other_preparation_wall_s')) == performance['elapsed_s']
    assert performance['completed'] is True
