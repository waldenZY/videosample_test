"""提速必须保留严格核对、参考顺序和覆盖记录。"""
from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from videosample import media


def track(pixels, index=0, origin=0):
    return media._LocalTrack(index, pixels.copy(), origin, 1., 'local', index)


def test_original_position_skips_search_only_after_strict_rgb_check(monkeypatch):
    pixels = np.random.default_rng(12).integers(0, 256, (100, 80, 3), dtype=np.uint8)
    config = media.KeyframeConfig()
    slow = media._locate_local_content(track(pixels, origin=-20), pixels, replace(config, local_fast_alignment=False))
    work = media._LocalWork(config)
    monkeypatch.setattr(work, 'peak', lambda *args: pytest.fail('等高原位通过时不应搜索'))
    fast = media._locate_local_content(track(pixels, origin=-20), pixels, config, work=work)
    assert fast[0] == slow[0] == -20
    assert fast.method == 'original_position'
    assert work.counts['original_position_hits'] == 1
    assert work.counts['full_search_attempts'] == 0


def test_changed_local_text_cannot_pass_original_position():
    pixels = np.full((160, 100, 3), 245, dtype=np.uint8)
    pixels[30:130:8, 10:90] = 30
    changed = pixels.copy()
    changed[67:73, 40:62] = [200, 20, 60]
    config = media.KeyframeConfig()
    work = media._LocalWork(config)
    fast = media._locate_local_content(track(pixels), changed, config, work=work)
    slow = media._locate_local_content(track(pixels), changed, replace(config, local_fast_alignment=False))
    assert fast == slow
    assert work.counts['original_position_hits'] == 0
    assert work.counts['full_search_attempts'] == 1


def test_unequal_visible_heights_still_use_search():
    pixels = np.random.default_rng(7).integers(0, 256, (160, 80, 3), dtype=np.uint8)
    crop = pixels[23:123].copy()
    config = media.KeyframeConfig()
    work = media._LocalWork(config)
    found = media._locate_local_content(track(pixels), crop, config, work=work)
    slow = media._locate_local_content(track(pixels), crop, replace(config, local_fast_alignment=False))
    assert found == slow and found[0] == 23
    assert work.counts['original_position_hits'] == 0
    assert work.counts['full_search_attempts'] == 1


def test_exact_search_cache_reuses_peaks_but_still_checks_color(monkeypatch):
    config = media.KeyframeConfig(local_fast_alignment=False)
    work = media._LocalWork(config)
    pixels = np.random.default_rng(4).integers(0, 256, (80, 70, 3), dtype=np.uint8)
    cv = media._local_cv()
    gray = cv.cvtColor(pixels, cv.COLOR_RGB2GRAY)
    reference = track(pixels)
    assert media._locate_local_content(reference, pixels, config, work=work) is not None
    calls = work.counts['template_calls']
    assert media._locate_local_content(reference, pixels.copy(), config, work=work) is not None
    assert work.counts['template_calls'] == calls
    assert work.counts['template_cache_hits'] > 0
    # 完全相同的灰度输入可复用搜索；RGB 严格核对仍必须执行。
    checked = []
    monkeypatch.setattr(media, '_strict_region_equal', lambda left, right, config: checked.append(True) or False)
    assert media._locate_local_content(reference, pixels, config, crop_gray=gray, work=work) is None
    assert checked and work.counts['template_calls'] == calls
    work.close()
    assert not work.peaks


def test_expanding_reference_invalidates_prepared_gray_and_search_data():
    pixels = np.random.default_rng(2).integers(0, 256, (80, 64, 3), dtype=np.uint8)
    reference = track(pixels)
    media._locate_local_content(reference, pixels, media.KeyframeConfig(local_fast_alignment=False))
    assert reference.gray is not None and reference.search_data is not None
    assert media._extend_local_track(reference, pixels, 10)
    assert reference.gray is None and reference.search_data is None
    assert reference.pixels.shape[0] == 90


def test_parallel_comparisons_keep_reference_order_and_finish_before_mutation(monkeypatch):
    pixels = np.zeros((30, 20, 3), dtype=np.uint8)
    references = [track(pixels, i) for i in range(5)]
    running, finished = set(), []
    lock = threading.Lock()

    def locate(reference, *args):
        with lock:
            running.add(reference.index)
        time.sleep(.025 if reference.index == 1 else .003)
        with lock:
            running.remove(reference.index)
            finished.append(reference.index)
        return reference.index

    monkeypatch.setattr(media, '_locate_local_content', locate)
    work = media._LocalWork(media.KeyframeConfig(local_match_workers=2))
    generator = work.comparisons(references, pixels, None, None)
    first, value = next(generator)
    assert first.index == value == 0 and finished == [0]
    second, value = next(generator)
    assert second.index == value == 1  # 2 先算完也不能抢走 1 的优先级。
    assert 2 in finished and not running
    generator.close()
    work.close()


def test_failed_analysis_releases_workers_and_keeps_diagnostics(tmp_path, monkeypatch):
    instances = []
    original = media._LocalWork

    class Captured(original):
        def __init__(self, config):
            super().__init__(config)
            instances.append(self)

    completed = threading.Event()

    def fail(candidates, config, work):
        from concurrent.futures import ThreadPoolExecutor
        work.pool = ThreadPoolExecutor(max_workers=2)
        work.pool.submit(lambda: (time.sleep(.02), completed.set()))
        raise ValueError('expected')

    monkeypatch.setattr(media, '_LocalWork', Captured)
    monkeypatch.setattr(media, '_analyze_local_content_impl', fail)
    with media.record_process(tmp_path / 'process') as recorder:
        with pytest.raises(ValueError, match='expected'):
            media._analyze_local_content([], media.KeyframeConfig())
        assert recorder.local_analysis['input_frames'] == 0
        assert recorder.local_analysis['elapsed_s'] >= 0
    assert not instances[0].peaks
    assert completed.is_set()


@pytest.mark.parametrize('kwargs', [{'local_fast_alignment': 1}, {'local_search_reuse': 'yes'},
                                    {'local_match_workers': True}, {'local_match_workers': 0},
                                    {'local_match_workers': 5}])
def test_optimization_config_validation(kwargs):
    with pytest.raises(ValueError):
        media.KeyframeConfig(**kwargs)
