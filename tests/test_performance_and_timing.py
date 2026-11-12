from pathlib import Path

import numpy as np
from PIL import Image

from videosample import media


def test_image_features_are_reused_only_within_one_extraction(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (96, 160), "white").save(path)

    @media._with_feature_cache
    def read():
        first = media._content_signature(path)
        assert media._content_signature(path) is first
        return first.copy()

    original = read()
    Image.new("RGB", (96, 160), "black").save(path)
    changed = read()
    assert original.mean() == 255 and changed.mean() == 0
    assert media._FEATURES.get() is None


def test_feature_cache_is_bounded_and_cleared_on_failure():
    loads = []

    @media._cached_feature
    def feature(path):
        loads.append(str(path))
        return object()

    @media._with_feature_cache
    def run():
        for i in range(600):
            feature(Path(str(i)))
        assert len(media._FEATURES.get()) == 512
        feature(Path("0"))
        assert len(loads) == 601
        raise ValueError("expected")

    try:
        run()
    except ValueError:
        pass
    assert media._FEATURES.get() is None


def test_budget_does_not_recalculate_nearest_retained_images(tmp_path, monkeypatch):
    candidates = []
    random = np.random.default_rng(42)
    for index in range(40):
        path = tmp_path / f"{index}.png"
        Image.fromarray(random.integers(0, 256, (160, 96, 3), dtype=np.uint8)).save(path)
        candidates.append(media._Candidate(index, path, "4fps", page_id=index % 6))
    calls = []
    original = media._content_difference

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(media, "_content_difference", counted)
    selected = media._limit_candidates(candidates, 10)
    assert len(selected) == 10
    assert len(calls) < len(candidates) * 10
