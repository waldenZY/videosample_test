"""固定布局的计时遮挡贯穿比较入口，原始图片仍可查看。"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from videosample import media


def picture(path, timer=False, changed=False):
    image = Image.new('RGB', (720, 1554), '#151515')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 180, 690, 370), fill='#253d54')
    draw.rectangle((40, 210, 250, 230), fill='white')
    draw.rectangle((600 if changed else 540, 270, 635 if changed else 575, 305), fill='white')
    draw.rectangle((470, 25, 500 if timer else 515, 50), fill='white')
    image.save(path)
    return path


def test_all_feature_entries_ignore_timer_but_keep_other_changes(tmp_path):
    a, b, c = [picture(tmp_path / f'{i}.png', timer=i > 0, changed=i == 2) for i in range(3)]
    original = [p.read_bytes() for p in (a, b, c)]
    for feature in (media._signature, media._layout_signature,
                    media._assistant_shell_signature, media._content_signature):
        np.testing.assert_array_equal(feature(a), feature(b))
    assert not np.array_equal(media._content_signature(b), media._content_signature(c))
    assert [p.read_bytes() for p in (a, b, c)] == original
    candidates = [media._Candidate(i, p, '3fps') for i, p in enumerate((a, b))]
    assert media._limit_candidates(candidates, 15) == [candidates[0]]


def test_mask_setting_reaches_parallel_workers_and_is_released(tmp_path):
    path = picture(tmp_path / 'timer.png')

    @media._with_feature_cache
    def calculate(video, directory, config):
        candidates = [media._Candidate(i, path, '3fps') for i in range(8)]
        media._prewarm_features(candidates, media._content_signature)
        return media._content_signature(path).copy()

    masked = calculate(None, None, media.KeyframeConfig())
    unmasked = calculate(None, None, media.KeyframeConfig(analysis_mask_box=None))
    assert not np.array_equal(masked, unmasked)
    np.testing.assert_array_equal(masked, media._content_signature(path))


def test_scene_detection_applies_mask_without_changing_capture_filter(monkeypatch):
    commands = []
    class Result:
        stderr = ''
    monkeypatch.setattr(media, 'resolve_ffmpeg_executable', lambda: 'ffmpeg')
    monkeypatch.setattr(media.subprocess, 'run', lambda command, **kwargs: commands.append(command) or Result())
    media.scene_change_times('video.mp4')
    filters = commands[0][commands[0].index('-vf') + 1]
    assert filters.startswith('drawbox=') and ',select=' in filters


@pytest.mark.parametrize('box', [(1, 0, 0, 1), (0, 0, 2, 1), (0, 0, float('nan'), 1)])
def test_bad_mask_settings_are_rejected(box):
    with pytest.raises(ValueError):
        media.KeyframeConfig(analysis_mask_box=box)
