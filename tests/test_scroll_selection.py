"""录屏计时、固定工具栏与滚动正文的独立回归，不包含业务素材。"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from videosample import media


def screenshot(tmp_path, name, time, offset=0, value='123', dark=True, timer='17', toggle=False):
    bg, ink = ('#101010', 'white') if dark else ('#eeeeee', '#111111')
    image = Image.new('RGB', (720, 1000), bg)
    draw = ImageDraw.Draw(image)
    font_path = next((p for p in (Path('C:/Windows/Fonts/arial.ttf'),
                                 Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')) if p.exists()), None)
    if font_path is None:
        pytest.skip('需要可复现的 TrueType 字体')
    font = ImageFont.truetype(str(font_path), 25)
    draw.ellipse((170, 10, 195, 35), fill='#ed3434')
    draw.text((465, 8), '00:' + timer, font=font, fill=ink)
    draw.text((25, 65), 'FIXED HEADER', font=font, fill=ink)
    for i in range(6):
        text = f'Content {i}: ' + (value if i == 2 else f'number {i * 19}')
        draw.text((30, 220 + i * 65 + offset), text, font=font, fill=ink)
    draw.rounded_rectangle((530, 170 + offset, 640, 215 + offset), radius=20,
                           fill='#2584ed' if toggle else '#777777')
    draw.ellipse((600 if toggle else 535, 175 + offset,
                  635 if toggle else 570, 210 + offset), fill=ink)
    draw.text((25, 920), 'FIXED INPUT TOOLBAR', font=font, fill=ink)
    path = tmp_path / f'{name}.png'
    image.save(path)
    return media._Candidate(time, path, '4fps', support_s=.25)


def select(candidates):
    config = media.KeyframeConfig()
    analysis = media._analyze_local_content(candidates, config)
    selected = media._limit_candidates(analysis.representatives, 15, config=config, analysis=analysis)
    return selected, analysis


def test_recording_timer_does_not_hide_other_status_or_modify_output(tmp_path):
    a = screenshot(tmp_path, 'a', 0)
    b = screenshot(tmp_path, 'b', 1, timer='18')
    config = media.KeyframeConfig()
    raw = np.array(Image.open(a.path)); saved = raw.copy()
    left, boxes = media._analysis_pixels(raw, config)
    right, _ = media._analysis_pixels(np.array(Image.open(b.path)), config)
    assert boxes and np.array_equal(raw, saved)
    assert np.array_equal(left, right)
    assert media._analysis_pixels(raw, media.KeyframeConfig(local_recording_timer_mask=False))[1] == []
    # 不把整个状态栏都屏蔽，右侧状态图标变化仍可见。
    changed = raw.copy(); changed[15:30, 660:680] = (0, 255, 0)
    result, _ = media._analysis_pixels(changed, config)
    assert not np.array_equal(left, result)


def test_no_red_recording_indicator_means_no_timer_mask(tmp_path):
    c = screenshot(tmp_path, 'ordinary', 0)
    raw = np.array(Image.open(c.path)); raw[5:40, 165:200] = 16
    result, boxes = media._analysis_pixels(raw, media.KeyframeConfig())
    assert not boxes and result is raw


@pytest.mark.parametrize('dark', [True, False])
def test_scrolling_same_content_shares_coverage_without_filling_limit(tmp_path, dark):
    a = screenshot(tmp_path, 'a', 0, dark=dark)
    b = screenshot(tmp_path, 'b', 1, offset=60, dark=dark, timer='18')
    c = screenshot(tmp_path, 'c', 2, offset=100, dark=dark, timer='19')
    selected, analysis = select([a, b, c])
    assert len(selected) <= 2
    assert any('scroll_comparison' in d for d in analysis.details.values())
    assert not analysis.summary['hit_limit']
    again, repeated = select([a, b, c])
    assert [x.time for x in selected] == [x.time for x in again]
    assert repeated.summary == analysis.summary


@pytest.mark.parametrize('change', ['text', 'toggle', 'digit', 'same_position_text'])
def test_scroll_does_not_erase_changed_text_or_toggle(tmp_path, change):
    a = screenshot(tmp_path, 'a', 0)
    b = screenshot(tmp_path, 'b', 1, offset=0 if change == 'same_position_text' else 60,
                   value='128' if change in ('text', 'same_position_text') else '127' if change == 'digit' else '123',
                   toggle=change == 'toggle')
    selected, analysis = select([a, b])
    assert selected == [a, b]
    assert analysis.coverage[id(b)] - analysis.coverage[id(a)]


def test_more_repeated_waiting_does_not_add_outputs(tmp_path):
    candidates = [screenshot(tmp_path, str(i), i, offset=(i % 3) * 40,
                             timer=str(10 + i)) for i in range(8)]
    selected, analysis = select(candidates)
    assert len(selected) <= 2
    assert analysis.summary['stop_reason'] == 'covered'
