"""用可描述的画面验证选图，不使用业务视频。"""
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from videosample import media


def font():
    for path in (Path('C:/Windows/Fonts/arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if path.exists():
            return ImageFont.truetype(str(path), 15)
    return ImageFont.load_default()


def panel(value='TIME 07:00', height=180):
    image = Image.new('RGB', (200, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 199, height - 1), outline='#46566a', width=2)
    draw.text((12, 12), value, font=font(), fill='#182433')
    for row, y in enumerate(range(48, height - 25, 35)):
        draw.text((12, y), f'Field {row:02d} : {row * 7 + 13}', font=font(), fill='#32465a')
    draw.rectangle((12, height - 25, 187, height - 10), fill='#2b7296')
    return image


def shot(tmp_path, name, time, content=None, y=110, color='#b8c2cc', end=False):
    image = Image.new('RGB', (240, 400), color)
    if content is not None:
        image.paste(content, (20, y))
    path = tmp_path / f'{name}.png'
    image.save(path)
    return media._Candidate(time, path, 'task-end' if end else '4fps', page_id=0,
                            support_s=.25, keep_reason='task-end-explicit' if end else '')


def select(candidates, limit=15):
    config = media.KeyframeConfig()
    analysis = media._analyze_local_content(candidates, config)
    selected = media._limit_candidates(candidates, limit, config=config, analysis=analysis)
    return selected, analysis


@pytest.mark.parametrize('settings', [3, 6])
def test_different_settings_on_same_page_are_kept_without_filling_limit(tmp_path, settings):
    candidates = [shot(tmp_path, 'start', 0)]
    required = [shot(tmp_path, f'setting-{i}', i + 1, panel(f'SETTING {i}: {i * 23}'))
                for i in range(settings)]
    candidates += required + [shot(tmp_path, 'end', settings + 1, end=True)]
    selected, analysis = select(candidates)
    assert all(candidate in selected for candidate in required)
    assert len(selected) == settings + 2
    assert analysis.summary['stop_reason'] == 'covered'
    assert not analysis.summary['hit_limit']


@pytest.mark.parametrize('encoding', ['png', 'jpeg', 'hd_png'])
def test_full_card_replaces_entering_and_leaving_partial_cards(tmp_path, encoding):
    content = panel()
    start = shot(tmp_path, 'start', 0)
    entering = shot(tmp_path, 'entering', 1, content, y=310)
    full = shot(tmp_path, 'full', 2, content, y=110)
    leaving = shot(tmp_path, 'leaving', 3, content, y=-90)
    end = shot(tmp_path, 'end', 4, end=True)
    for candidate in [start, entering, full, leaving, end]:
        with Image.open(candidate.path) as image:
            encoded = image.convert('RGB')
        if encoding == 'jpeg':
            encoded.save(candidate.path, format='JPEG', quality=95)
        elif encoding == 'hd_png':
            encoded.resize((720, 1280)).save(candidate.path)
    selected, analysis = select([start, entering, full, leaving, end])
    assert selected == [start, full, end]
    assert analysis.replacements[id(entering)] is full
    assert analysis.replacements[id(leaving)] is full


def test_tall_card_keeps_complementary_views(tmp_path):
    content = panel(height=560)
    start = shot(tmp_path, 'start', 0)
    top = shot(tmp_path, 'top', 1, content, y=20)
    bottom = shot(tmp_path, 'bottom', 2, content, y=-160)
    end = shot(tmp_path, 'end', 3, end=True)
    selected, analysis = select([start, top, bottom, end])
    assert selected == [start, top, bottom, end]
    assert analysis.summary['uncovered_count'] == 0


def test_exposure_change_can_change_card_average_color(tmp_path):
    content = Image.new('RGB', (200, 180), '#d52b32')
    draw = ImageDraw.Draw(content)
    draw.rectangle((0, 90, 199, 179), fill='#2058d2')
    draw.rectangle((0, 0, 199, 179), outline='white', width=2)
    draw.text((12, 15), 'TOP CONTENT', font=font(), fill='white')
    draw.text((12, 120), 'BOTTOM CONTENT', font=font(), fill='white')
    start = shot(tmp_path, 'start', 0)
    top = shot(tmp_path, 'top', 1, content, y=310)
    full = shot(tmp_path, 'full', 2, content, y=110)
    bottom = shot(tmp_path, 'bottom', 3, content, y=-90)
    end = shot(tmp_path, 'end', 4, end=True)
    selected, _ = select([start, top, full, bottom, end])
    assert selected == [start, full, end]


def test_low_contrast_blank_variation_has_no_reliable_addition(tmp_path):
    start = shot(tmp_path, 'start', 0, color=(180, 180, 180))
    variations = [shot(tmp_path, f'noise-{i}', i + 1, color=(i, i, i)) for i in range(182, 187)]
    end = shot(tmp_path, 'end', 200, color=(180, 180, 180), end=True)
    selected, analysis = select([start, *variations, end])
    assert selected == [start, end]
    assert analysis.summary['stop_reason'] == 'no_reliable_addition'
    assert not analysis.summary['hit_limit']


def test_more_waiting_and_repeated_frames_do_not_add_outputs(tmp_path):
    start = shot(tmp_path, 'start', 0)
    content = panel()
    candidates = [start]
    for i in range(35):
        candidates.append(shot(tmp_path, f'wait-{i}', i + 1, content))
    end = shot(tmp_path, 'end', 120, end=True)
    selected, analysis = select(candidates + [end])
    assert len(selected) == 3
    assert analysis.summary['uncovered_count'] == 0
    assert not analysis.summary['hit_limit']


def test_looping_vertical_loading_motion_does_not_fill_limit(tmp_path):
    candidates = [shot(tmp_path, 'start', 0)]
    for i, y in enumerate([90, 110, 130, 150, 170, 150, 130, 110] * 3):
        image = Image.new('RGB', (240, 400), '#b8c2cc')
        draw = ImageDraw.Draw(image)
        draw.ellipse((106, y, 134, y + 28), fill='#24496a')
        path = tmp_path / f'loading-{i}.png'
        image.save(path)
        candidates.append(media._Candidate(i + 1, path, '4fps', page_id=i, support_s=.25))
    candidates.append(shot(tmp_path, 'end', 30, end=True))
    selected, analysis = select(candidates)
    assert len(selected) <= 5
    assert not analysis.summary['hit_limit']


def test_global_page_diversity_cannot_exclude_local_popup(tmp_path):
    start = shot(tmp_path, 'start', 0)
    pages = [shot(tmp_path, f'page-{i}', i + 1, color=(i * 23, i * 19, i * 17)) for i in range(8)]
    for i, candidate in enumerate(pages):
        candidate.page_id = i + 1
    popup = shot(tmp_path, 'popup', 10, panel('SAVED'))
    end = shot(tmp_path, 'end', 11, end=True)
    selected, _ = select([start, *pages, popup, end], limit=10)
    assert popup in selected


def test_limit_reports_uncovered_content_and_is_deterministic(tmp_path):
    candidates = [shot(tmp_path, 'start', 0)]
    candidates += [shot(tmp_path, f'setting-{i}', i + 1, panel(f'SET {i}: {i * 29}')) for i in range(18)]
    candidates.append(shot(tmp_path, 'end', 20, end=True))
    first, analysis = select(candidates)
    second, repeated = select(candidates)
    assert first == second and len(first) == 15
    assert analysis.summary == repeated.summary
    assert analysis.summary['stop_reason'] == 'limit_reached'
    assert analysis.summary['hit_limit'] and analysis.summary['uncovered_count'] > 0
    assert analysis.summary['uncovered_content']
