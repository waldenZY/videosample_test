"""重复内容在自适应选择前处理，不依赖末尾再删图。"""
from PIL import Image, ImageDraw

from videosample import media


def frame(tmp_path, name, time, color, support=0, reason='content-state'):
    path = tmp_path / f'{name}.png'
    Image.new('RGB', (96, 160), color).save(path)
    return media._Candidate(time, path, '4fps', keep_reason=reason, support_s=support)


def test_repeat_is_covered_across_other_pages_and_large_time_gap(tmp_path):
    start = frame(tmp_path, 'start', 0, 'white')
    card = frame(tmp_path, 'card', 10, 'black', 3)
    app = frame(tmp_path, 'app', 30, 'red')
    repeated = frame(tmp_path, 'repeated', 200, 'black', .25)
    assert media._limit_candidates([start, card, app, repeated], 15) == [start, card, app]


def test_small_encoding_difference_does_not_add_an_output(tmp_path):
    frames = [frame(tmp_path, 'start', 0, 'white'), frame(tmp_path, 'a', 10, (40, 40, 40), 2),
              frame(tmp_path, 'other', 20, 'blue'), frame(tmp_path, 'b', 80, (41, 41, 41))]
    selected = media._limit_candidates(frames, 15)
    assert frames[1] in selected and frames[3] not in selected


def test_different_local_text_keeps_independent_content(tmp_path):
    candidates = [frame(tmp_path, 'start', 0, 'white')]
    for index, text in enumerate(['123', '128']):
        image = Image.new('RGB', (96, 160), 'white')
        ImageDraw.Draw(image).text((20, 60), text, fill='black')
        path = tmp_path / f'text-{index}.png'
        image.save(path)
        candidates.append(media._Candidate(10 + index * 100, path, '4fps'))
    assert media._limit_candidates(candidates, 15) == candidates


def test_review_similarity_does_not_chain_through_discarded_images(tmp_path):
    config = media.KeyframeConfig(global_dedup_rms_threshold=1.5 / 255)
    start = frame(tmp_path, 'start', 0, 'white')
    a = frame(tmp_path, 'a', 10, (40, 40, 40), 3)
    b = frame(tmp_path, 'b', 50, (41, 41, 41), 2)
    c = frame(tmp_path, 'c', 100, (42, 42, 42), 1)
    reviewed, _ = media._review_merged_content([start, a], [b, c], config)
    assert reviewed == [start, a, c]


def test_identical_endpoints_still_count_as_required_positions(tmp_path):
    start = frame(tmp_path, 'start', 0, 'white')
    ordinary = frame(tmp_path, 'ordinary', 20, 'white')
    end = frame(tmp_path, 'end', 40, 'white', reason='task-end-explicit')
    assert media._limit_candidates([start, ordinary, end], 15) == [start, end]
