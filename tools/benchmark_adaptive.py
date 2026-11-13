"""生成固定的 1–2 分钟验收组合；输入和测量结果只写入 runs/。"""
import argparse
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videosample import media


def picture(kind, value=0, y=110, tall=False, size=(240, 400)):
    image = Image.new('RGB', (240, 400), '#b8c2cc')
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    if kind in {'setting', 'card', 'result'}:
        height = 560 if tall else 180
        card = Image.new('RGB', (200, height), 'white')
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, 199, height - 1), outline='#46566a', width=2)
        draw.text((10, 10), f'SETTING {value}' if kind == 'setting' else 'RESULT / CARD', font=font, fill='#172635')
        for row, row_y in enumerate(range(42, height - 25, 30)):
            draw.text((10, row_y), f'Item {row}: {13 + row * 7 + value * 23}', font=font, fill='#25395a')
            if kind == 'setting':
                draw.rectangle((142, row_y, 152 + (value * 7 + row * 3) % 35, row_y + 12), fill='#387b9e')
        draw.rectangle((10, height - 24, 189, height - 10), fill='#387b9e')
        image.paste(card, (20, y))
    elif kind == 'loading':
        ImageDraw.Draw(image).ellipse((106, y, 134, y + 28), fill='#24496a')
    return image.resize(size)


def encode(directory, name, segments, size):
    folder = directory / name
    folder.mkdir()
    lines, required, cursor, images = [], [], 0.0, {}
    for seconds, kind, value, y, tall, needed in segments:
        key = (kind, value, y, tall)
        if key not in images:
            path = folder / f'{len(images):03d}.png'
            picture(kind, value, y, tall, size).save(path)
            images[key] = path
        lines += [f"file '{images[key].resolve().as_posix()}'", f'duration {seconds}']
        if needed:
            required.append({'range': [cursor, cursor + seconds], 'image': images[key].name})
        cursor += seconds
    lines.append(f"file '{images[key].resolve().as_posix()}'")
    listing = folder / 'frames.txt'
    listing.write_text('\n'.join(lines), encoding='utf-8')
    video = folder / 'input.mp4'
    subprocess.run([media.resolve_ffmpeg_executable(), '-y', '-v', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-t', str(cursor), '-r', '12', '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-crf', '18', '-threads', '2', str(video)], check=True)
    return video, cursor, required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/adaptive_acceptance'))
    parser.add_argument('--skip-tail', action='store_true', help='只检查普通组合；完整验收不能跳过长尾')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2, help='默认与平台一致，同时处理两题')
    args = parser.parse_args()
    directory = args.output / uuid.uuid4().hex
    directory.mkdir(parents=True)
    blank = lambda duration: (duration, 'blank', 0, 110, False, False)
    cases = [
        ('simple_60s', [blank(10), (15, 'setting', 0, 110, False, True),
                       (15, 'setting', 1, 110, False, True), (20, 'result', 0, 110, False, True)], (240, 400), (3, 5)),
        ('medium_120s', [blank(15), *[(15, 'setting', i, 110, False, True) for i in range(6)], blank(15)], (720, 1280), (6, 10)),
        ('scroll_120s', [blank(10), (45, 'card', 0, 20, True, True),
                        (45, 'card', 0, -160, True, True), blank(20)], (720, 1280), (3, 5)),
        ('loading_120s', [blank(10), *[(1, 'loading', 0, y, False, False)
                         for _ in range(20) for y in (90, 110, 130, 150, 170)], blank(10)], (240, 400), (2, 5)),
    ]
    prepared = [(name, *encode(directory, name, segments, size), expected) for name, segments, size, expected in cases]
    if not args.skip_tail:
        tail = directory / 'dynamic_120s.mp4'
        existing = Path('runs/performance_check/input_120s_hd.mp4')
        if existing.is_file():
            tail = existing
        else:
            subprocess.run([media.resolve_ffmpeg_executable(), '-y', '-v', 'error', '-f', 'lavfi', '-i',
                            'testsrc2=size=240x400:rate=12:duration=120', '-vf', 'scale=720:1280,fps=30',
                            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-threads', '2', str(tail)], check=True)
        prepared.append(('dynamic_120s', tail, 120.0, [], (2, 15)))
    results = []
    def evaluate(item):
        name, video, seconds, required, expected = item
        output = directory / (name + '_result')
        started = time.perf_counter()
        with media.record_process(output / 'process') as recorder:
            frames = media.extract_scene_keyframes(video, output / 'final',
                                                   task_start_time=0, task_end_time=seconds - .5)
        elapsed = time.perf_counter() - started
        metadata = json.loads((output / 'final/keyframes.json').read_text(encoding='utf-8'))
        times = [frame['time'] for frame in metadata['selected']]
        # 编码后切换时间可能提前一个视频帧；按实际图片内容验收，不靠毫秒命中。
        templates = []
        for path in sorted(video.parent.glob('*.png')) if required else []:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB').resize((120, 200)), dtype=np.float32)
            templates.append((path.name, pixels))
        matched = set()
        for frame in frames if templates else []:
            with Image.open(frame) as image:
                pixels = np.asarray(image.convert('RGB').resize((120, 200)), dtype=np.float32)
            def distance(template):
                delta = np.abs(pixels - template[1])
                return float(np.mean(delta * delta) + .5 * delta.reshape(10, 20, 10, 12, 3).mean(axis=(1, 3, 4)).max() ** 2)
            matched.add(min(templates, key=distance)[0])
        missing = [item['range'] for item in required if item['image'] not in matched]
        result = {'case': name, 'seconds': seconds, 'frames': len(frames), 'extraction_s': round(elapsed, 3),
                  'selected_times': times,
                  'expected_range': expected, 'missing_intervals': missing,
                  'quality_pass': expected[0] <= len(frames) <= expected[1] and not missing,
                  'selection': metadata['selection'], 'stages': recorder.timing_summary()}
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        for result in workers.map(evaluate, prepared):
            results.append(result)
            (directory / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({key: result[key] for key in ('case', 'frames', 'extraction_s', 'quality_pass', 'missing_intervals')}), flush=True)
    mean = sum(result['extraction_s'] for result in results) / len(results)
    print(json.dumps({'mean_extraction_s': round(mean, 3), 'all_quality_pass': all(r['quality_pass'] for r in results),
                      'timing_target_pass': mean <= 10, 'workers': args.workers,
                      'limit_hits': sum(result['selection']['hit_limit'] for result in results),
                      'tail_included': not args.skip_tail, 'result_directory': str(directory.resolve())}), flush=True)
    if not all(result['quality_pass'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
