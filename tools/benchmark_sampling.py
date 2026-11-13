"""用本地视频顺序测量实验链路；含过程记录，不含下载和线程池排队。"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videosample import algorithms, media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--limits", nargs="+", type=int, choices=[5, 10, 15], default=[15])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmarks"))
    args = parser.parse_args()
    if args.repeats < 1 or any(not path.is_file() for path in args.videos):
        parser.error("请提供存在的本地视频，repeats 必须大于零")
    directory = args.output / uuid.uuid4().hex
    directory.mkdir(parents=True)
    results = []
    for index, video in enumerate(args.videos):
        for repeat in range(args.repeats):
            for limit in args.limits:
                output = directory / f"{index:03d}_{repeat:02d}_{limit}"
                started = time.perf_counter()
                with media.record_process(output / "process") as recorder:
                    frames = media.extract_scene_keyframes(video, output / "final", max_frames=limit)
                result = {"video": str(video.resolve()), "repeat": repeat + 1,
                          "frame_limit": limit, "frame_count": len(frames),
                          "extraction_s": round(time.perf_counter() - started, 4),
                          "initial_frame_count": recorder.initial_frame_count,
                          "timings_s": recorder.timing_summary(),
                          "algorithm": algorithms.snapshot("current", max_frames=limit)}
                result['selection'] = recorder.selection
                results.append(result)
                (directory / "results.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({key: result[key] for key in
                      ("video", "repeat", "frame_limit", "frame_count", "extraction_s")},
                      ensure_ascii=False), flush=True)
    print(f"结果已保存：{directory.resolve()}")


if __name__ == "__main__":
    main()
