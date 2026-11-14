"""固定 base，实验修改仅在 media.py；每次运行保存实际实现与默认配置。"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from . import media
from .versions import base_media

_MODULES = {"base": base_media, "current": media}
_SOURCES = {key: Path(module.__file__).read_text(encoding="utf-8").encode("utf-8")
            for key, module in _MODULES.items()}
_BASE = json.loads(Path(base_media.__file__).with_name("base.json").read_text(encoding="utf-8-sig"))
if hashlib.sha256(_SOURCES["base"]).hexdigest() != _BASE["sha256"]:
    raise RuntimeError("base 文件已被修改，请恢复固定基线，实验修改应放在 videosample/media.py")


def implementation(key):
    if key not in _MODULES:
        raise ValueError("请选择 base 或当前实验版")
    return _MODULES[key]


def snapshot(key, *, max_frames=None):
    module = implementation(key)
    result = {"key": key, "name": "base · 原始方法" if key == "base" else "当前实验版",
            "algorithm_version": module.KEYFRAME_ALGORITHM_VERSION,
            "source_sha256": hashlib.sha256(_SOURCES[key]).hexdigest(),
            "defaults": asdict(module.KeyframeConfig())}
    if key == "current" and max_frames is not None:
        result["overrides"] = {"max_frames": max_frames}
    if key == "current":
        result['selection_policy'] = 'adaptive-scroll-v1'
    return result


def save_source(key, directory):
    (directory / "algorithm.py").write_bytes(_SOURCES[key])
