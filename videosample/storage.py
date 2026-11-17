"""Only remove app-owned run directories and downloaded copies, never source paths."""
import re
import shutil
from pathlib import Path


def managed_directory(root: Path, name: str, *, deleting=False):
    pattern = r'\.deleting-[0-9a-f]{32}' if deleting else r'[0-9a-f]{32}'
    if not re.fullmatch(pattern, name):
        raise ValueError('无效的批次目录')
    root = root.resolve()
    target = root / name
    # Reject a batch symlink/junction, including one pointing at another batch.
    if target.resolve() != target or target.parent != root:
        raise ValueError('批次目录不能是符号链接或指向其它位置')
    return target


def remove_download(root: Path, identity: str, index: int):
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError('无效样本序号')
    batch = managed_directory(root, identity)
    folder = batch / 'inputs' / f'{index:05d}'
    if folder.resolve() != folder:
        raise ValueError('下载目录不能指向其它位置')
    removed = False
    for name in ('video.mp4', 'video.part'):
        path = folder / name
        if path.resolve() != path:
            raise ValueError('下载文件不能指向其它位置')
        if path.is_file():
            path.unlink()
            removed = True
    return removed


def remove_tombstone(root: Path, name: str):
    target = managed_directory(root, name, deleting=True)
    if target.exists():
        # Python >=3.10 rmtree removes nested links/junctions, not their targets.
        shutil.rmtree(target)
