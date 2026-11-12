"""下载实现摘自 auto_eval；实验期间将视频保留在本次运行目录。"""
from __future__ import annotations
import threading
import time
import socket
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
import httpx
_CANCEL = ContextVar("download_cancel", default=None)
_DOWNLOAD_SLOT = ContextVar("download_slot", default=False)
_LOCK = threading.Lock()
_LIMITERS = {}
def load_remote_video_config():
    return RemoteVideoConfig()


@dataclass(frozen=True)
class RemoteVideoConfig:
    download_concurrency: int = 10
    connect_timeout_s: float = 10
    read_timeout_s: float = 30
    total_timeout_s: float = 120
    max_size_mb: float = 500
    max_redirects: int = 5
    max_attempts: int = 3
    retry_delay_s: float = 1
    stale_after_hours: float = 24

def is_remote_video_url(value: str) -> bool:
    return str(value).strip().lower().startswith(("http://", "https://"))

def video_reference_for_log(value: str) -> str:
    """日志去掉签名查询参数；原始输入和导出仍完整保留。"""
    if not is_remote_video_url(value):
        return value
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
    except ValueError:
        return "<无效视频 URL>"

def _resolve_video_addresses(url: str) -> tuple[httpx.URL, list[str]]:
    """解析 HTTP/HTTPS 视频地址，不限制域名或公网/内网 IP。"""
    try:
        parsed = httpx.URL(url)
        if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.userinfo:
            raise ValueError("只支持不含用户名密码的 HTTP/HTTPS 视频链接")
        addresses = list(dict.fromkeys(
            row[4][0] for row in socket.getaddrinfo(
                parsed.host, parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        ))
        if not addresses:
            raise ValueError("视频 URL 无效或域名无法解析")
        return parsed, addresses
    except (httpx.InvalidURL, socket.gaierror) as exc:
        raise ValueError("视频 URL 无效或域名无法解析") from exc

def _check_stop(cancel: threading.Event | None, deadline: float) -> None:
    if cancel is not None and cancel.is_set():
        raise ValueError("远程视频准备已取消")
    if time.monotonic() >= deadline:
        raise ValueError("远程视频下载超过总时间限制")

def download_remote_video(
    url: str, directory: Path, *, config: RemoteVideoConfig | None = None,
) -> Path:
    """下载到调用方拥有的临时目录；失败不留下半文件，不返回原始签名错误。"""
    cfg = config or load_remote_video_config()
    directory.mkdir(parents=True, exist_ok=True)
    cancel = _CANCEL.get()
    with _LOCK:
        limiter = _LIMITERS.setdefault(cfg.download_concurrency, threading.BoundedSemaphore(cfg.download_concurrency))
    owns_slot = not _DOWNLOAD_SLOT.get()
    while owns_slot and not limiter.acquire(timeout=0.2):
        if cancel is not None and cancel.is_set():
            raise ValueError("远程视频准备已取消")
    part = directory / "video.part"
    target = directory / "video.mp4"
    deadline = time.monotonic() + cfg.total_timeout_s
    max_bytes = int(cfg.max_size_mb * 1024 * 1024)
    try:
        for attempt in range(cfg.max_attempts):
            _check_stop(cancel, deadline)
            current_url = url.strip()
            try:
                # 保留原有直连方式；HTTPS 保留原域名用于证书校验。
                with httpx.Client(trust_env=False, follow_redirects=False) as client:
                    for redirect in range(cfg.max_redirects + 1):
                        parsed, addresses = _resolve_video_addresses(current_url)
                        remaining = max(0.1, deadline - time.monotonic())
                        pinned = parsed.copy_with(host=addresses[attempt % len(addresses)])
                        request = client.build_request(
                            "GET", pinned,
                            headers={"Host": parsed.netloc.decode("ascii"), "Accept-Encoding": "identity"},
                            extensions={"sni_hostname": parsed.host},
                            timeout=httpx.Timeout(
                                min(cfg.read_timeout_s, remaining),
                                connect=min(cfg.connect_timeout_s, remaining),
                            ),
                        )
                        response = client.send(request, stream=True)
                        try:
                            if response.status_code in {301, 302, 303, 307, 308}:
                                if redirect == cfg.max_redirects or not response.headers.get("location"):
                                    raise ValueError("远程视频重定向过多或缺少目标地址")
                                current_url = urljoin(current_url, response.headers["location"])
                                continue
                            response.raise_for_status()
                            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                            if content_type in {"text/html", "application/json", "text/plain"}:
                                raise ValueError("链接返回的是网页或错误信息，不是视频文件")
                            length = response.headers.get("content-length", "")
                            if length.isdigit() and int(length) > max_bytes:
                                raise ValueError(f"远程视频超过 {cfg.max_size_mb:g}MB 大小限制")
                            size = 0
                            with part.open("wb") as handle:
                                for chunk in response.iter_bytes():
                                    _check_stop(cancel, deadline)
                                    size += len(chunk)
                                    if size > max_bytes:
                                        raise ValueError(f"远程视频超过 {cfg.max_size_mb:g}MB 大小限制")
                                    handle.write(chunk)
                            _check_stop(cancel, deadline)
                            if not size:
                                raise ValueError("远程视频为空文件")
                            part.replace(target)
                            return target
                        finally:
                            response.close()
            except (httpx.HTTPError, OSError) as exc:
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                retryable = status is None or status in {408, 429, 500, 502, 503, 504}
                if attempt + 1 >= cfg.max_attempts or not retryable:
                    reason = f"HTTP {status}" if status is not None else type(exc).__name__
                    raise ValueError(f"远程视频下载失败：{reason}（{video_reference_for_log(url)}）") from None
                delay = min(cfg.retry_delay_s * (2 ** attempt), max(0, deadline - time.monotonic()))
                if cancel is not None:
                    cancel.wait(delay)
                else:
                    time.sleep(delay)
            finally:
                part.unlink(missing_ok=True)
        raise ValueError("远程视频下载失败")
    finally:
        if owns_slot:
            limiter.release()
