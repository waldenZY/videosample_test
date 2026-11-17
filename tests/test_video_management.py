import json
import os
import subprocess

import pytest
from fastapi.testclient import TestClient

from videosample import server
from videosample.storage import managed_directory


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(server.POOL, 'submit', lambda *args: None)
    with TestClient(server.app) as c:
        yield c


def imported(client, source='https://example.test/input.mp4?token=original'):
    payload = json.dumps({'id': 'q', 'query': '测试原视频', 'video_path': source}).encode()
    return client.post('/api/import', files={'file': ('video.jsonl', payload)}).json()


def started(client, *, retain=False, mode='both'):
    run = imported(client)
    return client.post(f"/api/runs/{run['id']}/start", json={'run_mode': mode, 'retain_downloaded_video': retain}).json()


def downloaded(run):
    path = server.RUNS / run['id'] / 'inputs/00000/video.mp4'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'download copy')
    return path


def finish(run, status='done'):
    for identity in run.get('chain_ids', {'one': run['id']}).values():
        server.update_row(identity, 0, status=status)


def test_unrun_local_video_reads_source_without_preparation_or_record_creation(client, tmp_path, monkeypatch):
    source = tmp_path / 'original.mp4'
    source.write_bytes(b'original file')
    run = imported(client, str(source))
    folder = server.run_directory(run['id'])
    before = (folder / 'run.json').read_bytes()
    monkeypatch.setattr(server, 'prepared_video', lambda *a: pytest.fail('Viewing must not prepare/download video'))
    response = client.get(f"/api/runs/{run['id']}/items/0/video")
    assert response.status_code == 200 and response.content == b'original file'
    assert not (folder / 'items').exists() and not (folder / 'inputs').exists()
    assert (folder / 'run.json').read_bytes() == before
    assert client.get(f"/api/runs/{run['id']}/items/1/video").status_code == 404


def test_remote_link_uses_original_url_even_if_cache_gone(client, monkeypatch):
    run = imported(client)
    monkeypatch.setattr(server, 'download_remote_video', lambda *a: pytest.fail('Link must not download video'))
    response = client.get(f"/api/runs/{run['id']}/items/0/video", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers['location'] == run['rows'][0]['item']['video_path']
    assert not (server.run_directory(run['id']) / 'inputs').exists()


@pytest.mark.parametrize('retain', [False, True])
@pytest.mark.parametrize('last_status', ['done', 'error'])
def test_download_reused_until_both_consumers_finish(client, monkeypatch, retain, last_status):
    run = started(client, retain=retain)
    path = server.RUNS / run['id'] / 'inputs/00000/video.mp4'
    downloads = []

    def download(url, directory):
        downloads.append(url)
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'video')
        return path

    def work(identity, index):
        record = server.read_run(identity)
        assert server.prepared_video(record, index, record['rows'][index]['item']).read_bytes() == b'video'
        status = last_status if identity == run['chain_ids']['current'] else 'done'
        server.update_row(identity, index, status=status)

    monkeypatch.setattr(server, 'download_remote_video', download)
    monkeypatch.setattr(server, '_process_row', work)
    server.process_row(run['chain_ids']['base'], 0)
    assert path.exists(), 'Other chain is queued, so the copy must remain'
    server.process_row(run['chain_ids']['current'], 0)
    assert len(downloads) == 1
    assert path.exists() is retain
    assert server.read_run(run['chain_ids']['current'])['rows'][0]['status'] == last_status
    assert not any(job[0] in run['chain_ids'].values() for job in server.ACTIVE_JOBS)


def test_active_worker_blocks_cleanup_and_delete_even_after_done_status(client):
    run = started(client)
    path = downloaded(run)
    finish(run)
    job = (run['chain_ids']['current'], 0)
    server.ACTIVE_JOBS.add(job)
    try:
        server.cleanup_sample_download(run['id'], 0)
        assert path.exists()
        assert client.delete(f"/api/runs/{run['id']}").status_code == 409
    finally:
        server.ACTIVE_JOBS.discard(job)
    server.cleanup_sample_download(run['id'], 0)
    assert not path.exists()


def test_cleanup_failure_warns_without_turning_extraction_into_error(client, monkeypatch):
    run = started(client)
    path = downloaded(run)
    finish(run)
    original = server.remove_download
    def fail(*args):
        raise PermissionError('file busy')
    monkeypatch.setattr(server, 'remove_download', fail)
    server.cleanup_sample_download(run['id'], 0)
    row = server.read_run(run['id'])['rows'][0]
    assert row['status'] == 'done' and row['download_cleanup_error'] and path.exists()
    monkeypatch.setattr(server, 'remove_download', original)
    server.cleanup_sample_download(run['id'], 0)
    row = server.read_run(run['id'])['rows'][0]
    assert not path.exists() and not row.get('download_cleanup_error')


def test_restart_cleans_interrupted_opted_out_run_but_keeps_history(client):
    run = started(client)
    path = downloaded(run)
    old = imported(client)
    legacy_path = downloaded(old)
    old.pop('retain_downloaded_video')
    server.save_run(old)
    with TestClient(server.app):
        assert not path.exists()
        assert legacy_path.exists()
        assert server.read_run(run['id'])['rows'][0]['status'] == 'interrupted'


def test_delete_whole_group_from_child_keeps_sources_and_reruns(client, tmp_path):
    source = tmp_path / 'original.mp4'
    source.write_bytes(b'original')
    run = started(client, retain=True)
    downloaded(run)
    finish(run)
    for identity in run['chain_ids'].values():
        directory = server.run_directory(identity) / 'items/00000/process/candidates'
        directory.mkdir(parents=True)
        (directory / 'f00001.jpg').write_bytes(b'image')
        (directory.parent.parent / 'video_ref.json').write_text(json.dumps({'path': str(source)}))
    copied = client.post(f"/api/runs/{run['id']}/copy").json()
    assert copied['retain_downloaded_video'] is False
    response = client.delete(f"/api/runs/{run['chain_ids']['current']}")
    assert response.status_code == 200 and response.json()['cleanup_pending'] is False
    assert set(response.json()['deleted_ids']) == set(run['chain_ids'].values())
    for identity in run['chain_ids'].values():
        assert not (server.RUNS / identity).exists()
    assert source.read_bytes() == b'original'
    assert client.get(f"/api/runs/{copied['id']}").status_code == 200
    assert client.get(f"/api/runs/{run['id']}").status_code == 404


def test_running_delete_is_rejected_and_imported_delete_works(client):
    run = started(client)
    assert client.delete(f"/api/runs/{run['id']}").status_code == 409
    assert server.run_directory(run['id']).exists()
    fresh = imported(client)
    assert client.delete(f"/api/runs/{fresh['id']}").status_code == 200
    assert not (server.RUNS / fresh['id']).exists()
    assert client.delete('/api/runs/invalid').status_code == 404


def test_owned_directory_rejects_escape_and_root(client, tmp_path):
    for name in ('..', '.', str(tmp_path), '../other'):
        with pytest.raises(ValueError):
            managed_directory(server.RUNS, name)


def test_failed_disk_cleanup_is_recovered_on_restart(client, monkeypatch):
    run = imported(client)
    path = downloaded(run)
    original = server.remove_tombstone
    def busy(*args):
        raise PermissionError('file busy')
    monkeypatch.setattr(server, 'remove_tombstone', busy)
    response = client.delete(f"/api/runs/{run['id']}")
    assert response.status_code == 200 and response.json()['cleanup_pending']
    assert client.get('/api/runs').json() == []
    assert list(server.RUNS.glob('.deleting-*'))
    monkeypatch.setattr(server, 'remove_tombstone', original)
    with TestClient(server.app):
        assert not list(server.RUNS.glob('.deleting-*'))
        assert not path.exists()


def test_cleanup_never_follows_download_symlink(client, tmp_path):
    run = started(client)
    finish(run)
    original = tmp_path / 'original.mp4'
    original.write_bytes(b'do not delete')
    target = server.RUNS / run['id'] / 'inputs/00000/video.mp4'
    target.parent.mkdir(parents=True)
    try:
        target.symlink_to(original)
    except OSError:
        pytest.skip('System does not permit creating symlinks')
    server.cleanup_sample_download(run['id'], 0)
    assert original.read_bytes() == b'do not delete'
    assert server.read_run(run['id'])['rows'][0]['download_cleanup_error']


def junction(path, target):
    def quoted(value):
        return "'" + str(value).replace("'", "''") + "'"
    subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command',
        f'New-Item -ItemType Junction -Path {quoted(path)} -Value {quoted(target)} | Out-Null'],
        check=True, capture_output=True, timeout=20)


@pytest.mark.skipif(os.name != 'nt', reason='Windows directory junction behavior')
def test_batch_junction_cannot_redirect_batch_deletion(client, tmp_path):
    run = imported(client)
    folder = server.run_directory(run['id'])
    source = tmp_path / 'source_directory'
    assert folder.resolve().is_relative_to(tmp_path.resolve())
    assert source.resolve().is_relative_to(tmp_path.resolve())
    folder.rename(source)
    junction(folder, source)
    response = client.delete(f"/api/runs/{run['id']}")
    assert response.status_code == 409
    assert (source / 'run.json').is_file()


@pytest.mark.skipif(os.name != 'nt', reason='Windows directory junction behavior')
def test_nested_junction_target_is_preserved_during_batch_delete(client, tmp_path):
    run = imported(client)
    source = tmp_path / 'original_video_directory'
    source.mkdir()
    (source / 'original.mp4').write_bytes(b'preserve source')
    folder = server.run_directory(run['id']) / 'nested_link'
    assert folder.absolute().is_relative_to(server.RUNS.resolve())
    assert source.resolve().is_relative_to(tmp_path.resolve())
    junction(folder, source)
    response = client.delete(f"/api/runs/{run['id']}")
    assert response.status_code == 200 and response.json()['cleanup_pending'] is False
    assert (source / 'original.mp4').read_bytes() == b'preserve source'


@pytest.mark.skipif(os.name != 'nt', reason='Windows directory junction behavior')
def test_download_directory_junction_cannot_delete_source_video(client, tmp_path):
    run = started(client)
    finish(run)
    source = tmp_path / 'source_videos'
    source.mkdir()
    (source / 'video.mp4').write_bytes(b'original')
    folder = server.run_directory(run['id']) / 'inputs/00000'
    folder.parent.mkdir()
    assert folder.absolute().is_relative_to(server.RUNS.resolve())
    assert source.resolve().is_relative_to(tmp_path.resolve())
    junction(folder, source)
    server.cleanup_sample_download(run['id'], 0)
    assert (source / 'video.mp4').read_bytes() == b'original'
    assert server.read_run(run['id'])['rows'][0]['download_cleanup_error']
