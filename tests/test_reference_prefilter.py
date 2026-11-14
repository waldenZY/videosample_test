"""参考预筛只否定不可能通过原 RGB 核对的位移，不判定合并、不删候选。"""
from dataclasses import replace
import numpy as np
import pytest

from videosample import media


@pytest.mark.parametrize('width', [12,35,257,720])
@pytest.mark.parametrize('rh,ch,q', [(30,30,0),(45,70,-25),(80,40,31),(40,80,0),(40,30,28)])
def test_exact_overlap_is_never_rejected(width,rh,ch,q):
    rng=np.random.default_rng(91)
    reference=rng.integers(0,256,(rh,width,3),dtype=np.uint8)
    crop=rng.integers(0,256,(ch,width,3),dtype=np.uint8)
    top,bottom=max(0,q),min(rh,q+ch)
    crop[top-q:bottom-q]=reference[top:bottom]
    assert media._LocalRows(reference).possibly_matches(media._LocalRows(crop),media.KeyframeConfig())


@pytest.mark.parametrize('seed', range(12))
def test_rejected_pair_has_no_overlap_passing_original_strict_check(seed):
    rng=np.random.default_rng(seed)
    width=int(rng.integers(12,63));rh=int(rng.integers(20,60));ch=int(rng.integers(20,60))
    reference=rng.integers(0,90,(rh,width,3),dtype=np.uint8)
    crop=rng.integers(160,256,(ch,width,3),dtype=np.uint8)
    config=media.KeyframeConfig()
    assert not media._LocalRows(reference).possibly_matches(media._LocalRows(crop),config)
    minimum=max(8,int(np.ceil(min(rh,ch)*.40)))
    for q in range(-ch+minimum,rh-minimum+1):
        top,bottom=max(0,q),min(rh,q+ch)
        assert not media._strict_region_equal(reference[top:bottom,2:-2],crop[top-q:bottom-q,2:-2],config)


@pytest.mark.parametrize('base', [0,240])
def test_low_contrast_small_differences_remain_uncertain(base):
    rng=np.random.default_rng(81)
    reference=rng.integers(base,base+8,(333,257,3),dtype=np.uint8)
    crop=reference[30:203].copy();crop[:10,10:20]=base+7
    assert media._LocalRows(reference).possibly_matches(media._LocalRows(crop),media.KeyframeConfig())


def test_reference_filter_skips_search_without_declaring_frames_duplicate(monkeypatch):
    pixels=np.full((120,400,3),20,dtype=np.uint8)
    crop=np.full_like(pixels,240)
    track=media._LocalTrack(0,pixels,0,1.,'local',0)
    work=media._LocalWork(media.KeyframeConfig())
    monkeypatch.setattr(work,'peak',lambda *args:pytest.fail('不可能匹配的参考不应启动位置搜索'))
    assert media._locate_local_content(track,crop,work.config,work=work) is None
    assert work.counts['reference_filter_rejections']==1
    assert work.counts['full_search_attempts']==0
    assert work.counts['failed_pairs']==1
    work.close()


def test_scene_gray_preparation_is_preserved_for_weak_content_classification():
    pixels=np.full((100,40,3),100,dtype=np.uint8)
    crop=np.full_like(pixels,107)
    tracks=[media._LocalTrack(0,pixels.copy(),0,1.,'scene',0) for _ in range(2)]
    config=media.KeyframeConfig()
    for track,active in zip(tracks,[False,True]):
        assert media._locate_local_content(track,crop,replace(config,local_reference_prefilter=active)) is None
    assert all(track.gray is not None and float(track.gray.std())==0 for track in tracks)
    np.testing.assert_array_equal(tracks[0].gray,tracks[1].gray)


def test_expanding_reference_invalidates_row_summary():
    pixels=np.random.default_rng(3).integers(0,256,(80,40,3),dtype=np.uint8)
    track=media._LocalTrack(0,pixels.copy(),0,1.,'local',0)
    track.row_summary=media._LocalRows(pixels)
    assert media._extend_local_track(track,pixels,10)
    assert track.row_summary is None


def test_width_prefilter_preserves_order_of_eligible_references(monkeypatch):
    crop=np.zeros((30,40,3),dtype=np.uint8)
    tracks=[media._LocalTrack(i,np.zeros((30,w,3),dtype=np.uint8),0,1.,'local',0)
            for i,w in enumerate([35,40,39,40])]
    visited=[]
    monkeypatch.setattr(media,'_locate_local_content',lambda track,*args:visited.append(track.index))
    work=media._LocalWork(media.KeyframeConfig(local_match_workers=1))
    assert [track.index for track,_ in work.comparisons(tracks,crop,None,None)]==[1,3]
    assert visited==[1,3] and work.counts['reference_width_skips']==2
    work.close()


def test_reference_prefilter_can_be_disabled():
    assert media.KeyframeConfig(local_reference_prefilter=False).local_reference_prefilter is False
    with pytest.raises(ValueError):media.KeyframeConfig(local_reference_prefilter='yes')
