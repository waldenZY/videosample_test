"""小图只限制位置，低对比和重复位置不得被当成可靠的新合并证据。"""
from dataclasses import replace
import numpy as np
import pytest

from videosample import media


def engine(threshold=.9):
    return media._LocalBoundedSearch(media._LocalWork(media.KeyframeConfig(
        local_bounded_search=False, local_template_threshold=threshold)))


@pytest.mark.parametrize('width,height,strip', [(260,80,8),(320,100,24),(720,180,24),(719,160,17)])
@pytest.mark.parametrize('style', ['noise','light','dark','flat','low-contrast'])
def test_bound_covers_original_correlation(width,height,strip,style):
    rng=np.random.default_rng(69)
    if style=='flat':
        target=np.full((height,width),245,dtype=np.uint8)
    elif style=='low-contrast':
        target=rng.integers(241,246,(height,width),dtype=np.uint8)
    else:
        target=rng.integers(0,256,(height,width),dtype=np.uint8)
        if style in {'light','dark'}:
            target[:]=245 if style=='light' else 20
            for row in range(10,height-10,13):
                target[row:row+3,17:width-17:3]=20 if style=='light' else 240
    source=rng.integers(0,256,(height+17,width),dtype=np.uint8)
    source[15:15+strip,2:-2]=target[20:20+strip,2:-2]
    if np.std(source[15:15+strip,2:-2])<3:
        source[15:15+strip,2:-2]=rng.integers(0,256,(strip,width-4),dtype=np.uint8)
    sd,td=media._LocalSearchData(source),media._LocalSearchData(target)
    patch=source[15:15+strip,2:-2]
    search=engine()
    upper=search.bounds(sd,td,15,patch)
    cv=media._local_cv()
    actual=cv.matchTemplate(target,patch,cv.TM_CCOEFF_NORMED)
    assert upper.shape==actual.shape
    assert np.all(upper+.01>=actual)
    proposed=search.propose(sd,td,15,patch)
    if proposed is not None:
        _,score,_,(x,y)=cv.minMaxLoc(actual)
        accepted=lambda result:result[0]>=.9 and abs(result[1]-2)<=1
        assert accepted(proposed)==accepted((score,x,y))
        if accepted(proposed):
            assert proposed[1:]==(x,y)


def test_repeated_rows_remain_uncertain_or_choose_original_peak():
    rng=np.random.default_rng(44)
    target=rng.integers(0,256,(400,720),dtype=np.uint8)
    target[200:224]=target[30:54]
    sd,td=media._LocalSearchData(target.copy()),media._LocalSearchData(target)
    patch=sd.gray[30:54,2:-2]
    found=engine().propose(sd,td,30,patch)
    if found is not None:
        cv=media._local_cv()
        _,_,_,(x,y)=cv.minMaxLoc(cv.matchTemplate(target,patch,cv.TM_CCOEFF_NORMED))
        assert found[1:]==(x,y)


@pytest.mark.parametrize('threshold', [.5,.8,.95,1.])
def test_configured_threshold_and_partial_visibility_keep_original_location(threshold):
    rng=np.random.default_rng(62)
    pixels=rng.integers(0,256,(220,400,3),dtype=np.uint8)
    config=media.KeyframeConfig(local_template_threshold=threshold)
    reference=lambda:media._LocalTrack(0,pixels.copy(),0,1.,'local',0)
    slow=media._locate_local_content(reference(),pixels[43:143],replace(config,local_bounded_search=False))
    fast=media._locate_local_content(reference(),pixels[43:143],config)
    assert (fast is None)==(slow is None)
    if fast is not None:
        assert fast[0]==slow[0]==43


def test_projection_cache_is_bounded_and_released_after_close():
    rng=np.random.default_rng(5)
    work=media._LocalWork(media.KeyframeConfig())
    search=work.coarse
    for _ in range(12):
        search.prepare(media._LocalSearchData(rng.integers(0,256,(100,400),dtype=np.uint8)))
        assert len(search.cache)<=8
    work.close()
    assert not search.cache and work.coarse is None and not work.peaks


def test_failure_releases_projected_images_and_records_counts(tmp_path,monkeypatch):
    rng=np.random.default_rng(13)
    work=media._LocalWork(media.KeyframeConfig())
    search=work.coarse
    data=media._LocalSearchData(rng.integers(0,256,(180,400),dtype=np.uint8))
    search.prepare(data)
    monkeypatch.setattr(search,'bounds',lambda *args: (_ for _ in ()).throw(ValueError('expected')))
    try:
        with pytest.raises(ValueError,match='expected'):
            work.peak(data,data,20,data.gray[20:44,2:-2])
    finally:
        work.close()
    assert not search.cache and work.coarse is None


def test_bounded_search_can_be_disabled_without_changing_other_defaults():
    config=media.KeyframeConfig(local_bounded_search=False)
    work=media._LocalWork(config)
    assert work.coarse is None and config.sample_fps==3 and config.max_frames==15
    with pytest.raises(ValueError):
        media.KeyframeConfig(local_bounded_search='yes')


@pytest.mark.parametrize('base', [0,240])
def test_near_flat_images_fall_back_instead_of_using_unstable_normalization(base):
    rng=np.random.default_rng(34)
    target=rng.integers(base,base+16,(180,400),dtype=np.uint8)
    data=media._LocalSearchData(target)
    search=engine()
    assert search.propose(data,data,20,target[20:44,2:-2]) is None
    assert search.work.counts['coarse_broad_fallback']==1


def test_near_threshold_score_falls_back_to_original_full_search(monkeypatch):
    rng=np.random.default_rng(73)
    target=rng.integers(0,256,(180,400),dtype=np.uint8)
    data=media._LocalSearchData(target)
    search=engine()
    original=search.work.native_match
    def narrow_bounds(*args):
        bounds=np.zeros((157,5),dtype=np.float64)
        bounds[20]=1
        return bounds
    monkeypatch.setattr(search,'bounds',narrow_bounds)
    def near_threshold(image,patch):
        matches,(_,x,y)=original(image,patch)
        return matches,(.9005,x,y)
    monkeypatch.setattr(search.work,'native_match',near_threshold)
    assert search.propose(data,data,20,target[20:44,2:-2]) is None
    assert search.work.counts['coarse_ambiguous_fallback']==1


def test_structured_region_can_use_bounded_original_size_refinement():
    rng=np.random.default_rng(91)
    target=np.full((180,400),245,dtype=np.uint8)
    target[20:44]=np.repeat(rng.integers(20,180,(24,32),dtype=np.uint8),13,axis=1)[:,:400]
    data=media._LocalSearchData(target)
    search=engine()
    found=search.propose(data,data,20,target[20:44,2:-2])
    assert found is not None and found[1:]==(2,20)
    assert search.work.counts['coarse_refined_results']==1
