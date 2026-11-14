"""Selection-time deduplication must free slots without chaining or unsafe drops."""
from types import SimpleNamespace
import numpy as np
import pytest
from videosample import media

def fixture_candidates(tmp_path):
    from PIL import Image
    candidates=[]
    for i in range(4):
        path=tmp_path/f'{i}.png'
        Image.new('RGB',(96,160),(i*30,0,0)).save(path)
        candidates.append(media._Candidate(i,path,'3fps',support_s=1))
    coverage={id(c):{str(i)} for i,c in enumerate(candidates)}
    analysis=media._LocalAnalysis(candidates)
    analysis.representatives=list(candidates)
    analysis.coverage=coverage
    analysis.units={str(i):{'id':str(i),'weight':4-i,'kind':'local'} for i in range(4)}
    analysis.quality={id(c):(0,) for c in candidates}
    analysis.descriptors={id(c):np.zeros((2,2),np.float32) for c in candidates}
    return candidates,analysis

def test_duplicate_frees_slot_and_records_actual_representative(tmp_path,monkeypatch):
    candidates,analysis=fixture_candidates(tmp_path)
    def compare(self,a,b):
        self.counts['judge']+=1
        result={'a':a,'b':b,'decision':'merge' if (a,b)==('c00001','c00002') else 'keep'}
        if result['decision']=='merge':result['representative']='a'
        self.logs.append(result)
        return result
    monkeypatch.setattr(media._ScrollEngine,'compare',compare)
    # Start anchor plus content 1; duplicate 2 must give way to content 3.
    selected=media._limit_candidates(candidates,3,analysis=analysis)
    assert selected==[candidates[0],candidates[1],candidates[3]]
    assert analysis.replacements[id(candidates[2])] is candidates[1]
    assert analysis.coverage[id(candidates[1])]=={'1','2'}
    assert analysis.summary['scroll_dedup']['merged_frames']==1
    assert not analysis.summary['hit_limit']

def test_failed_comparison_releases_image_caches(tmp_path,monkeypatch):
    candidates,analysis=fixture_candidates(tmp_path);engines=[]
    original=media._ScrollEngine.__init__
    def init(self,*args,**kwargs):original(self,*args,**kwargs);engines.append(self)
    def fail(self,a,b):
        self.raw(a);self.view(a)
        raise RuntimeError('comparison failed')
    monkeypatch.setattr(media._ScrollEngine,'__init__',init)
    monkeypatch.setattr(media._ScrollEngine,'compare',fail)
    with pytest.raises(RuntimeError,match='comparison failed'):
        media._limit_candidates(candidates,3,analysis=analysis)
    assert engines and not engines[0].views and not engines[0].raw_cache and not engines[0].features

def test_unchecked_at_limit_has_truthful_trace(tmp_path,monkeypatch):
    candidates,analysis=fixture_candidates(tmp_path)
    monkeypatch.setattr(media._ScrollEngine,'compare',lambda self,a,b:{'decision':'keep'})
    with media.record_process(tmp_path/'process') as recorder:
        media._stage('content','content',[],candidates)
        selected=media._limit_candidates(candidates,2,analysis=analysis)
        media._stage('budget','budget',candidates,selected)
    dropped=next(f for f in recorder.frames if f['time']==3)['history'][-1]
    assert dropped['status']=='dropped' and '上限' in dropped['reason']
    assert dropped['details']['scroll_checked'] is False
    assert dropped['details']['scroll_comparisons']==[]

def test_budget_exhaustion_preserves_unconfirmed_states(tmp_path):
    candidates,analysis=fixture_candidates(tmp_path)
    ids={id(c):str(i) for i,c in enumerate(candidates)}
    r=media._choose_with_scroll(candidates,analysis,[candidates[0]],ids,
        lambda a,b:pytest.fail('zero budget must not compare'),limit=3,max_checks=0)
    baseline=media._choose_with_scroll(candidates,analysis,[candidates[0]],ids,limit=3)
    assert r['selected_ids']==baseline['selected_ids'] and r['budget_exhausted']

def test_replacement_checks_all_original_members(tmp_path):
    candidates,analysis=fixture_candidates(tmp_path)
    # No anchors in this isolated selector check; B can represent A, C matches
    # B but not A, so C must remain independent rather than extending a chain.
    ids={id(c):str(i) for i,c in enumerate(candidates)}
    analysis.quality={id(c):(i,) for i,c in enumerate(candidates)}
    analysis.representatives=candidates[:3]
    matches={('0','1'),('1','2')}
    def compare(a,b):return {'decision':'merge','representative':'b'} if (a,b) in matches else {'decision':'keep'}
    r=media._choose_with_scroll(candidates[:3],analysis,[],ids,compare,limit=3)
    assert '1' in r['selected_ids'] and '2' in r['selected_ids']
    assert r['replacement_checks']>=1

@pytest.mark.parametrize('values',[{'scroll_max_related':0},{'scroll_checks_per_candidate':True},
 {'scroll_analysis_max_edge':32},{'scroll_agent_indicator_box':(1,0,0,1)}])
def test_invalid_scroll_configuration_is_rejected(values):
    with pytest.raises(ValueError):media.KeyframeConfig(**values)

def test_small_inputs_are_not_deleted_using_phone_pixel_tolerances(tmp_path):
    candidates,_=fixture_candidates(tmp_path)
    engine=media._ScrollEngine([{'id':str(i),'image':c.path} for i,c in enumerate(candidates)],tmp_path)
    try:
        result=engine.compare('0','1')
        assert result['decision']=='uncertain' and '尺寸' in result['reason']
        assert not engine.features
    finally:
        engine.close()
