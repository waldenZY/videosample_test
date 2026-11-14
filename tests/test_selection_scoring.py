from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image

from videosample import media


def fixture(tmp_path):
    candidates=[]
    for index,timestamp in enumerate([0,5,10,99]):
        path=tmp_path/f'{index}.png';Image.new('RGB',(96,160),(index*40,0,0)).save(path)
        candidates.append(media._Candidate(timestamp,path,'3fps',support_s=1))
    candidates[-1].keep_reason='final-frame'
    analysis=media._LocalAnalysis(candidates)
    analysis.representatives=list(candidates)
    analysis.coverage={id(c):{str(i)} for i,c in enumerate(candidates)}
    analysis.units={str(i):{'id':str(i),'weight':weight,'kind':'local','track':i}
                    for i,weight in enumerate([1,3,1,1])}
    analysis.quality={id(c):(0,0,0,0) for c in candidates}
    analysis.quality[id(candidates[1])]=(2,0,0,0)
    analysis.descriptors={id(c):np.zeros((2,2),np.float32) for c in candidates}
    ids={id(c):str(i) for i,c in enumerate(candidates)}
    return candidates,analysis,ids


def test_ratio_removes_additive_group_count_advantage(tmp_path):
    candidates,analysis,ids=fixture(tmp_path)
    analysis.quality={id(c):(0,0,0,0) for c in candidates}
    analysis.units['extra']={'id':'extra','weight':3,'kind':'local','track':8}
    analysis.coverage[id(candidates[1])].add('extra')
    old=media._choose_with_scroll(candidates,analysis,[candidates[0],candidates[3]],ids,limit=3)
    ratio=media._choose_with_scroll(candidates,analysis,[candidates[0],candidates[3]],ids,limit=3,
                                    scoring='new_content_ratio')
    assert old['selected_ids']==['0','1','3']
    assert ratio['selected_ids']==['0','2','3']
    # 分组越多不能让完全新增的画面超过另一张同样完全新增的画面。
    scorer=media._SelectionScorer(analysis,analysis.coverage,'new_content_ratio')
    assert scorer.metrics(candidates[1],set())['score']==scorer.metrics(candidates[2],set())['score']==1


def test_background_only_gain_does_not_look_like_full_new_foreground(tmp_path):
    candidates,analysis,_=fixture(tmp_path)
    analysis.units['1']['kind']='background'
    scorer=media._SelectionScorer(analysis,analysis.coverage,'new_content_ratio')
    assert scorer.metrics(candidates[1],set())['score']==0
    assert scorer.metrics(candidates[2],set())['score']==1


@pytest.mark.parametrize('method',['content_weight','new_content_ratio'])
def test_real_competitor_is_not_chronological_last_frame(tmp_path,method):
    candidates,analysis,_=fixture(tmp_path)
    with media.record_process(tmp_path/'process') as recorder:
        media._stage('content','content',[],candidates)
        selected=media._limit_candidates(candidates,3,analysis=analysis,
            config=media.KeyframeConfig(scroll_dedup_enabled=False,selection_scoring=method))
        media._stage('budget','budget',candidates,selected)
    event=next(frame for frame in recorder.frames if frame['time']==10)['history'][-1]
    winner=recorder.identify(candidates[1]);end=recorder.identify(candidates[3])
    assert event['related']==[winner] and end not in event['related']
    comparison=event['details']['selection_comparison']
    assert comparison['winner']['frame_id']==winner and comparison['selection_slot']==3
    assert comparison['candidate_score']['method']==method
    assert 'last_selected_comparison' not in event['details']
    assert comparison['winner']['action']=='select'
    json.dumps(event,ensure_ascii=False)


def test_anchor_only_limit_has_no_fabricated_competitor(tmp_path):
    candidates,analysis,_=fixture(tmp_path)
    with media.record_process(tmp_path/'process') as recorder:
        media._stage('content','content',[],candidates)
        selected=media._limit_candidates(candidates,2,analysis=analysis,
            config=media.KeyframeConfig(scroll_dedup_enabled=False))
        media._stage('budget','budget',candidates,selected)
    event=next(frame for frame in recorder.frames if frame['time']==10)['history'][-1]
    assert event['related']==[]
    assert event['details']['comparison_status']=='anchors_filled_limit'
    assert 'selection_comparison' not in event['details']


def test_same_round_comparison_is_not_recomputed_after_winner_is_added(tmp_path):
    candidates,analysis,_=fixture(tmp_path)
    analysis.coverage[id(candidates[2])]=set(analysis.coverage[id(candidates[1])])
    with media.record_process(tmp_path/'process') as recorder:
        media._stage('content','content',[],candidates)
        selected=media._limit_candidates(candidates,15,analysis=analysis,
            config=media.KeyframeConfig(scroll_dedup_enabled=False,selection_scoring='new_content_ratio'))
        media._stage('budget','budget',candidates,selected)
    event=next(frame for frame in recorder.frames if frame['time']==10)['history'][-1]
    assert event['details']['selection_comparison']['candidate_score']['new_content_ratio']==1
    assert event['details']['final_remaining_score']['new_content_ratio']==0


def test_recording_competitions_does_not_change_decisions_or_comparisons(tmp_path):
    candidates,analysis,ids=fixture(tmp_path)
    calls=[]
    def compare(a,b):calls.append((a,b));return {'decision':'keep'}
    first=media._choose_with_scroll(candidates,analysis,[candidates[0]],ids,compare,limit=3)
    baseline=list(calls);calls.clear()
    second=media._choose_with_scroll(candidates,analysis,[candidates[0]],ids,compare,limit=3,
                                     record_competitions=True)
    assert first['selected_ids']==second['selected_ids'] and calls==baseline
    assert not first['last_competitions'] and second['last_competitions']


def test_configuration_defaults_to_existing_score_and_rejects_invalid_method():
    assert media.KeyframeConfig().selection_scoring=='content_weight'
    with pytest.raises(ValueError):media.KeyframeConfig(selection_scoring='invalid')
