"""不启动服务，检查新旧结果的统计和停止原因能被前端渲染。"""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_adaptive_statistics_and_legacy_records_render_without_server():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node 仅用于前端回归，不是平台运行依赖')
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{hidden:false,innerHTML:'',textContent:'',value:'',open:false});return elements.get(id);}
const context={document:{getElementById:element},fetch:()=>new Promise(()=>{}),setTimeout:()=>0,clearTimeout:()=>{}};
vm.createContext(context);vm.runInContext(fs.readFileSync('static/app.js','utf8'),context);
vm.runInContext(`renderSelection({frame_count:14,selection:{limit:15,representative_count:18,selected_after_dedup:14,uncovered_count:3,stop_reason:'limit_reached'}})`,context);
assert.equal(element('selectionInfo').hidden,false);
assert(element('selectionInfo').innerHTML.includes('达到张数上限'));
vm.runInContext(`traceData={stages:[{id:'budget',title:'自适应选图',input_ids:[],output_ids:[],removed_ids:[],events:[]}],frames:[]}`,context);
element('showMissingFrames').onclick();assert.equal(element('frameFilter').value,'dropped');
vm.runInContext(`renderSelection({frame_count:8,selection:{limit:15,representative_count:12,selected_count:8,uncovered_count:0,stop_reason:'covered'}})`,context);
assert(element('selectionInfo').innerHTML.includes('最终 8 张'));
vm.runInContext(`renderSelection({selection:{limit:15,representative_count:20,selected_count:14,uncovered_count:0,stop_reason:'covered',scroll_dedup:{enabled:true,checked_frames:17,comparisons:31,merged_frames:3,budget_exhausted:true}}})`,context);
assert(element('selectionInfo').innerHTML.includes('入选前滑动检查'));
assert(element('selectionInfo').innerHTML.includes('合并 3 张'));
assert(element('selectionInfo').innerHTML.includes('未确认重复的候选仍正常参选'));
vm.runInContext('renderSelection({frame_count:10})',context);assert.equal(element('selectionInfo').hidden,true);
const value=context.renderDistribution({output_distribution:{samples:2,counts:{'1-5':1,'6-10':0,'11-14':1,'15':0,other:0}},adaptive_selection:{samples:2,hit_limit_count:1,hit_limit_rate:.5,stop_reasons:{covered:1,no_reliable_addition:0,limit_reached:1}}});
assert(value.includes('50.0%'));assert(value.includes('11-14 张：1 题'));
assert(context.renderDistribution({}).includes('暂无张数分布'));
vm.runInContext(`renderTimings({local_analysis:{elapsed_s:2.31,input_frames:89,original_position_hits:18,full_search_attempts:30,failed_pairs:12,template_calls:90,template_cache_hits:4,preparation_reuses:20,match_workers:2}})`,context);
assert.equal(element('localPerformance').hidden,false);
assert(element('localPerformance').innerHTML.includes('2.31 秒'));
assert(element('localPerformance').innerHTML.includes('完整搜索 30 次'));
assert(element('localPerformance').innerHTML.includes('不是丢帧'));
assert(element('localPerformance').innerHTML.includes('该版本未记录局部耗时细分'));
vm.runInContext(`renderLocalPerformance({elapsed_s:10,input_frames:2,template_search_work_s:5,timing_breakdown:{comparison_wall_s:6,background_strict_wall_s:1,other_preparation_wall_s:3,position_search_work_s:8,strict_check_work_s:4},slowest_frames:[{frame_id:'f00002',candidate_index:1,time:12.3,elapsed_s:3,completed:true,pair_attempts:8,full_search_attempts:5,template_calls:30,width_mismatches:3,position_search_work_s:2,strict_check_work_s:1,source_size:[720,1592],regions:[[600,1000]],largest_reference_size:[600,2000]}]})`,context);
const local=element('localPerformance').innerHTML;
assert(local.includes('搜索位置累计 8.00 秒'));assert(local.includes('严格核对累计 4.00 秒'));
assert(local.includes('其它准备 3.00 秒'));assert(local.includes('不能与实际耗时相加'));
assert(local.includes('最慢的 1 张候选图'));assert(local.includes('data-local-frame="f00002"'));
assert(local.includes('600 × 2000 像素'));assert(!local.includes('NaN'));
vm.runInContext(`let clicked=null;openFrame=id=>{clicked=id}`,context);
element('localPerformance').onclick({target:{closest:()=>({dataset:{localFrame:'f00002'}})}});
assert.equal(vm.runInContext('clicked',context),'f00002');
vm.runInContext(`renderLocalPerformance({elapsed_s:5,input_frames:63,coarse_search:{enabled:true,coarse_attempts:100,coarse_negative_results:50,coarse_refined_results:30,coarse_fallbacks:20,fallback_rate:.2,template_full_calls:20,coarse_refine_calls:40,bound_work_s:1.2}})`,context);
assert(element('localPerformance').innerHTML.includes('小图限制搜索范围'));
assert(element('localPerformance').innerHTML.includes('20.0%'));
assert(element('localPerformance').innerHTML.includes('原图全范围搜索 20 次'));
assert(element('localPerformance').innerHTML.includes('需搜索位置的比较'));
vm.runInContext(`renderLocalPerformance({elapsed_s:3,input_frames:63,reference_filter:{enabled:true,reference_filter_checks:100,reference_filter_rejections:70,reference_width_skips:20,work_s:.2}})`,context);
assert(element('localPerformance').innerHTML.includes('参考快速排除'));
assert(element('localPerformance').innerHTML.includes('跳过重搜索 70 次'));
assert(element('localPerformance').innerHTML.includes('帧仍可参选'));
vm.runInContext(`renderLocalPerformance({elapsed_s:0,input_frames:0,coarse_search:{enabled:true,coarse_attempts:0,fallback_rate:null}})`,context);
assert(!element('localPerformance').innerHTML.includes('NaN'));
vm.runInContext('renderTimings({})',context);
assert.equal(element('localPerformance').hidden,true);
vm.runInContext(`current={chains:[{name:'实验',statistics:{successful_samples:1,selected_samples:1,failed_samples:0,local_analysis:{elapsed_s:{samples:1,mean:2.31},original_position_hits:{mean:18},full_search_attempts:{mean:30}}}}]};renderChainStatistics()`,context);
assert(element('chainStatisticsRows').innerHTML.includes('局部检查平均 2.31 秒'));
vm.runInContext(`current={chains:[{name:'旧记录',statistics:{successful_samples:1,selected_samples:1,failed_samples:0}}]};renderChainStatistics()`,context);
assert(element('chainStatisticsRows').innerHTML.includes('该版本未记录'));
assert.equal(context.renderLocalCostAverage({}), '');
const averages=context.renderLocalCostAverage({timing_breakdown:{comparison_wall_s:{samples:2,mean:3},position_search_work_s:{samples:1,mean:4}}});
assert(averages.includes('平均局部开销'));assert(averages.includes('3.00 秒（n=2）'));
assert(averages.includes('4.00 秒（n=1）'));assert(averages.includes('不与实际耗时相加'));
const competition=context.renderScoreComparison({selection_comparison:{round:3,selection_slot:15,candidate_score:{method:'new_content_ratio',score:.5,new_content_ratio:.5,new_content_weight:1,foreground_group_count:1},winner:{frame_id:'f00009',time:9,action:'select',score:{score:1,new_content_ratio:1,new_content_weight:3,foreground_group_count:3}},deciding_factor:{label:'新增内容比例',candidate_value:.5,winner_value:1}}});
assert(competition.includes('选择第 15 张之前'));assert(competition.includes('新增比例（试验）'));
assert(competition.includes('50.0%'));assert(competition.includes('100.0%'));
assert(competition.includes('data-peer="f00009"'));assert(competition.includes('新增内容比例'));
assert(context.renderScoreComparison({comparison_status:'anchors_filled_limit'}).includes('首尾位置已占满'));
assert(context.renderScoreComparison({last_selected_comparison:{round:15}}).includes('旧版只保存'));
assert.equal(context.renderScoreComparison({}), '');
const html=fs.readFileSync('static/index.html','utf8');
assert(html.includes('自适应，最多 15 张'));assert(html.includes('数量分布 / 停止情况'));
assert(html.includes('新增权重（现有）'));assert(html.includes('新增比例（试验）'));
'''
    subprocess.run([node, '-e', script], cwd=root, check=True, capture_output=True, text=True, timeout=15)
