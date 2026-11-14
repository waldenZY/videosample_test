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
vm.runInContext('renderTimings({})',context);
assert.equal(element('localPerformance').hidden,true);
vm.runInContext(`current={chains:[{name:'实验',statistics:{successful_samples:1,selected_samples:1,failed_samples:0,local_analysis:{elapsed_s:{samples:1,mean:2.31},original_position_hits:{mean:18},full_search_attempts:{mean:30}}}}]};renderChainStatistics()`,context);
assert(element('chainStatisticsRows').innerHTML.includes('局部检查平均 2.31 秒'));
vm.runInContext(`current={chains:[{name:'旧记录',statistics:{successful_samples:1,selected_samples:1,failed_samples:0}}]};renderChainStatistics()`,context);
assert(element('chainStatisticsRows').innerHTML.includes('该版本未记录'));
const html=fs.readFileSync('static/index.html','utf8');
assert(html.includes('自适应，最多 15 张'));assert(html.includes('数量分布 / 停止情况'));
'''
    subprocess.run([node, '-e', script], cwd=root, check=True, capture_output=True, text=True, timeout=15)
