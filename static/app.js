const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={imported:'待开始',running:'运行中',done:'已结束',interrupted:'已中断',ready:'待开始',queued:'排队中',skipped:'未勾选',error:'失败'};
const oldStageTitles={collected:'最初收集',timestamps:'合并相近时刻',range:'确定任务范围',states:'合并重复页面',endpoints:'保留任务首尾',budget:'控制图片数量',final:'最终结果'};
function chainStages(run){const found=new Map();for(const row of run.rows){for(const stage of row.stages||Object.keys(row.stage_counts||{}).map(id=>({id,title:oldStageTitles[id]||id})))if(!found.has(stage.id))found.set(stage.id,stage);}return [...found.values()];}
function closeSample(){
 detailToken++;$('sample').hidden=true;$('video').pause();$('video').removeAttribute('src');
 if($('frameDialog').open)$('frameDialog').close();
 $('sampleParking').append($('sample'));currentIndex=null;traceData=null;
}
let current=null, traceData=null, currentIndex=null, stageId='final', chosenFrame=null;
let selected=new Set(), timer=null, loading=false, detailToken=0;
async function api(url,options={}){const r=await fetch(url,options);if(!r.ok){let e;try{e=await r.json()}catch{e={detail:r.statusText}}throw Error(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail))}return r.json()}
function notice(text,error=false){$('notice').hidden=!text;$('notice').textContent=text;$('notice').className=error?'error':''}
function post(url,data={}){return api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})}
function rowName(row){return row.item.id||`样本 ${row.index+1}`}
function frameURL(id,thumb=false){return `/api/runs/${current.id}/items/${currentIndex}/frames/${encodeURIComponent(id)}${thumb?'?thumb=true':''}`}
function frameById(id){return traceData.frames.find(f=>f.id===id)}
function sourceLabel(source){if(source.startsWith('start'))return '任务开始画面';if(source.startsWith('protected'))return '开头 / 结尾保护采样';if(source.startsWith('scene'))return '发现画面变化';if(source.startsWith('terminal'))return '视频末尾画面';if(source==='task-end')return '指定任务结束画面';return '沿途采样'}
async function refreshRuns(){const runs=await api('/api/runs');$('runs').innerHTML=runs.length?runs.map(r=>`<button class="run-card ${current?.id===r.id?'active':''}" data-run="${r.id}"><strong>${esc(r.name)}</strong><small>${esc(r.created_at.replace('T',' ').slice(0,16))}</small><small>${r.chains?.length===2?'base + 实验':esc(r.algorithm?.name||'旧记录')}</small><small>${labels[r.status]||esc(r.status)} · ${r.done}/${r.count} 条结果完成</small></button>`).join(''):'<p class="small">还没有记录</p>'}
async function selectRun(id){clearTimeout(timer);detailToken++;$('sample').hidden=true;$('video').pause();$('video').removeAttribute('src');traceData=null;currentIndex=null;current=await api(`/api/runs/${id}`);$('runMode').value=current.run_mode||'current';$('frameLimit').value='15';$('frameLimit').disabled=$('runMode').value==='base';$('adaptiveOption').hidden=$('runMode').value==='base';selected=new Set(current.rows.map(r=>r.index));renderBatch();await refreshRuns();poll()}
function renderBatch(){if(!current)return;renderChainTabs();renderChainStatistics();const stages=chainStages(current);$('rowHead').innerHTML=`<tr><th><input id="selectAll" type="checkbox" aria-label="全选"></th><th>样本 / 任务</th><th>进度</th>${stages.map(s=>`<th>${esc(s.title)}</th>`).join('')}<th>最终图片</th><th>耗时</th></tr>`;$('empty').hidden=true;$('batch').hidden=false;$('batchTitle').textContent=current.name;$('batchMeta').textContent=`${current.algorithm?.name||'旧记录'} · ${current.version}${current.algorithm?.source_sha256?' · '+current.algorithm.source_sha256.slice(0,8):''} · ${current.created_at.replace('T',' ')}${current.sheet?' · 工作表 '+current.sheet:''}`;
 const counts={};for(const r of current.rows)counts[r.status]=(counts[r.status]||0)+1;
 $('summary').innerHTML=[['有效样本',current.rows.length],['已完成',counts.done||0],['运行 / 等待',(counts.running||0)+(counts.queued||0)],['失败 / 中断',(counts.error||0)+(counts.interrupted||0)]].map(([title,count])=>`<div class="stat"><span>${title}</span><strong>${count}</strong></div>`).join('');
 const imported=current.status==='imported';$('start').hidden=!imported;$('start').disabled=!selected.size;$('startOptions').hidden=!imported;if(document.activeElement!==$('runName'))$('runName').value=current.name;
 const frameLimit=current.algorithm?.overrides?.max_frames;if(frameLimit)$('batchMeta').textContent+=` · ${current.algorithm?.selection_policy?'自适应，':''}最多 ${frameLimit} 张`;
 const messages=[...current.errors,...current.warnings.map(w=>JSON.stringify(w))];$('importDetails').hidden=!messages.length;$('importMessages').textContent=messages.join('\n');
 const f=$('rowFilter').value;
 const rows=current.rows.filter(r=>f==='all'||(f==='error'?['error','interrupted'].includes(r.status):f==='running'?['running','queued'].includes(r.status):r.status===f));
 // 进度刷新会重建列表，先移出详情以保留图片、筛选与事件绑定。
 $('sampleParking').append($('sample'));
 if(currentIndex!==null&&!rows.some(r=>r.index===currentIndex))closeSample();
 $('rows').innerHTML=rows.map(r=>`<tr class="clickable ${currentIndex===r.index?'expanded':''}" data-index="${r.index}"><td><input type="checkbox" data-check="${r.index}" ${selected.has(r.index)?'checked':''} ${imported?'':'disabled'} aria-label="选择 ${esc(rowName(r))}"></td><td class="task-cell"><button class="sample-toggle" aria-expanded="${currentIndex===r.index}" aria-controls="sample-detail-${r.index}" aria-label="${currentIndex===r.index?'收起':'展开'} ${esc(rowName(r))}"><span aria-hidden="true">${currentIndex===r.index?'▾':'▸'}</span> ${esc(rowName(r))}</button><small>${esc(r.item.query)}</small></td><td><span class="status ${r.status}">${labels[r.status]||esc(r.status)}</span><div class="muted small">${esc(r.message)}</div>${r.error?`<div class="error-text">${esc(r.error)}</div>`:''}${r.warnings?.length?`<div class="error-text">${esc(r.warnings.join('；'))}</div>`:''}</td>${stages.map(s=>`<td>${r.stage_counts?.[s.id]??'<span class="muted">—</span>'}</td>`).join('')}<td>${r.frame_count??'—'}</td><td>${r.elapsed_s===undefined?'—':r.elapsed_s+'s'}</td></tr><tr class="sample-detail-row" id="sample-detail-${r.index}" hidden><td colspan="${stages.length+5}"><div id="sample-slot-${r.index}"></div></td></tr>`).join('');
 if(currentIndex!==null&&!$('sample').hidden){renderTimings(current.rows[currentIndex]);$(`sample-slot-${currentIndex}`).append($('sample'));$(`sample-detail-${currentIndex}`).hidden=false;}
 $('selectAll').disabled=!imported;$('selectAll').checked=!!current.rows.length&&selected.size===current.rows.length;
}
function poll(){clearTimeout(timer);if(current&&(current.status==='running'||current.chains?.some(c=>c.status==='running')))timer=setTimeout(async()=>{try{const id=current.id;const updated=await api(`/api/runs/${id}`);if(current?.id!==id)return;current=updated;renderBatch();if(current.status!=='running'&&!current.chains?.some(c=>c.status==='running'))await refreshRuns();poll()}catch(e){notice(e.message,true);timer=setTimeout(poll,2500)}},1500)}
async function openSample(index){if(currentIndex===index&&!$('sample').hidden){closeSample();renderBatch();return;}closeSample();const token=++detailToken;currentIndex=index;const row=current.rows[index];$('sample').hidden=false;$('sampleTitle').textContent=`${rowName(row)} · ${current.algorithm?.key==='base'?'base 固定链路':'实验链路'}`;$('sampleQuery').textContent=row.item.query;$('sampleInfo').textContent='正在读取过程结果…';$('gallery').innerHTML='';$('stages').innerHTML='';$('videoDetails').open=false;renderBatch();
 try{const data=await api(`/api/runs/${current.id}/items/${index}/trace`);if(token!==detailToken)return;traceData=data;stageId=data.stages.at(-1)?.id||'collected';$('frameFilter').value='kept';$('sampleInfo').textContent=`${current.version} · ${row.duration?.toFixed(2)||'—'} 秒视频 · ${data.frames.length} 张候选图 · 最终 ${row.frame_count??'尚未完成'} 张${row.error?' · '+row.error:''}`;renderStages();renderGallery()}catch(e){if(token===detailToken)$('sampleInfo').textContent=row.error||e.message}}
function renderTimings(row){
 renderSelection(row);
 const timings=row?.timings_s;$('timingDetails').hidden=!timings;
 if(!timings){$('timingBreakdown').innerHTML='';return;}
 const labels={video_prepare:'视频准备 / 下载及等待',probe:'读取视频信息',fingerprint:'核对视频内容',extraction:'抽帧处理合计'};
 const stageLabels={collected:'取图与候选准备',timestamps:'合并相近时刻',range:'确定任务范围',states:'同页内容筛选',endpoints:'保留首尾',content:'合并重复内容',review:'复核前段相近内容',local:'局部检查与选择代表',budget:'分配图片名额',final:'全局去重',process_io:'保存过程图片与记录',other_processing:'其余处理与最终保存',unfinished:'未完成步骤'};
 const stageNames=new Map((row.stages||[]).map(s=>[s.id,s.title]));
 const lines=Object.entries(labels).filter(([key])=>typeof timings[key]==='number').map(([key,name])=>`<div class="timing-line"><span>${name}</span><strong>${timings[key].toFixed(2)} 秒</strong></div>`).join('');
 const stages=Object.entries(timings.stages||{}).map(([key,value])=>`<div class="timing-line timing-stage"><span>${esc(stageLabels[key]||stageNames.get(key)||key)}</span><strong>${Number(value).toFixed(2)} 秒</strong></div>`).join('');
 $('timingBreakdown').innerHTML=lines+(stages?'<p class="small muted">下面为抽帧合计内部的分项，不与上面的合计重复相加。</p>'+stages:'<p class="small muted">base 保持原实现，仅显示抽帧总体耗时。</p>');
}
function renderStages(){$('stages').innerHTML=traceData.stages.map(s=>`<button data-stage="${s.id}" class="${stageId===s.id?'active':''}">${esc(s.title)} · ${s.output_ids.length}</button>`).join('')}
function renderGallery(){const stage=traceData?.stages.find(s=>s.id===stageId);if(!stage)return;const filter=$('frameFilter').value;$('stageSummary').textContent=stage.id==='collected'?`收集 ${stage.output_ids.length} 张`:`本步输入 ${stage.input_ids.length} 张 · 保留 ${stage.output_ids.length} 张 · 丢弃 ${stage.removed_ids.length} 张`;
 const ids=filter==='kept'?stage.output_ids:filter==='dropped'?stage.removed_ids:[...new Set([...stage.input_ids,...stage.output_ids])];const events=new Map(stage.events.map(e=>[e.frame_id,e]));
 $('gallery').innerHTML=ids.length?ids.map(id=>{const frame=frameById(id),event=events.get(id);return `<button class="frame-card ${event?.status||''}" data-frame="${id}"><img src="${frameURL(id,true)}" loading="lazy" alt="${frame.time} 秒画面"><div class="caption"><strong>${frame.time.toFixed(3)} 秒</strong> <span class="muted">${id}</span><div class="reason">${esc(event?.reason||sourceLabel(frame.source))}</div></div></button>`}).join(''):'<div class="no-frames">本步没有符合筛选条件的图片。</div>';
}
function openFrame(id){const frame=frameById(id);if(!frame)return;chosenFrame=frame;$('frameTitle').textContent=`${frame.time.toFixed(3)} 秒 · ${id}`;$('largeFrame').src=frameURL(id);$('frameSource').textContent=`来源：${sourceLabel(frame.source)}`;$('history').innerHTML=frame.history.map(event=>{const stage=traceData.stages.find(s=>s.id===event.stage);return `<div class="event"><div class="event-title"><strong>${esc(stage?.title||event.stage)}</strong><span class="status ${event.status==='dropped'?'error':'done'}">${event.status==='dropped'?'丢弃':'保留'}</span></div><p>${esc(event.reason)}</p>${event.related.length?`<div class="muted small">相关画面（点击可继续追查）</div><div class="peers">${event.related.map(peer=>{const other=frameById(peer);return `<button class="peer" data-peer="${peer}"><img src="${frameURL(peer,true)}" alt="相关画面" loading="lazy">${other?.time.toFixed(3)} 秒</button>`}).join('')}</div>`:''}${Object.keys(event.details).length?`<details><summary>查看具体比较记录</summary><pre>${esc(JSON.stringify(event.details,null,2))}</pre></details>`:''}</div>`}).join('');if(!$('frameDialog').open)$('frameDialog').showModal();$('frameDialog').scrollTop=0}
$('upload').onchange=async event=>{const file=event.target.files[0];if(!file||loading)return;loading=true;notice('正在导入表格…');try{const form=new FormData();form.append('file',file);const run=await api('/api/import',{method:'POST',body:form});await selectRun(run.id);notice(`已导入 ${run.rows.length} 个有效样本，请预览并开始抽帧。`)}catch(e){notice(e.message,true)}finally{loading=false;event.target.value=''}};
$('start').onclick=async()=>{if(!current)return;$('start').disabled=true;try{current=await post(`/api/runs/${current.id}/start`,{indices:[...selected],name:$('runName').value,run_mode:$('runMode').value,experiment_max_frames:Number($('frameLimit').value)});renderBatch();await refreshRuns();notice(current.chains?.length===2?'已开始两条链路，点击链路名称分别查看。':`已开始${current.algorithm?.key==='base'?' base':'实验'}链路，下面显示的是本次所选链路。`);poll()}catch(e){notice(e.message,true);renderBatch()}};
$('runMode').onchange=()=>{$('frameLimit').disabled=$('runMode').value==='base';$('adaptiveOption').hidden=$('runMode').value==='base'};
$('copyRun').onclick=async()=>{try{const run=await post(`/api/runs/${current.id}/copy`);await selectRun(run.id);notice('已新建一次运行，旧结果保留。')}catch(e){notice(e.message,true)}};
$('runs').onclick=e=>{const button=e.target.closest('[data-run]');if(button)selectRun(button.dataset.run).catch(e=>notice(e.message,true))};
$('rows').onclick=e=>{if(e.target.matches('[data-check]')){const index=Number(e.target.dataset.check);e.target.checked?selected.add(index):selected.delete(index);$('start').disabled=!selected.size;return}const row=e.target.closest('[data-index]');if(row)openSample(Number(row.dataset.index))};
$('rowHead').onchange=e=>{if(e.target.id==='selectAll'){selected=new Set(e.target.checked?current.rows.map(r=>r.index):[]);renderBatch()}};
$('rowFilter').onchange=renderBatch;$('frameFilter').onchange=renderGallery;
$('stages').onclick=e=>{const button=e.target.closest('[data-stage]');if(button){stageId=button.dataset.stage;renderStages();renderGallery()}};
$('gallery').onclick=e=>{const button=e.target.closest('[data-frame]');if(button)openFrame(button.dataset.frame)};
$('history').onclick=e=>{const button=e.target.closest('[data-peer]');if(button)openFrame(button.dataset.peer)};
$('closeDialog').onclick=()=>$('frameDialog').close();$('closeSample').onclick=()=>{closeSample();renderBatch()};
function loadVideo(){const url=`/api/runs/${current.id}/items/${currentIndex}/video`;$('videoLink').href=url;if(!$('video').getAttribute('src'))$('video').src=url}
$('videoDetails').ontoggle=()=>{if($('videoDetails').open)loadVideo();else $('video').pause()};
$('seekVideo').onclick=()=>{const time=chosenFrame.time;$('frameDialog').close();$('videoDetails').open=true;loadVideo();const video=$('video');if(video.readyState>=1)video.currentTime=time;else video.addEventListener('loadedmetadata',()=>{video.currentTime=time},{once:true});video.scrollIntoView({behavior:'smooth'})};
$('refreshRuns').onclick=()=>refreshRuns().catch(e=>notice(e.message,true));
const selectionLabels={covered:'已识别的内容已覆盖',no_reliable_addition:'剩余候选没有可靠新增内容',limit_reached:'达到张数上限，仍有内容未覆盖'};
function renderSelection(row){
 const selection=row?.selection, panel=$('selectionInfo');panel.hidden=!selection;
 if(!selection){panel.textContent='';return;}
 panel.innerHTML=`<strong>${esc(selectionLabels[selection.stop_reason]||selection.stop_reason)}</strong><p>上限 ${esc(selection.limit)} 张 · 代表候选 ${esc(selection.representative_count)} 张 · 最终 ${esc(selection.selected_after_dedup??row.frame_count??'—')} 张 · 未覆盖 ${esc(selection.uncovered_count)} 个图像内容片段</p><div class="small muted">片段按图像区域计算，不等同于业务操作数；局部与裁切线索不代表已经确认卡片完整。</div>${selection.uncovered_count?'<button id="showMissingFrames">查看数量分配中未选的图片</button>':''}`;
 const button=$('showMissingFrames');if(button)button.onclick=()=>{if(!traceData)return;stageId='budget';$('frameFilter').value='dropped';renderStages();renderGallery();};
}
function renderDistribution(stats){
 const distribution=stats.output_distribution, adaptive=stats.adaptive_selection;
 let result=distribution?.samples?Object.entries(distribution.counts).filter(([key,value])=>key!=='other'||value).map(([key,value])=>`${key==='other'?'其他':key+' 张'}：${value} 题`).join(' · '):'暂无张数分布';
 if(adaptive?.samples){const reasons=adaptive.stop_reasons;result+=`<div class="small muted">触顶 ${(adaptive.hit_limit_rate*100).toFixed(1)}%（${adaptive.hit_limit_count}/${adaptive.samples}） · 覆盖后停止 ${reasons.covered} · 无可靠新增 ${reasons.no_reliable_addition}</div>`;}
 else result+='<div class="small muted">该版本未记录自适应停止原因</div>';
 return result;
}
function renderChainStatistics(){
 const chains=current.chains||[];
 function cell(metric,unit,total=false){
  if(!metric||metric.mean===null||metric.mean===undefined)return '<span class="muted">暂无数据</span>';
  return `<strong>${Number(metric.mean).toFixed(unit==='秒'?2:1)} ${unit}/题</strong><div class="muted small">${total?'总计 '+Number(metric.total).toLocaleString()+' '+unit+' · ':''}n=${metric.samples}</div>`;
 }
 $('chainStatisticsRows').innerHTML=chains.map(chain=>{const stats=chain.statistics;if(!stats)return '';return `<tr><td><strong>${esc(chain.name)}${chain.frame_limit?' · 最多 '+chain.frame_limit+' 张':''}</strong><div class="muted small">${esc(chain.version||'')}</div></td><td>${stats.successful_samples} / ${stats.selected_samples}<div class="muted small">失败 / 中断 ${stats.failed_samples}</div></td><td>${cell(stats.sample_time_s,'秒')}</td><td>${cell(stats.extraction_time_s,'秒')}${stats.extraction_max_s==null?'':`<div class="muted small">最长 ${Number(stats.extraction_max_s).toFixed(2)} 秒</div>`}</td><td>${cell(stats.initial_frames,'张',true)}</td><td>${cell(stats.final_frames,'张',true)}</td><td>${renderDistribution(stats)}</td></tr>`}).join('');
}
function renderChainTabs(){
 const chains=current.chains||[];$('chainTabs').hidden=chains.length<2;
 $('chainTabs').innerHTML=chains.map(c=>`<button data-chain="${c.id}" class="${c.id===current.id?'active':''}">${esc(c.name)} · ${labels[c.status]||esc(c.status)} · ${c.done}/${c.count}${c.failed?' · 失败 '+c.failed:''}</button>`).join('');
}
$('chainTabs').onclick=async e=>{const button=e.target.closest('[data-chain]');if(button)try{await selectRun(button.dataset.chain)}catch(error){notice(error.message,true)}};
api('/api/health').then(h=>{$('health').textContent=h.ok?'抽帧环境就绪 · '+h.version:'FFmpeg 未就绪';if(!h.ok)notice(h.error,true)}).catch(e=>notice(e.message,true));refreshRuns().catch(e=>notice(e.message,true));
