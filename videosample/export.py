"""Read-only, sample-level XLSX export using the service's existing Excel dependency."""
import io
import json
import math
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


STATUS = {'done': '已完成', 'error': '失败', 'interrupted': '已中断', 'running': '运行中',
          'queued': '排队中', 'ready': '待开始', 'skipped': '未勾选', 'imported': '待开始'}
REASONS = {'covered': '已识别内容已覆盖', 'no_reliable_addition': '无可靠新增', 'limit_reached': '达到上限仍未覆盖'}
ANNOTATIONS = {'missing': '发现关键帧遗漏', 'no_missing': '未发现关键帧遗漏'}
PHASES = {'video_prepare': '视频准备及下载等待', 'probe': '读取视频信息', 'fingerprint': '视频指纹核对', 'extraction': '抽帧合计'}
STAGES = {'collected': '取图与候选准备', 'timestamps': '合并近时刻', 'range': '任务范围', 'states': '页面筛选',
          'endpoints': '首尾处理', 'content': '内容合并', 'review': '历史复核', 'local': '局部分析',
          'budget': '分配名额', 'final': '历史末尾去重', 'process_io': '过程保存',
          'other_processing': '其余处理', 'unfinished': '未完成步骤'}
LOCAL = {'input_frames': '输入张数', 'elapsed_s': '实际耗时（秒）', 'completed': '分析完成',
         'pair_attempts': '参考比较次数', 'original_position_checks': '原位检查次数',
         'original_position_hits': '原位匹配次数', 'full_search_attempts': '完整搜索次数',
         'failed_pairs': '未匹配次数', 'strict_checks': '严格核对次数', 'template_calls': '图案搜索次数',
         'accepted_matches': '接受匹配次数',
         'template_cache_hits': '搜索缓存复用次数', 'preparations': '图案准备次数',
         'preparation_reuses': '图案准备复用次数', 'width_mismatches': '宽度不符次数',
         'match_workers': '搜索线程上限', 'template_search_work_s': '图案搜索累计（秒）'}
COSTS = {'comparison_wall_s': '参考比较实际（秒）', 'background_strict_wall_s': '背景核对实际（秒）',
         'other_preparation_wall_s': '其它准备实际（秒）', 'position_search_work_s': '搜索位置累计（秒）',
         'strict_check_work_s': '严格核对累计（秒）'}
COARSE = {'enabled': '启用', 'coarse_attempts': '尝试次数', 'coarse_negative_results': '无可行位置次数',
          'coarse_refined_results': '缩小范围次数', 'coarse_fallbacks': '回退次数', 'coarse_small_bypass': '小区域直通次数',
          'coarse_broad_fallback': '宽范围回退次数', 'coarse_ambiguous_fallback': '歧义回退次数',
          'coarse_uncertain_fallback': '不确定回退次数', 'coarse_refine_calls': '原图局部搜索次数',
          'coarse_projection_calls': '投影计算次数', 'template_full_calls': '原图全范围搜索次数',
          'fallback_rate': '回退比例', 'bound_work_s': '预筛累计（秒）'}
REFERENCE = {'enabled': '启用', 'reference_filter_checks': '检查次数', 'reference_filter_rejections': '排除次数',
             'reference_width_skips': '宽度过滤数', 'reference_rows_prepared': '行统计准备次数', 'work_s': '预筛累计（秒）'}
SCROLL = {'enabled': '启用', 'checked_frames': '涉及张数', 'comparisons': '比较次数', 'merged_frames': '合并张数',
          'representative_changes': '代表替换次数', 'check_budget': '比较预算', 'budget_exhausted': '预算用尽'}


def at(value, path):
    for key in path.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def text(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return '未记录'
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return ILLEGAL_CHARACTERS_RE.sub('', value)[:32767] if isinstance(value, str) else value


def append(sheet, values):
    sheet.append([text(v) for v in values])
    # User-provided task/file names are text, even when they start with '='.
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value, str):
            cell.data_type = 's'


def saved_json(path):
    try:
        return json.loads(path.read_text('utf8'))
    except (OSError, ValueError):
        return {}


def slow_summary(local):
    frames = local.get('slowest_frames')
    if frames is None:
        return None
    def shown(value):
        return text(value)
    return '\n'.join(
        f"{f.get('frame_id', '候选')} · {shown(f.get('time'))}秒 · 局部{shown(f.get('elapsed_s'))}秒 · "
        f"参考{shown(f.get('pair_attempts'))}次 / 搜索{shown(f.get('full_search_attempts'))}次 / 图案{shown(f.get('template_calls'))}次 / 宽度不符{shown(f.get('width_mismatches'))}次 · "
        f"位置累计{shown(f.get('position_search_work_s'))}秒 / 严格核对累计{shown(f.get('strict_check_work_s'))}秒 · "
        f"原图{shown(f.get('source_size'))} / 区域{shown(f.get('regions'))} / 最大参考{shown(f.get('largest_reference_size'))}"
        for f in frames)


def workbook_bytes(run, directory: Path, statistics, exported_at):
    algorithm = run.get('algorithm') or {}
    config = {**algorithm.get('defaults', {}), **algorithm.get('overrides', {})}
    wb = Workbook()
    overview = wb.active
    overview.title = '批次概览'
    append(overview, ['项目', '内容'])
    for key, value in [('批次名称', run.get('name')), ('批次编号', run.get('id')), ('输入文件', run.get('filename')),
                       ('创建时间', run.get('created_at')), ('导出时间', exported_at), ('批次状态', STATUS.get(run.get('status'), run.get('status'))),
                       ('链路', algorithm.get('key')), ('算法版本', algorithm.get('algorithm_version', run.get('version'))),
                       ('算法代码校验值', algorithm.get('source_sha256')), ('张数上限', config.get('max_frames')),
                       ('每秒固定采样帧数', config.get('sample_fps')), ('评分策略', 'base 原始方法' if algorithm.get('key') == 'base' else config.get('selection_scoring'))]:
        append(overview, [key, value])
    for key, label in [('total_samples', '全部样本数'), ('selected_samples', '已选样本数'),
                       ('successful_samples', '成功样本数'), ('failed_samples', '失败及中断样本数'), ('extraction_max_s', '最长抽帧耗时（秒）')]:
        append(overview, [label, statistics.get(key)])
    append(overview, ['保留下载视频', run.get('retain_downloaded_video', True)])
    append(overview, ['待开始/等待/运行样本数', sum(r.get('status') in {'ready', 'queued', 'running'} for r in run.get('rows', []))])
    verdicts = [at(row, 'annotation.verdict') for row in run.get('rows', [])]
    append(overview, ['已作人工判断样本数', sum(verdict in ANNOTATIONS for verdict in verdicts)])
    for verdict, label in ANNOTATIONS.items():
        append(overview, [f'人工判断·{label}样本数', verdicts.count(verdict)])
    for key, label in [('sample_time_s', '每题总耗时（秒）'), ('extraction_time_s', '抽帧耗时（秒）'),
                       ('initial_frames', '初始候选张数'), ('final_frames', '最终输出张数')]:
        for sub, name in [('samples', '有效样本数'), ('mean', '平均'), ('total', '合计')]:
            append(overview, [f'{label}·{name}', at(statistics, f'{key}.{sub}')])
    for key, value in (at(statistics, 'output_distribution.counts') or {}).items():
        append(overview, [f'输出分布·{key}张', value])
    for key, value in (at(statistics, 'adaptive_selection.stop_reasons') or {}).items():
        append(overview, [f'停止原因·{REASONS.get(key, key)}', value])
    append(overview, ['因上限停止比例（0～1）', at(statistics, 'adaptive_selection.hit_limit_rate')])
    for key, value in config.items():
        append(overview, [f'保存配置·{key}', value])
    append(overview, ['统计口径', '平均值仅统计成功样本的已记录值；历史缺字段不当作零。导出是本次状态快照，不触发重跑。'])
    append(overview, ['单位与定位说明', '抽帧合计包含内部阶段与过程保存，不重复相加。累计工作时间可能跨线程相加，不与实际经过时间相加。图像片段不是业务操作数，未匹配次数不是漏帧数。'])
    append(overview, ['图片路径说明', '最终图片列是服务器目录路径，不保证能从本地 Excel 直接打开；本文件不含图片或逐帧清单。'])

    rows = []
    stage_titles = {}
    for row in run.get('rows', []):
        item_dir = directory / 'items' / f"{row['index']:05d}"
        stable = row.get('status') in {'done', 'error', 'interrupted'}
        trace = saved_json(item_dir / 'process/trace.json') if stable else {}
        metadata = saved_json(item_dir / 'final/keyframes.json') if row.get('status') == 'done' else {}
        effective = {**config, **metadata.get('config', {}), **(row.get('effective_config') or {})}
        local = row.get('local_analysis') or trace.get('local_analysis') or {}
        selection = row.get('selection') or metadata.get('selection') or trace.get('selection') or {}
        stage_info = row.get('stages') or trace.get('stages') or []
        counts = {s['id']: len(s['output_ids']) for s in trace.get('stages', []) if 'output_ids' in s}
        counts.update(row.get('stage_counts') or {})
        for s in stage_info:
            stage_titles.setdefault(s['id'], s.get('title', s['id']))
        for key in counts:
            stage_titles.setdefault(key, STAGES.get(key, key))
        data = {'row': row, 'config': effective, 'local': local, 'selection': selection, 'counts': counts,
                'inputs': {s['id']: len(s['input_ids']) for s in trace.get('stages', []) if 'input_ids' in s},
                'slow': slow_summary(local), 'version': metadata.get('algorithm_version', algorithm.get('algorithm_version', run.get('version'))),
                'end': metadata.get('effective_task_end_time'),
                'begin': metadata.get('selected', [{}])[0].get('time') if metadata.get('selected') else None,
                'initial': row.get('initial_frame_count', counts.get('collected', trace.get('initial_frame_count'))),
                'stop': REASONS.get(selection.get('stop_reason'), selection.get('stop_reason')),
                'hit': selection.get('stop_reason') == 'limit_reached' if selection.get('stop_reason') in REASONS else None,
                'images': str((item_dir / 'final').resolve()) if row.get('status') == 'done' and (item_dir / 'final').is_dir() else ('未完成' if not stable else '未找到最终目录'),
                'status': STATUS.get(row.get('status'), row.get('status')), 'number': row['index'] + 1,
                'annotation_label': ANNOTATIONS.get(at(row, 'annotation.verdict'), '未标注'),
                'scoring': 'base 原始方法' if algorithm.get('key') == 'base' else effective.get('selection_scoring')}
        rows.append(data)
    columns = [('有效样本序号', 'number'), ('样本编号', 'row.item.id'), ('题目', 'row.item.query'), ('状态', 'status'),
               ('人工判断', 'annotation_label'), ('标注备注', 'row.annotation.note'), ('标注更新时间', 'row.annotation.updated_at'),
               ('错误', 'row.error'), ('视频路径', 'row.item.video_path'), ('视频时长（秒）', 'row.duration'),
               ('下载副本清理提醒', 'row.download_cleanup_error'),
               ('指定开始（秒）', 'row.item.task_start_time'), ('指定结束（秒）', 'row.item.task_end_time'),
               ('实际开始（秒）', 'begin'), ('实际结束（秒）', 'end'), ('算法版本', 'version'),
               ('张数上限', 'config.max_frames'), ('每秒采样帧数', 'config.sample_fps'), ('评分策略', 'scoring'),
               ('初始候选张数', 'initial'), ('最终张数', 'row.frame_count'), ('停止原因', 'stop'), ('因上限停止', 'hit'),
               ('识别片段数', 'selection.identified_content_count'), ('覆盖片段数', 'selection.covered_content_count'),
               ('未覆盖片段数', 'selection.uncovered_count'), ('不确定片段数', 'selection.uncertain_content_count'),
               ('最终图片', 'images'), ('每题总耗时（秒）', 'row.elapsed_s')]
    columns += [(f'{label}（秒）', f'row.timings_s.{key}') for key, label in PHASES.items()]
    for key, label in stage_titles.items():
        columns += [(f'{label}·输入张数', f'inputs.{key}'), (f'{label}·输出张数', f'counts.{key}')]
    timing_keys = dict(STAGES)
    for row in rows:
        for key in at(row, 'row.timings_s.stages') or {}:
            timing_keys.setdefault(key, stage_titles.get(key, key))
    columns += [(f'内部阶段·{label}（秒）', f'row.timings_s.stages.{key}') for key, label in timing_keys.items()]
    for prefix, fields, path in [('局部', LOCAL, 'local'), ('局部耗时', COSTS, 'local.timing_breakdown'),
                                 ('搜索范围预筛', COARSE, 'local.coarse_search'), ('参考快速排除', REFERENCE, 'local.reference_filter'),
                                 ('滑动检查', SCROLL, 'selection.scroll_dedup')]:
        columns += [(f'{prefix}·{label}', f'{path}.{key}') for key, label in fields.items()]
    columns += [('最慢候选摘要（最多3张）', 'slow')]
    samples = wb.create_sheet('样本统计')
    append(samples, [label for label, _ in columns])
    for row in rows:
        append(samples, [at(row, path) for _, path in columns])
    for sheet in wb:
        sheet.freeze_panes = 'D2' if sheet == samples else 'B2'
        sheet.sheet_view.showGridLines = False
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 48
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='244B42')
            cell.font = Font(color='FFFFFF', bold=True)
        for i in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(i)].width = 22 if sheet == samples else (30 if i == 1 else 100)
        for row in sheet.iter_rows(min_row=2):
            if sheet == overview:
                sheet.row_dimensions[row[0].row].height = 42
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if row[0].row % 2 == 0:
                    cell.fill = PatternFill('solid', fgColor='F1F5F2')
                if isinstance(cell.value, float):
                    cell.number_format = '0.000'
        for cell in sheet[1]:
            cell.alignment = Alignment(vertical='center', wrap_text=True)
    samples.column_dimensions['C'].width = 48
    samples.column_dimensions[get_column_letter(len(columns))].width = 85
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
