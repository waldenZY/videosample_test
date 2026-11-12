# 抽帧实验台

从 auto_eval_agent 的现有抽帧方法开始，查看一批视频在每一步留下和丢弃的画面，帮助判断修改是否保住了重要信息。

仓库：<https://github.com/waldenZY/videosample_test>

## 当前状态

- 已建立本地仓库与远程 main 分支的跟踪配置；远程仓初始化时为空。
- `reference/media.py` 保存当前 auto_eval_agent 抽帧模块的原样副本。
- `reference/source.json` 记录来源、版本与文件校验值。
- 已实现文件上传、批跑、各阶段查看、淘汰原因追踪、相关图片与原视频对照。
- 第一版方案见 [docs/方案.md](docs/方案.md)。

原始副本用于后续核对结果。第一版只增加过程记录与查看能力，不顺手修正现有算法；算法修改应保存为新版本，与原始版本比较。

输入及抽帧前处理与 auto_eval_agent 一致，抽帧完成后进入查看页面，不继续评估。实际实现位于 `videosample/media.py`，当前先检查抽帧效果；定稿迁回时允许少量适配，去掉实验用过程记录与中间图片，保留正常抽帧输出。

## 启动

进入本项目目录，在 moto 环境运行（Python 3.10 或更新版本）：

```bash
conda activate moto
python -m pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 8503
```

本机打开 <http://127.0.0.1:8503>；部署到服务器时，本地浏览器访问 `http://服务器地址:8503`。网页无需另行构建。FFmpeg 可使用系统安装或依赖中的 imageio-ffmpeg，首页会检查是否就绪。

## 使用

1. 上传现有 Excel、CSV 或 JSONL，预览样本、错误与提醒。
2. 勾选样本后开始抽帧，默认同时处理两题，一题失败不影响其他题。
3. 点击样本，切换各步骤，选择查看本步保留、本步丢弃或本步全部。
4. 点击图片，查看其处理经过与相关画面；可以定位到原视频相应时刻。
5. 点击“新建一次重跑”复用输入，旧结果保留。当前通过左侧记录切换查看，尚无自动左右对比。

过程记录只针对实际收集到的候选画面；没有抽到的时刻，需要结合原视频检查。原视频是否能在网页播放取决于浏览器支持的编码，不支持时可打开或下载查看。

## 同时处理本地视频与视频链接

`video_path` 可以填写服务器本地路径或 HTTP/HTTPS 视频链接，同一张表可以混用。远程视频先下载再抽帧，支持跳转和失败重试，不限制主机名。下载实现来自 auto_eval 的 `a5b0584` 更新（允许全部远程视频主机），没有调用模型。

上传表格不会同时上传电脑上的视频。相对路径默认从本项目查找；若现有路径相对于 auto_eval 目录，启动前通过原有的 `OPERATION_VIDEO_ROOTS` 规则指定该目录。

Linux：

```bash
export OPERATION_VIDEO_ROOTS=/path/to/auto_eval_agent
python app.py --host 0.0.0.0 --port 8503
```

PowerShell：

```powershell
$env:OPERATION_VIDEO_ROOTS = 'D:\wwc\project_4\auto_eval_agent'
python app.py --host 0.0.0.0 --port 8503
```

## 保存与重跑

结果都在项目 `runs/` 下，包括输入、进度、全部候选图、各步处理记录和最终图片；下载的视频保留供对照，用户原有的本地视频不复制、不删除。删除整个项目即可清除本项目实验数据。

刷新网页不影响批跑，服务重启后可查看保存结果。中断项可新建一次重跑，不自动续跑。

## 验证

在 moto 环境安装 pytest 后运行：

```bash
python -m pytest -q tests
```

测试用生成的视频核对最终图片与原始方法一致，并覆盖表格批跑、坏视频隔离、远程下载和跳转、过程查看与重启恢复；不调用模型和外网。
