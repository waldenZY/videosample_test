"""只比较保存结果，不重新抽帧。"""


def compare_runs(left, right, load_trace):
    def keyed(run):
        return {("id", row["item"]["id"]) if row["item"].get("id") else ("row", row["index"]): row
                for row in run["rows"]}

    def side(run, row):
        if row is None:
            return None
        result = {"run_id": run["id"], "index": row["index"], "status": row["status"],
                  "error": row.get("error"), "frame_count": row.get("frame_count"),
                  "stage_counts": row.get("stage_counts", {}), "frames": [], "stages": []}
        try:
            trace = load_trace(run["id"], row["index"])
            by_id = {frame["id"]: frame for frame in trace["frames"]}
            for stage in trace["stages"]:
                frames = [{"id": identity, "time": by_id[identity]["time"],
                           "image_url": f"/api/runs/{run['id']}/items/{row['index']}/frames/{identity}"}
                          for identity in stage["output_ids"]]
                result["stages"].append({"id": stage["id"], "title": stage["title"], "frames": frames})
            final = next((stage for stage in result["stages"] if stage["id"] == "final"), None)
            result["frames"] = final["frames"] if final else []
        except (FileNotFoundError, ValueError, KeyError):
            result["trace_missing"] = True
        return result

    left_rows, right_rows = keyed(left), keyed(right)
    rows = []
    for key in dict.fromkeys([*left_rows, *right_rows]):
        a, b = left_rows.get(key), right_rows.get(key)
        item = (a or b)["item"]
        notices = []
        if a is None or b is None:
            notices.append("只有一次运行包含这道题")
        elif a["status"] != "done" or b["status"] != "done":
            notices.append("至少一侧未成功完成")
        else:
            if not a.get("video_sha256") or not b.get("video_sha256"):
                notices.append("旧结果未记录视频内容，无法确认视频是否相同")
            elif a["video_sha256"] != b["video_sha256"]:
                notices.append("两侧视频内容不同，不能当作同一视频比较")
            if not a.get("effective_config") or not b.get("effective_config"):
                notices.append("旧结果未记录完整配置，无法确认任务时间范围是否相同")
            elif any(a["effective_config"].get(field) != b["effective_config"].get(field)
                     for field in ("task_start_time", "task_end_time")):
                notices.append("两侧任务时间范围不同")
        if key[0] == "row":
            notices.append("样本没有 id，按输入序号对齐，请核对任务")
        rows.append({"id": item.get("id") or f"样本 {key[1] + 1}", "query": item.get("query", ""),
                     "comparable": not notices, "notices": notices,
                     "left": side(left, a), "right": side(right, b)})
    return {"left": {k: left.get(k) for k in ("id", "name", "version", "algorithm")},
            "right": {k: right.get(k) for k in ("id", "name", "version", "algorithm")}, "rows": rows}
