"""
低空监视 Web 态势平台 (Flask)

启动:
    python web/app.py --data ../shuju --output ../output
    然后浏览器访问 http://127.0.0.1:5000
"""

import os
import sys
import json
import argparse
from datetime import datetime

from flask import Flask, render_template, jsonify

# 将上级目录加入路径以导入 core/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from utils.coords import enu_to_wgs84

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 使用絕對路徑，避免 Flask debug reloader 改變工作目錄
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
DATA_DIR = os.path.join(_BASE, "shuju")
OUTPUT_DIR = os.path.join(_BASE, "output")

# ENU 參考原點
REF_LAT, REF_LON, REF_ALT = 30.451439, 114.009479, 0.0


def _load_json_safe(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    """系统运行摘要"""
    report = _load_json_safe(os.path.join(OUTPUT_DIR, "flow_report.json"))
    assoc = _load_csv_safe(os.path.join(OUTPUT_DIR, "association_results.csv"))
    anomaly = _load_csv_safe(os.path.join(OUTPUT_DIR, "anomaly_events.csv"))
    conflict = _load_csv_safe(os.path.join(OUTPUT_DIR, "conflict_events.csv"))

    return jsonify({
        "peak_active_targets": report.get("peak_active_targets", 0),
        "total_time_span_sec": report.get("total_time_span_sec", 0),
        "corridor_count": len(report.get("corridors", [])),
        "association_count": len(assoc),
        "confirmed_count": int((assoc["level"] == "confirmed").sum()) if not assoc.empty else 0,
        "anomaly_count": len(anomaly),
        "conflict_count": len(conflict),
    })


@app.route("/api/tracks")
def api_tracks():
    """获取原始与融合航迹（采样后用于前端展示）"""
    tracks = []
    # 原始数据采样
    for src, fname in [("adsb", "ad_adsb.xlsx"), ("remote_id", "ad_remote_id.xlsx"), ("radar", "ad_radar.xlsx")]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path)
        lat_col = next((c for c in ["lat", "latitude"] if c in df.columns), None)
        lon_col = next((c for c in ["lon", "lng", "longitude"] if c in df.columns), None)
        if not lat_col or not lon_col:
            continue
        tid_col = "icao" if src == "adsb" else ("uas_id" if src == "remote_id" else "track_number")
        if tid_col not in df.columns:
            continue
        tids = df[tid_col].dropna().unique()[:30]  # 限制数量
        for tid in tids:
            sub = df[df[tid_col] == tid].dropna(subset=[lat_col, lon_col]).head(80)
            coords = [[float(r[lat_col]), float(r[lon_col])] for _, r in sub.iterrows()]
            if len(coords) >= 2:
                tracks.append({
                    "id": f"{src}:{tid}",
                    "source": src,
                    "type": "original",
                    "coords": coords,
                })

    # 融合航迹（读取 fused_tracks_all.csv）
    fused_path = os.path.join(OUTPUT_DIR, "fused_tracks_all.csv")
    if os.path.exists(fused_path):
        fdf = pd.read_csv(fused_path)
        if "e" in fdf.columns and "n" in fdf.columns:
            lats, lons, _ = enu_to_wgs84(
                fdf["e"].values, fdf["n"].values, fdf.get("u", 0).values,
                REF_LAT, REF_LON, REF_ALT
            )
            fdf["lat"] = lats
            fdf["lon"] = lons
        stids = fdf["system_track_id"].unique()[:20]
        for stid in stids:
            sub = fdf[fdf["system_track_id"] == stid].dropna(subset=["lat", "lon"]).head(80)
            coords = [[float(r["lat"]), float(r["lon"])] for _, r in sub.iterrows()]
            if len(coords) >= 2:
                tracks.append({
                    "id": stid,
                    "source": "fused",
                    "type": "fused",
                    "coords": coords,
                })

    return jsonify(tracks)


@app.route("/api/anomalies")
def api_anomalies():
    """异常事件列表"""
    df = _load_csv_safe(os.path.join(OUTPUT_DIR, "anomaly_events.csv"))
    if df.empty:
        return jsonify([])
    # 取前200条，避免前端过载
    df = df.sort_values("time_sec").head(200)
    return jsonify(df.fillna("").to_dict(orient="records"))


@app.route("/api/conflicts")
def api_conflicts():
    """冲突预警列表"""
    df = _load_csv_safe(os.path.join(OUTPUT_DIR, "conflict_events.csv"))
    if df.empty:
        return jsonify([])
    df = df.sort_values("time_sec").head(200)
    return jsonify(df.fillna("").to_dict(orient="records"))


@app.route("/api/flow")
def api_flow():
    """时域流量数据"""
    df = _load_csv_safe(os.path.join(OUTPUT_DIR, "temporal_flow.csv"))
    if df.empty:
        return jsonify({"times": [], "active": [], "reports": []})
    return jsonify({
        "times": df["time_sec"].tolist(),
        "active": df["active_targets"].tolist(),
        "reports": df["total_reports"].tolist(),
    })


def main():
    global DATA_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(_BASE, "shuju"))
    parser.add_argument("--output", default=os.path.join(_BASE, "output"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    DATA_DIR = os.path.abspath(args.data)
    OUTPUT_DIR = os.path.abspath(args.output)
    print(f"展動 Web 平台: http://{args.host}:{args.port}")
    print(f"資料目錄: {DATA_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
