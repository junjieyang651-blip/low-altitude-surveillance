"""
导出静态数据 JSON 供 GitHub Pages 使用
"""
import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.coords import enu_to_wgs84

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs", "data")
os.makedirs(DOCS_DIR, exist_ok=True)

REF_LAT, REF_LON, REF_ALT = 30.451439, 114.009479, 0.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "shuju")


def export_summary():
    report = {}
    rp = os.path.join(OUTPUT_DIR, "flow_report.json")
    if os.path.exists(rp):
        with open(rp, "r", encoding="utf-8") as f:
            report = json.load(f)

    anomaly = pd.read_csv(os.path.join(OUTPUT_DIR, "anomaly_events.csv")) if os.path.exists(
        os.path.join(OUTPUT_DIR, "anomaly_events.csv")) else pd.DataFrame()
    conflict = pd.read_csv(os.path.join(OUTPUT_DIR, "conflict_events.csv")) if os.path.exists(
        os.path.join(OUTPUT_DIR, "conflict_events.csv")) else pd.DataFrame()
    assoc = pd.read_csv(os.path.join(OUTPUT_DIR, "association_results.csv")) if os.path.exists(
        os.path.join(OUTPUT_DIR, "association_results.csv")) else pd.DataFrame()

    summary = {
        "peak_active_targets": report.get("peak_active_targets", 0),
        "total_time_span_sec": report.get("total_time_span_sec", 0),
        "corridor_count": len(report.get("corridors", [])),
        "association_count": len(assoc),
        "anomaly_count": len(anomaly),
        "conflict_count": len(conflict),
    }
    with open(os.path.join(DOCS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f)
    print(f"  summary.json: {summary}")


def export_tracks():
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
        tids = df[tid_col].dropna().unique()[:30]
        for tid in tids:
            sub = df[df[tid_col] == tid].dropna(subset=[lat_col, lon_col]).head(80)
            coords = [[round(float(r[lat_col]), 6), round(float(r[lon_col]), 6)] for _, r in sub.iterrows()]
            if len(coords) >= 2:
                tracks.append({"id": f"{src}:{tid}", "source": src, "type": "original", "coords": coords})

    # 融合航迹
    fused_path = os.path.join(OUTPUT_DIR, "fused_tracks_all.csv")
    if os.path.exists(fused_path):
        fdf = pd.read_csv(fused_path)
        if "e" in fdf.columns and "n" in fdf.columns:
            lats, lons, _ = enu_to_wgs84(fdf["e"].values, fdf["n"].values,
                                          fdf["u"].values if "u" in fdf.columns else np.zeros(len(fdf)),
                                          REF_LAT, REF_LON, REF_ALT)
            fdf["lat"] = lats
            fdf["lon"] = lons
        stids = fdf["system_track_id"].unique()[:20]
        for stid in stids:
            sub = fdf[fdf["system_track_id"] == stid].dropna(subset=["lat", "lon"]).head(80)
            coords = [[round(float(r["lat"]), 6), round(float(r["lon"]), 6)] for _, r in sub.iterrows()]
            if len(coords) >= 2:
                tracks.append({"id": stid, "source": "fused", "type": "fused", "coords": coords})

    with open(os.path.join(DOCS_DIR, "tracks.json"), "w") as f:
        json.dump(tracks, f)
    print(f"  tracks.json: {len(tracks)} tracks")


def export_anomalies():
    path = os.path.join(OUTPUT_DIR, "anomaly_events.csv")
    if not os.path.exists(path):
        with open(os.path.join(DOCS_DIR, "anomalies.json"), "w") as f:
            json.dump([], f)
        return
    df = pd.read_csv(path).sort_values("time_sec").head(200)
    data = df.fillna("").to_dict(orient="records")
    with open(os.path.join(DOCS_DIR, "anomalies.json"), "w") as f:
        json.dump(data, f)
    print(f"  anomalies.json: {len(data)} records")


def export_conflicts():
    path = os.path.join(OUTPUT_DIR, "conflict_events.csv")
    if not os.path.exists(path):
        with open(os.path.join(DOCS_DIR, "conflicts.json"), "w") as f:
            json.dump([], f)
        return
    df = pd.read_csv(path).sort_values("time_sec").head(200)
    data = df.fillna("").to_dict(orient="records")
    with open(os.path.join(DOCS_DIR, "conflicts.json"), "w") as f:
        json.dump(data, f)
    print(f"  conflicts.json: {len(data)} records")


def export_flow():
    path = os.path.join(OUTPUT_DIR, "temporal_flow.csv")
    if not os.path.exists(path):
        with open(os.path.join(DOCS_DIR, "flow.json"), "w") as f:
            json.dump({"times": [], "active": [], "reports": []}, f)
        return
    df = pd.read_csv(path)
    data = {
        "times": df["time_sec"].tolist(),
        "active": df["active_targets"].tolist(),
        "reports": df["total_reports"].tolist(),
    }
    with open(os.path.join(DOCS_DIR, "flow.json"), "w") as f:
        json.dump(data, f)
    print(f"  flow.json: {len(data['times'])} points")


def export_network():
    path = os.path.join(OUTPUT_DIR, "network_analysis.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(os.path.join(DOCS_DIR, "network.json"), "w") as f:
            json.dump(data, f)
        print(f"  network.json: exported")


def export_rmse():
    path = os.path.join(OUTPUT_DIR, "rmse_metrics.json")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(DOCS_DIR, "rmse.json"), "w") as f:
        json.dump(data, f)
    print(f"  rmse.json: fused={data.get('fused_prediction_rmse_m',0):.1f}m")


def export_spectrum_pog():
    path = os.path.join(OUTPUT_DIR, "spectrum_pog_report.json")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(DOCS_DIR, "spectrum_pog.json"), "w") as f:
        json.dump(data, f)
    print(f"  spectrum_pog.json: detections={data.get('n_detections',0)}")


def export_track_maintenance():
    """导出航迹维持结果"""
    path = os.path.join(OUTPUT_DIR, "track_maintenance.csv")
    if not os.path.exists(path):
        # 从 flow_report 中提取
        rp = os.path.join(OUTPUT_DIR, "flow_report.json")
        if os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as f:
                report = json.load(f)
            maint = report.get("track_maintenance", {})
            with open(os.path.join(DOCS_DIR, "track_maintenance.json"), "w") as f:
                json.dump(maint, f)
            print(f"  track_maintenance.json: from flow_report")
        return
    df = pd.read_csv(path)
    if "status" in df.columns:
        confirmed = int((df["status"] == "confirmed").sum())
        suspected = int((df["status"] == "suspected").sum())
    elif "confirmed" in df.columns:
        confirmed = int(df["confirmed"].sum())
        suspected = len(df) - confirmed
    else:
        confirmed = 0
        suspected = len(df)
    data = {
        "total_broken_pairs": len(df),
        "confirmed": confirmed,
        "suspected": suspected,
        "sample": df.head(20).fillna("").to_dict(orient="records")
    }
    with open(os.path.join(DOCS_DIR, "track_maintenance.json"), "w") as f:
        json.dump(data, f)
    print(f"  track_maintenance.json: {data['total_broken_pairs']} pairs")


if __name__ == "__main__":
    print("导出静态数据到 docs/data/ ...")
    export_summary()
    export_tracks()
    export_anomalies()
    export_conflicts()
    export_flow()
    export_network()
    export_rmse()
    export_spectrum_pog()
    export_track_maintenance()
    print("完成！")
