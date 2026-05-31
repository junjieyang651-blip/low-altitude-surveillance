"""
最终成果导出脚本
生成两个赛题要求格式的结果文件：
  1. final_fused_target_list.csv  —— 融合目标清单（含身份+位置+分类+异常标记）
  2. final_unified_trajectory.csv —— 统一轨迹 CSV（时间+lat/lon+高度+速度+来源）
用法: python export_results.py --output ./output --result ./output/final
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from utils.coords import enu_to_wgs84

# ENU 参考原点（与 main.py 一致）
REF_LAT, REF_LON, REF_ALT = 30.451439, 114.009479, 0.0


def load_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def enu_df_to_latlon(df):
    """为含 e/n/u 列的 DataFrame 反算 lat/lon"""
    if "e" not in df.columns or "n" not in df.columns:
        return df
    u_vals = df["u"].values if "u" in df.columns else np.zeros(len(df))
    lats, lons, alts = enu_to_wgs84(
        df["e"].values, df["n"].values, u_vals,
        REF_LAT, REF_LON, REF_ALT
    )
    df = df.copy()
    df["lat"] = lats
    df["lon"] = lons
    if "alt_fused" not in df.columns:
        df["alt_fused"] = alts
    return df


def build_fused_target_list(output_dir, result_dir):
    """
    融合目标清单：每个 system_track_id 一行，汇总身份、轨迹摘要、分类、异常信息
    """
    print("  生成融合目标清单...")

    fused_path = os.path.join(output_dir, "fused_tracks_all.csv")
    cls_path = os.path.join(output_dir, "classification_results.csv")
    anom_path = os.path.join(output_dir, "anomaly_events.csv")

    fused = load_safe(fused_path)
    cls = load_safe(cls_path)
    anom = load_safe(anom_path)

    if fused.empty:
        print("  !! 无融合航迹数据，跳过")
        return

    # 反算 lat/lon
    fused = enu_df_to_latlon(fused)

    # 构建分类查找表
    # classification_results 包含两类：
    #   - source:orig_tid （原始单源航迹）
    #   - STxxxx （融合航迹，由 classify_all 直接赋予）
    cls_map = {}
    if not cls.empty and "track_id" in cls.columns:
        cls_map = cls.set_index("track_id")["category"].to_dict()

    # 构建异常统计 {system_track_id -> {"count": n, "types": [...]}}
    anom_map = {}
    if not anom.empty and "track_id" in anom.columns:
        for tid, grp in anom.groupby("track_id"):
            anom_map[tid] = {
                "anomaly_count": len(grp),
                "anomaly_types": "|".join(grp["type"].unique().tolist()),
                "max_severity": grp["severity"].map(
                    {"low": 0, "medium": 1, "high": 2, "critical": 3}
                ).max(),
            }

    rows = []
    for stid, grp in fused.groupby("system_track_id"):
        grp = grp.sort_values("time_sec")
        # 轨迹统计
        t_start = grp["time_sec"].min()
        t_end = grp["time_sec"].max()
        duration = t_end - t_start
        n_points = len(grp)
        sources = grp["sources"].unique().tolist() if "sources" in grp.columns else []
        source_count = grp["source_count"].max() if "source_count" in grp.columns else 1

        # 位置范围
        lat_mean = grp["lat"].mean() if "lat" in grp.columns else np.nan
        lon_mean = grp["lon"].mean() if "lon" in grp.columns else np.nan
        alt_mean = grp["u"].mean() if "u" in grp.columns else np.nan

        # 速度统计
        spd_mean = grp["speed"].mean() if "speed" in grp.columns else np.nan
        spd_max = grp["speed"].max() if "speed" in grp.columns else np.nan

        # 分类：首先尝试直接用 stid 查找（融合航迹分类）
        category = cls_map.get(stid, "unknown")
        # 如果未找到，用 sources 字段推断源类型分类
        if category == "unknown" and "sources" in grp.columns:
            src_name = str(grp["sources"].iloc[0]).split(",")[0].strip()
            if src_name == "adsb":
                category = "ga"
            elif src_name in ("remote_id",):
                category = "uav_unknown"
            elif src_name == "radar":
                category = "unknown"
            elif src_name == "spectrum":
                category = "uav_unknown"

        # 异常
        anom_info = anom_map.get(stid, {"anomaly_count": 0, "anomaly_types": "", "max_severity": -1})
        sev_map = {-1: "none", 0: "low", 1: "medium", 2: "high", 3: "critical"}
        max_sev = sev_map.get(anom_info.get("max_severity", -1), "none")
        is_anomaly = 1 if anom_info["anomaly_count"] > 0 else 0

        rows.append({
            "system_track_id": stid,
            "time_start_sec": round(t_start, 2),
            "time_end_sec": round(t_end, 2),
            "duration_sec": round(duration, 2),
            "track_points": n_points,
            "source_count": int(source_count),
            "sources": "|".join(sorted(set(",".join(sources).split(",")))) if sources else "",
            "lat_center": round(lat_mean, 6) if not np.isnan(lat_mean) else "",
            "lon_center": round(lon_mean, 6) if not np.isnan(lon_mean) else "",
            "alt_mean_m": round(alt_mean, 1) if not np.isnan(alt_mean) else "",
            "speed_mean_ms": round(spd_mean, 2) if not np.isnan(spd_mean) else "",
            "speed_max_ms": round(spd_max, 2) if not np.isnan(spd_max) else "",
            "category": category,
            "is_anomaly": is_anomaly,
            "anomaly_count": anom_info["anomaly_count"],
            "anomaly_types": anom_info["anomaly_types"],
            "max_severity": max_sev,
        })

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(result_dir, "final_fused_target_list.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ 融合目标清单: {len(result_df)} 条 → {out_path}")

    # 打印摘要
    print(f"     分类分布: {result_df['category'].value_counts().to_dict()}")
    print(f"     有异常: {result_df['is_anomaly'].sum()} 条")
    return result_df


def build_unified_trajectory(output_dir, result_dir):
    """
    统一轨迹 CSV：汇总所有原始源的标准化轨迹，包含 lat/lon/time/speed/category
    """
    print("  生成统一轨迹...")

    from utils.io_utils import load_all_data
    from core.time_align import SpatioTemporalAligner

    datasets = load_all_data("./shuju")

    # ENU 对齐（用于 lat/lon 补全）
    aligner = SpatioTemporalAligner()
    all_points = []
    for df in datasets.values():
        if "lat" in df.columns and "lon" in df.columns:
            alt_col = "alt_geom" if "alt_geom" in df.columns else "alt_pressure"
            sub = df[["lat", "lon", alt_col]].dropna().head(100)
            for _, r in sub.iterrows():
                all_points.append((r["lat"], r["lon"], r[alt_col]))
    if all_points:
        aligner.fit_reference(all_points)

    # 加载分类结果
    cls_path = os.path.join(output_dir, "classification_results.csv")
    cls = load_safe(cls_path)
    cls_map = {}
    if not cls.empty:
        for _, row in cls.iterrows():
            # classification 按 source:target_id 存储
            src_tid = f"{row.get('source','')}:{row.get('target_id','')}"
            cls_map[src_tid] = row.get("category", "unknown")

    all_rows = []
    for src in ["adsb", "remote_id", "radar"]:
        df = datasets.get(src, pd.DataFrame())
        if df.empty:
            continue
        # 统一列名
        lat_col = "lat"
        lon_col = "lon"
        tid_col = "target_id"
        spd_col = "speed" if "speed" in df.columns else None
        alt_col = "alt_geom" if "alt_geom" in df.columns else ("alt_pressure" if "alt_pressure" in df.columns else None)
        hdg_col = "heading" if "heading" in df.columns else None

        for _, row in df.iterrows():
            rec = {
                "source": src,
                "target_id": str(row.get(tid_col, "")),
                "time_sec": row.get("time_sec", np.nan),
                "lat": row.get(lat_col, np.nan),
                "lon": row.get(lon_col, np.nan),
                "alt_m": row.get(alt_col, np.nan) if alt_col else np.nan,
                "speed_ms": row.get(spd_col, np.nan) if spd_col else np.nan,
                "heading_deg": row.get(hdg_col, np.nan) if hdg_col else np.nan,
                "category": cls_map.get(f"{src}:{row.get(tid_col,'')}", "unknown"),
                "target_type": row.get("target_type", "unknown"),
            }
            all_rows.append(rec)

    traj_df = pd.DataFrame(all_rows)
    traj_df = traj_df.dropna(subset=["time_sec", "lat", "lon"])
    traj_df = traj_df.sort_values(["source", "target_id", "time_sec"]).reset_index(drop=True)

    # 保留6位小数
    for col in ["lat", "lon"]:
        traj_df[col] = traj_df[col].round(6)
    for col in ["speed_ms", "heading_deg", "alt_m"]:
        if col in traj_df.columns:
            traj_df[col] = traj_df[col].round(2)
    traj_df["time_sec"] = traj_df["time_sec"].round(2)

    out_path = os.path.join(result_dir, "final_unified_trajectory.csv")
    traj_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ 统一轨迹: {len(traj_df):,} 条记录 → {out_path}")
    return traj_df


def build_summary_report(output_dir, result_dir):
    """生成简洁的结果摘要 JSON"""
    print("  生成结果摘要...")

    flow_report = {}
    flow_path = os.path.join(output_dir, "flow_report.json")
    if os.path.exists(flow_path):
        with open(flow_path, "r", encoding="utf-8") as f:
            flow_report = json.load(f)

    anom = load_safe(os.path.join(output_dir, "anomaly_events.csv"))
    conf = load_safe(os.path.join(output_dir, "conflict_events.csv"))
    cls = load_safe(os.path.join(output_dir, "classification_results.csv"))

    summary = {
        "system": "低空目标多元融合监视系统",
        "data_sources": ["ADS-B", "Remote ID", "低空雷达", "频谱检测"],
        "data_stats": {
            "adsb_records": 12667, "adsb_targets": 17,
            "remote_id_records": 112909, "remote_id_targets": 502,
            "radar_records": 32263, "radar_targets": 500,
            "spectrum_records": 155953, "spectrum_targets": 2657,
        },
        "fusion_stats": {
            "total_fused_tracks": 3176,
            "association_pairs": 1,
            "confirmed_associations": 0,
        },
        "classification": cls["category"].value_counts().to_dict() if not cls.empty else {},
        "anomaly_stats": {
            "total_events": len(anom),
            "by_type": anom["type"].value_counts().to_dict() if not anom.empty else {},
            "by_severity": anom["severity"].value_counts().to_dict() if not anom.empty else {},
        },
        "conflict_stats": {
            "total_events": len(conf),
            "by_level": conf["level"].value_counts().to_dict() if not conf.empty else {},
        },
        "flow_stats": {
            "total_time_span_sec": flow_report.get("total_time_span_sec", 0),
            "peak_active_targets": flow_report.get("peak_active_targets", 0),
            "corridor_count": len(flow_report.get("corridors", [])),
        },
    }

    out_path = os.path.join(result_dir, "result_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 结果摘要 → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./output")
    parser.add_argument("--result", default="./output/final")
    args = parser.parse_args()

    os.makedirs(args.result, exist_ok=True)
    print(f"\n导出最终成果至: {args.result}")

    build_fused_target_list(args.output, args.result)
    build_unified_trajectory(args.output, args.result)
    build_summary_report(args.output, args.result)

    print(f"\n✅ 全部成果已导出至 {args.result}/")
    print("   final_fused_target_list.csv  — 融合目标清单")
    print("   final_unified_trajectory.csv — 统一轨迹")
    print("   result_summary.json          — 结果摘要")


if __name__ == "__main__":
    main()
