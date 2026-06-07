"""
低空目标多元融合监视系统 — 主运行流水线
一键执行：数据加载 → 时空对齐 → 关联融合 → 分类 → 异常检测 → 冲突检测 → 流量分析 → 输出报表

使用方法:
    python main.py --data ./shuju --output ./output
"""

import os
import sys
# 将脚本所在目录加入路径，确保能正确导入 utils / core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
from datetime import datetime

import pandas as pd
import numpy as np

from utils.io_utils import load_all_data
from utils.coords import compute_centroid
from core.time_align import SpatioTemporalAligner
from core.association import TrackAssociator, TrackMaintenance
from core.fusion import TrackFusionEngine
from core.imm_fusion import IMMFusionEngine
from core.ds_evidence import DSEvidenceEngine
from core.classification import TargetClassifier
from core.anomaly_detection import AnomalyDetector
from core.conflict_detection import ConflictDetector
from core.flow_analysis import FlowAnalyzer
from core.network_topology import AirspaceNetworkGraph


def run_pipeline(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # \u5c06 numpy \u7c7b\u578b\u8f6c\u4e3a\u539f\u751f Python \u7c7b\u578b\u4ee5\u4fbf JSON \u5e8f\u5217\u5316
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return obj

    print("=" * 60)
    print("低空目标多元融合监视系统 — 主流水线")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # ---------------------------
    # Step 1: 加载数据
    # ---------------------------
    print("\n[Step 1/7] 加载异构数据源...")
    datasets = load_all_data(data_dir)
    for name, df in datasets.items():
        print(f"  {name:12s}: {len(df):6d} 条记录, {df['target_id'].nunique():4d} 个目标")

    # ---------------------------
    # Step 2: 时空对齐
    # ---------------------------
    print("\n[Step 2/7] 时空对齐与坐标转换...")
    # 计算全局参考原点
    all_points = []
    for df in datasets.values():
        if "lat" in df.columns and "lon" in df.columns:
            alt_col = "alt_geom" if "alt_geom" in df.columns else "alt_pressure"
            sub = df[["lat", "lon", alt_col]].dropna().head(100)
            for _, r in sub.iterrows():
                all_points.append((r["lat"], r["lon"], r[alt_col]))

    aligner = SpatioTemporalAligner()
    if all_points:
        aligner.fit_reference(all_points)
        print(f"  ENU 参考原点: lat={aligner.ref_lat:.6f}, lon={aligner.ref_lon:.6f}")
    else:
        print("  警告: 无有效参考点，使用默认原点")

    for src in datasets:
        if "lat" in datasets[src].columns and "lon" in datasets[src].columns:
            datasets[src] = aligner.add_enu_coordinates(datasets[src])
            datasets[src] = aligner.outlier_rejection(datasets[src])
            print(f"  {src:12s}: 已添加 ENU 坐标并剔除野值")

    # ---------------------------
    # Step 3: 航迹关联
    # ---------------------------
    print("\n[Step 3/7] \u591a\u6e90\u5f02\u6784\u822a\u8ff9\u5173\u8054...")
    associator = TrackAssociator()
    assoc_df = associator.associate_tracks_global(datasets, time_step=2.0)
    if not assoc_df.empty:
        print(f"  \u5173\u8054\u5bf9\u603b\u6570: {len(assoc_df)}")
        print(f"  confirmed: {(assoc_df['level'] == 'confirmed').sum()}")
        print(f"  suspected: {(assoc_df['level'] == 'suspected').sum()}")
        assoc_df.to_csv(os.path.join(output_dir, "association_results.csv"), index=False)
    else:
        print("  \u672a\u4ea7\u751f\u5173\u8054\u7ed3\u679c\uff08\u53ef\u80fd\u6570\u636e\u65f6\u95f4\u8986\u76d6\u4e0d\u8db3\uff09")
    
    # \u822a\u8ff9\u7ef4\u6301 (Track Maintenance) - \u5904\u7406ID\u8df3\u53d8
    print("\n  [\u822a\u8ff9\u7ef4\u6301] \u68c0\u6d4b\u65ad\u88c2\u822a\u8ff9\u5e76\u91cd\u5173\u8054...")
    maintainer = TrackMaintenance(max_gap_sec=30.0, gate_radius_m=150.0)
    maintenance_df = maintainer.detect_broken_tracks(datasets)
    maint_stats = maintainer.get_maintenance_stats(maintenance_df)
    print(f"    \u68c0\u6d4b\u65ad\u88c2\u822a\u8ff9\u5bf9: {maint_stats['total_broken']}")
    print(f"    \u786e\u8ba4\u91cd\u5173\u8054: {maint_stats['confirmed']}")
    print(f"    \u7591\u4f3c\u91cd\u5173\u8054: {maint_stats['suspected']}")
    if not maintenance_df.empty:
        maintenance_df.to_csv(os.path.join(output_dir, "track_maintenance.csv"), index=False)
        # \u5c06\u786e\u8ba4\u7684\u91cd\u5173\u8054\u7ed3\u679c\u8865\u5145\u5230 assoc_df
        confirmed_maint = maintenance_df[maintenance_df["status"] == "confirmed"]
        if not confirmed_maint.empty:
            extra_assoc = []
            for _, row in confirmed_maint.iterrows():
                extra_assoc.append({
                    "time_sec": 0.0,
                    "src_a": row["source"],
                    "tid_a": row["tid_old"],
                    "src_b": row["source"],
                    "tid_b": row["tid_new"],
                    "score": row["score"],
                    "level": "confirmed",
                    "dist_3d_m": row["pred_error_m"],
                })
            if extra_assoc:
                assoc_df = pd.concat([assoc_df, pd.DataFrame(extra_assoc)], ignore_index=True)
                print(f"    \u8865\u5145\u5173\u8054\u5bf9: {len(extra_assoc)} \u6761")

    # ---------------------------
    # Step 4: 航迹融合
    # ---------------------------
    print("\n[Step 4/7] IMM-AEKF动态航迹融合...")
    fusion_engine = IMMFusionEngine()
    fused_tracks = fusion_engine.build_fused_tracks(assoc_df, datasets, min_confidence="suspected")
    print(f"  \u751f\u6210\u878d\u5408\u822a\u8ff9\u6570: {len(fused_tracks)}")
    print(f"  \u878d\u5408\u5f15\u64ce: IMM (CV/CT/CA\u4e09\u6a21\u578b) + AEKF\u81ea\u9002\u5e94\u6ee4\u6ce2")
    
    # RMSE\u91cf\u5316\u5bf9\u6bd4
    rmse_metrics = fusion_engine.compute_rmse_metrics(fused_tracks, datasets, assoc_df)
    print(f"  RMSE\u7cbe\u5ea6\u5bf9\u6bd4: {rmse_metrics['summary']}")
    for src, m in rmse_metrics.get('per_source_rmse', {}).items():
        imp = rmse_metrics.get('improvement_ratio_pct', {}).get(src, 0)
        print(f"    {src}: RMSE={m['rmse_total_m']:.1f}m (\u878d\u5408\u6539\u8fdb{imp:.0f}%)")
    with open(os.path.join(output_dir, "rmse_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(convert_for_json(rmse_metrics), f, ensure_ascii=False, indent=2)
    # 合并保存所有融合航迹到一个文件，避免数千次磁盘写入
    if fused_tracks:
        all_fused = []
        for stid, fdf in fused_tracks.items():
            fdf_copy = fdf.copy()
            fdf_copy["system_track_id"] = stid
            all_fused.append(fdf_copy)
        pd.concat(all_fused, ignore_index=True).to_csv(
            os.path.join(output_dir, "fused_tracks_all.csv"), index=False)
        print(f"  已保存融合航迹总表: fused_tracks_all.csv")

    # ---------------------------
    # Step 5: 目标分类
    # ---------------------------
    print("\n[Step 5/7] 目标分类...")
    classifier = TargetClassifier()
    class_df = classifier.classify_all(datasets, fused_tracks=fused_tracks)
    if not class_df.empty:
        print(class_df.groupby("category").size().to_string())
        class_df.to_csv(os.path.join(output_dir, "classification_results.csv"), index=False)

    # ---------------------------
    # Step 6: 异常检测
    # ---------------------------
    print("\n[Step 6/7] 异常检测...")
    detector = AnomalyDetector()
    anomaly_df = detector.detect_all(datasets, fused_tracks=fused_tracks)
    if not anomaly_df.empty:
        print(f"  检测到异常事件: {len(anomaly_df)} 条")
        print(anomaly_df["type"].value_counts().head(10).to_string())
        anomaly_df.to_csv(os.path.join(output_dir, "anomaly_events.csv"), index=False)
    else:
        print("  未检测到异常事件")

    # ---------------------------
    # Step 7: 冲突检测
    # ---------------------------
    print("\n[Step 7/7] 冲突检测与流量分析...")
    conflict_det = ConflictDetector()
    if fused_tracks:
        conflict_df = conflict_det.detect_all(
            fused_tracks, time_step=2.0,
            ref_lat=aligner.ref_lat, ref_lon=aligner.ref_lon
        )
        if not conflict_df.empty:
            print(f"  冲突预警事件: {len(conflict_df)} 条")
            print(conflict_df["level"].value_counts().to_string())
            conflict_df.to_csv(os.path.join(output_dir, "conflict_events.csv"), index=False)
        else:
            print("  未检测到冲突")

        # 地理围栏检测
        fence_events = []
        for stid, fdf in fused_tracks.items():
            # 融合航迹只有 e/n/u，围栏检测需要 lat/lon
            # 这里跳过纯融合航迹的围栏检测（因缺少 lat/lon）
            pass
        # 对原始雷达/adsb做围栏检测
        for src in ["adsb", "radar"]:
            if src not in datasets:
                continue
            for tid, grp in datasets[src].groupby("target_id"):
                evs = conflict_det.check_geofence(grp, f"{src}:{tid}")
                fence_events.extend(evs)
        if fence_events:
            pd.DataFrame(fence_events).to_csv(
                os.path.join(output_dir, "geofence_events.csv"), index=False)

    # 流量分析
    flow = FlowAnalyzer(grid_size_m=500.0)
    flow_report = flow.generate_report(datasets)
    print(f"\n  流量分析摘要:")
    print(f"    总时间跨度: {flow_report['total_time_span_sec']:.0f} s")
    print(f"    峰值活跃目标数: {flow_report['peak_active_targets']}")
    print(f"    总报告点数: {flow_report['total_reports']}")
    print(f"    \u822a\u7ebf\u8d70\u5eca\u6570: {len(flow_report['corridors'])}")
    with open(os.path.join(output_dir, "flow_report.json"), "w", encoding="utf-8") as f:
        json.dump(convert_for_json(flow_report), f, ensure_ascii=False, indent=2)
    
    # 保存时域流量CSV
    temporal_df = flow.temporal_flow(datasets)
    if not temporal_df.empty:
        temporal_df.to_csv(os.path.join(output_dir, "temporal_flow.csv"), index=False)

    # ---------------------------
    # Step 8: 频谱概率占据网格 (POG)
    # ---------------------------
    print("\n[Step 8] 频谱概率占据网格 (POG) 分析...")
    from core.spectrum_pog import SpectrumPOGProcessor
    pog_processor = SpectrumPOGProcessor(
        grid_size_m=200.0, extent_m=8000.0,
        ref_lat=aligner.ref_lat, ref_lon=aligner.ref_lon
    )
    spectrum_data = datasets.get("spectrum", pd.DataFrame())
    if not spectrum_data.empty:
        pog_report = pog_processor.generate_report(spectrum_data, fused_tracks=fused_tracks)
        print(f"  \u4f20\u611f\u5668\u6570: {pog_report['n_sensors']}")
        print(f"  \u603b\u68c0\u6d4b\u6570: {pog_report['n_detections']}")
        print(f"  \u591a\u7ad9\u4ea4\u53c9\u5b9a\u4f4d\u4e8b\u4ef6: {pog_report['n_multistation_events']}")
        print(f"  \u9ad8\u6982\u7387\u7f51\u683c\u6570: {pog_report['high_probability_cells']}")
        print(f"  \u7a7a\u95f4\u8986\u76d6\u7387: {pog_report['coverage_ratio']:.4f}")
        # \u865a\u8b66\u5206\u6790\u7ed3\u679c
        fa = pog_report.get('false_alarm_analysis', {})
        if fa:
            print(f"  \u865a\u8b66\u5206\u6790: \u603b\u68c0\u6d4b={fa.get('total_detections',0)}, "
                  f"\u865a\u8b66={fa.get('false_alarms',0)}, "
                  f"\u786e\u8ba4={fa.get('confirmed_detections',0)}, "
                  f"\u865a\u8b66\u7387={fa.get('false_alarm_rate',0):.2%}")
            if fa.get('filter_breakdown'):
                for reason, cnt in fa['filter_breakdown'].items():
                    print(f"    {reason}: {cnt} \u6b21")
        with open(os.path.join(output_dir, "spectrum_pog_report.json"), "w", encoding="utf-8") as f:
            json.dump(convert_for_json(pog_report), f, ensure_ascii=False, indent=2)
    else:
        print("  无频谱数据，跳过POG分析")

    # ---------------------------
    # Step 9: D-S证据理论可信度评估
    # ---------------------------
    print("\n[Step 9] D-S证据理论航迹可信度评估...")
    ds_engine = DSEvidenceEngine(conflict_threshold=0.65)
    credibility_results = []
    # 对多源融合航迹进行真正的冲突评估
    for stid, fdf in list(fused_tracks.items())[:200]:
        if "sources" not in fdf.columns:
            continue
        sources_in_track = fdf["sources"].unique()
        # 构建观测证据（利用实际源种类和数量）
        observations = []
        for s in sources_in_track[:5]:
            # 计算该源观测与融合位置的平均偏差
            src_data = None
            for src_name, src_df in datasets.items():
                if src_name == s:
                    src_data = src_df
                    break
            pos_error = None
            if src_data is not None and "e" in src_data.columns and "e" in fdf.columns:
                fused_mean_e = fdf["e"].mean()
                fused_mean_n = fdf["n"].mean()
                # 取该源中任一target_id的平均位置
                src_mean_e = src_data["e"].mean()
                src_mean_n = src_data["n"].mean()
                pos_error = float(np.sqrt((src_mean_e - fused_mean_e)**2 + (src_mean_n - fused_mean_n)**2))

            observations.append({
                "source": s,
                "detected": True,
                "signal_quality": 0.85 if s in ["adsb", "remote_id"] else 0.65,
                "position_error_m": pos_error,
            })
        if observations:
            result = ds_engine.evaluate_track_credibility(observations)
            result["system_track_id"] = stid
            result["n_sources"] = len(sources_in_track)
            credibility_results.append(result)
    if credibility_results:
        cred_df = pd.DataFrame(credibility_results)
        cred_df.to_csv(os.path.join(output_dir, "track_credibility.csv"), index=False)
        print(f"  评估航迹: {len(cred_df)} 条")
        print(f"  确认真实: {(cred_df['decision'] == 'confirmed_real').sum()}")
        print(f"  疑似虚假: {(cred_df['decision'] == 'suspected_false').sum()}")
        print(f"  高冲突: {(cred_df['decision'] == 'high_conflict').sum()}")
        print(f"  不确定: {(cred_df['decision'] == 'uncertain').sum()}")
        print(f"  融合策略: Dempster={( cred_df['strategy'] == 'dempster').sum()}, Murphy={(cred_df['strategy'] == 'murphy').sum()}")

    # ---------------------------
    # Step 10: 复杂网络拓扑分析
    # ---------------------------
    print("\n[Step 10] 低空复杂网络拓扑分析...")
    net_graph = AirspaceNetworkGraph(grid_size_m=500.0)
    net_report = net_graph.generate_report(fused_tracks)
    stats = net_report["network_statistics"]
    print(f"  \u7f51\u7edc\u8282\u70b9\u6570: {stats['n_nodes']}")
    print(f"  \u7f51\u7edc\u8fb9\u6570: {stats['n_edges']}")
    print(f"  \u5e73\u5747\u5ea6: {stats['average_degree']}")
    print(f"  \u805a\u7c7b\u7cfb\u6570: {stats['average_clustering_coefficient']}")
    print(f"  \u793e\u533a\u6570: {stats.get('n_communities', 0)}")
    print(f"  \u6838\u5fc3\u98ce\u9669\u8282\u70b9 (Top 5):")
    for cn in net_report["critical_risk_nodes"][:5]:
        print(f"    {cn['node_id']}: \u7efc\u5408\u5206={cn['composite_score']:.4f}, \u4ecb\u6570={cn['betweenness_centrality']:.4f}")
    if net_report.get("vulnerability_analysis"):
        print(f"  \u8106\u5f31\u8282\u70b9 (Top 3):")
        for vn in net_report["vulnerability_analysis"][:3]:
            print(f"    {vn['node_id']}: \u8106\u5f31\u5206={vn['vulnerability_score']:.4f}, \u6d41\u91cf={vn['traffic_load']}")
    # \u7a7a\u57df\u5bb9\u91cf\u5206\u6790
    capacity = net_report.get("airspace_capacity", {})
    if capacity:
        print(f"  \u7a7a\u57df\u5bb9\u91cf\u5206\u6790: {capacity.get('capacity_summary', '')}")
        print(f"    \u62e5\u5835\u8282\u70b9\u6570: {capacity.get('n_congested_nodes', 0)}")
        print(f"    \u9971\u548c\u793e\u533a\u6570: {capacity.get('n_saturated_communities', 0)}")
        top_congested = capacity.get('congestion_nodes', [])[:3]
        if top_congested:
            print(f"    Top3\u62e5\u5835\u8282\u70b9:")
            for cn in top_congested:
                print(f"      {cn['node_id']}: CI={cn['congestion_index']:.3f}, \u5bb9\u91cf={cn['dynamic_capacity']:.0f}, \u6d41\u91cf={cn['traffic_count']}")
    with open(os.path.join(output_dir, "network_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(convert_for_json(net_report), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("流水线执行完毕，所有结果已保存至:", output_dir)
    print("=" * 60)
    return {
        "datasets": datasets,
        "fused_tracks": fused_tracks,
        "association": assoc_df,
        "classification": class_df,
        "anomalies": anomaly_df,
        "conflicts": conflict_df if 'conflict_df' in dir() else pd.DataFrame(),
        "flow_report": flow_report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="低空目标多元融合监视系统")
    parser.add_argument("--data", default="./shuju", help="数据目录路径")
    parser.add_argument("--output", default="./output", help="输出目录路径")
    args = parser.parse_args()
    run_pipeline(args.data, args.output)
