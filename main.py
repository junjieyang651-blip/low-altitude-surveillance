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
from core.association import TrackAssociator
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
    print("\n[Step 3/7] 多源异构航迹关联...")
    associator = TrackAssociator()
    assoc_df = associator.associate_tracks_global(datasets, time_step=2.0)
    if not assoc_df.empty:
        print(f"  关联对总数: {len(assoc_df)}")
        print(f"  confirmed: {(assoc_df['level'] == 'confirmed').sum()}")
        print(f"  suspected: {(assoc_df['level'] == 'suspected').sum()}")
        assoc_df.to_csv(os.path.join(output_dir, "association_results.csv"), index=False)
    else:
        print("  未产生关联结果（可能数据时间覆盖不足）")

    # ---------------------------
    # Step 4: 航迹融合
    # ---------------------------
    print("\n[Step 4/7] IMM-AEKF动态航迹融合...")
    fusion_engine = IMMFusionEngine()
    fused_tracks = fusion_engine.build_fused_tracks(assoc_df, datasets, min_confidence="suspected")
    print(f"  生成融合航迹数: {len(fused_tracks)}")
    print(f"  融合引擎: IMM (CV/CT/CA三模型) + AEKF自适应滤波")
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
    print(f"    航线走廊数: {len(flow_report['corridors'])}")
    # 将 numpy 类型转为原生 Python 类型以便 JSON 序列化
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
        pog_report = pog_processor.generate_report(spectrum_data)
        print(f"  传感器数: {pog_report['n_sensors']}")
        print(f"  总检测数: {pog_report['n_detections']}")
        print(f"  多站交叉定位事件: {pog_report['n_multistation_events']}")
        print(f"  高概率网格数: {pog_report['high_probability_cells']}")
        print(f"  空间覆盖率: {pog_report['coverage_ratio']:.4f}")
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
    print(f"  网络节点数: {stats['n_nodes']}")
    print(f"  网络边数: {stats['n_edges']}")
    print(f"  平均度: {stats['average_degree']}")
    print(f"  聚类系数: {stats['average_clustering_coefficient']}")
    print(f"  社区数: {stats.get('n_communities', 0)}")
    print(f"  核心风险节点 (Top 5):")
    for cn in net_report["critical_risk_nodes"][:5]:
        print(f"    {cn['node_id']}: 综合分={cn['composite_score']:.4f}, 介数={cn['betweenness_centrality']:.4f}")
    if net_report.get("vulnerability_analysis"):
        print(f"  脆弱节点 (Top 3):")
        for vn in net_report["vulnerability_analysis"][:3]:
            print(f"    {vn['node_id']}: 脆弱分={vn['vulnerability_score']:.4f}, 流量={vn['traffic_load']}")
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
