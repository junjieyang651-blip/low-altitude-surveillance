"""
低空流量分析引擎

- 时域流量曲线：统计单位时间内的活跃目标数与总架次
- 空域热力栅格：将空域划分为网格，统计各网格点迹密度
- 航线聚类：基于网格密度的简化聚类提取主要航线走廊
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class FlowAnalyzer:
    def __init__(self, grid_size_m: float = 500.0):
        """
        grid_size_m: 空域栅格边长（米）
        """
        self.grid_size = grid_size_m

    def temporal_flow(self, datasets: Dict[str, pd.DataFrame],
                      time_window_sec: float = 60.0) -> pd.DataFrame:
        """
        时域流量统计
        返回每 time_window_sec 的:
        - active_targets: 该时段出现的目标种类数（按 target_id 去重）
        - total_reports: 总报告点数
        - avg_altitude: 平均高度
        """
        all_records = []
        for src, df in datasets.items():
            if df.empty or "time_sec" not in df.columns:
                continue
            sub = df[["time_sec", "target_id", "alt_geom" if "alt_geom" in df.columns else "u"]].copy()
            sub["source"] = src
            all_records.append(sub)
        if not all_records:
            return pd.DataFrame()

        master = pd.concat(all_records, ignore_index=True)
        master["time_bin"] = (master["time_sec"] / time_window_sec).astype(int) * time_window_sec

        stats = []
        for t, grp in master.groupby("time_bin"):
            stats.append({
                "time_sec": t,
                "active_targets": grp["target_id"].nunique(),
                "total_reports": len(grp),
                "avg_altitude": grp.iloc[:, 2].mean() if not grp.iloc[:, 2].isna().all() else np.nan,
            })
        return pd.DataFrame(stats).sort_values("time_sec").reset_index(drop=True)

    def spatial_heatmap(self, datasets: Dict[str, pd.DataFrame],
                        ref_e: float = 0.0, ref_n: float = 0.0) -> pd.DataFrame:
        """
        空域热力栅格统计（基于 ENU 坐标）
        返回每个栅格的密度计数与平均高度
        """
        all_points = []
        for src, df in datasets.items():
            if df.empty or "e" not in df.columns or "n" not in df.columns:
                continue
            sub = df[["e", "n", "u" if "u" in df.columns else "alt_geom"]].copy()
            sub.columns = ["e", "n", "u"]
            all_points.append(sub)
        if not all_points:
            return pd.DataFrame()

        master = pd.concat(all_points, ignore_index=True)
        master = master.dropna(subset=["e", "n"])

        # 栅格索引
        master["grid_x"] = np.floor((master["e"] - ref_e) / self.grid_size).astype(int)
        master["grid_y"] = np.floor((master["n"] - ref_n) / self.grid_size).astype(int)

        heat = []
        for (gx, gy), grp in master.groupby(["grid_x", "grid_y"]):
            heat.append({
                "grid_x": gx,
                "grid_y": gy,
                "center_e": (gx + 0.5) * self.grid_size + ref_e,
                "center_n": (gy + 0.5) * self.grid_size + ref_n,
                "count": len(grp),
                "avg_u": grp["u"].mean(),
            })
        return pd.DataFrame(heat).sort_values("count", ascending=False).reset_index(drop=True)

    def corridor_extraction(self, datasets: Dict[str, pd.DataFrame],
                            density_threshold: int = 5) -> List[Dict]:
        """
        基于栅格密度的航线走廊提取（简化 DBSCAN 思想）
        将高密度栅格视为航线走廊节点，连通邻接栅格形成走廊
        """
        heat = self.spatial_heatmap(datasets)
        if heat.empty:
            return []

        dense = heat[heat["count"] >= density_threshold].copy()
        if dense.empty:
            return []

        # 标记连通区域（4-邻接）
        dense_set = set(zip(dense["grid_x"].values, dense["grid_y"].values))
        visited = set()
        corridors = []

        def neighbors(gx, gy):
            return [(gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)]

        for start in dense_set:
            if start in visited:
                continue
            # BFS
            queue = [start]
            cluster = []
            visited.add(start)
            while queue:
                cur = queue.pop(0)
                cluster.append(cur)
                for nb in neighbors(*cur):
                    if nb in dense_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

            if len(cluster) >= 3:
                # 计算走廊几何中心与主方向
                cells = dense[dense[["grid_x", "grid_y"]].apply(tuple, axis=1).isin(cluster)]
                mean_e = cells["center_e"].mean()
                mean_n = cells["center_n"].mean()
                # 简单主方向：用PCA太复杂，用首尾点方向近似
                sorted_by_e = cells.sort_values("center_e")
                p_start = (sorted_by_e.iloc[0]["center_e"], sorted_by_e.iloc[0]["center_n"])
                p_end = (sorted_by_e.iloc[-1]["center_e"], sorted_by_e.iloc[-1]["center_n"])
                de = p_end[0] - p_start[0]
                dn = p_end[1] - p_start[1]
                heading = math.degrees(math.atan2(de, dn))
                if heading < 0:
                    heading += 360.0
                corridors.append({
                    "cell_count": len(cluster),
                    "total_points": int(cells["count"].sum()),
                    "center_e": mean_e,
                    "center_n": mean_n,
                    "direction_deg": round(heading, 1),
                    "length_m": round(math.sqrt(de ** 2 + dn ** 2), 1),
                })
        return corridors

    def generate_report(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        生成流量分析综合报告（字典）
        """
        temporal = self.temporal_flow(datasets)
        heat = self.spatial_heatmap(datasets)
        corridors = self.corridor_extraction(datasets)

        report = {
            "total_time_span_sec": temporal["time_sec"].max() - temporal["time_sec"].min() if not temporal.empty else 0,
            "peak_active_targets": temporal["active_targets"].max() if not temporal.empty else 0,
            "peak_time_sec": temporal.loc[temporal["active_targets"].idxmax(), "time_sec"] if not temporal.empty else None,
            "total_reports": int(temporal["total_reports"].sum()) if not temporal.empty else 0,
            "hotspot_count": len(heat[heat["count"] >= heat["count"].quantile(0.95)]) if not heat.empty else 0,
            "corridors": corridors,
        }
        return report
