"""
航迹融合引擎

基于关联结果，将多源观测融合为统一、高精度的系统航迹。
采用加权最小二乘融合，权重来源于各传感器的先验定位精度。
融合后航迹可用于后续冲突检测、异常分析等高阶应用。
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

from utils.constants import SOURCE_ACCURACY


class TrackFusionEngine:
    def __init__(self):
        self.accuracy = SOURCE_ACCURACY

    def _source_weight(self, source: str, component: str = "horiz") -> float:
        """
        根据先验精度计算权重：精度越高 → 权重越大
        weight = 1 / sigma^2
        """
        cfg = self.accuracy.get(source, self.accuracy["radar"])
        sigma = cfg.get(component, 50.0)
        return 1.0 / (sigma ** 2)

    def fuse_position(self, observations: List[Tuple[str, float, float, float]]) -> Tuple[float, float, float]:
        """
        融合多个源的位置观测得到最优估计
        observations: [(source, e, n, u), ...]
        返回: (e, n, u)
        """
        if not observations:
            return np.nan, np.nan, np.nan
        if len(observations) == 1:
            return observations[0][1], observations[0][2], observations[0][3]

        weights = []
        vals_e = []
        vals_n = []
        vals_u = []

        for src, e, n, u in observations:
            w = self._source_weight(src, "horiz")
            w_alt = self._source_weight(src, "alt")
            weights.append((w, w_alt))
            vals_e.append(e)
            vals_n.append(n)
            vals_u.append(u)

        weights_arr = np.array(weights)
        we = weights_arr[:, 0]
        wu = weights_arr[:, 1]

        # 加权平均（水平用 horiz 权重，垂直用 alt 权重）
        e_est = np.sum(we * np.array(vals_e)) / np.sum(we)
        n_est = np.sum(we * np.array(vals_n)) / np.sum(we)
        u_est = np.sum(wu * np.array(vals_u)) / np.sum(wu)

        return e_est, n_est, u_est

    def fuse_velocity(self, observations: List[Tuple[str, float, float]]) -> Tuple[float, float]:
        """
        融合速度信息
        observations: [(source, speed, heading), ...]
        将 speed+heading 转为 ENU 分量后加权融合，再转回 speed+heading
        避免直接对航向角做加权平均（360°环绕问题）
        """
        if not observations:
            return np.nan, np.nan

        from utils.coords import velocity_to_enu

        ve_list = []
        vn_list = []
        w_list = []

        for src, speed, heading in observations:
            if math.isnan(speed) or math.isnan(heading):
                continue
            w = self._source_weight(src, "vel")
            ve, vn, _ = velocity_to_enu(speed, heading, 0.0)
            ve_list.append(ve)
            vn_list.append(vn)
            w_list.append(w)

        if not w_list:
            return np.nan, np.nan

        ve_est = np.sum(np.array(w_list) * np.array(ve_list)) / np.sum(w_list)
        vn_est = np.sum(np.array(w_list) * np.array(vn_list)) / np.sum(w_list)

        speed_est = math.sqrt(ve_est ** 2 + vn_est ** 2)
        heading_est = math.degrees(math.atan2(ve_est, vn_est))
        if heading_est < 0:
            heading_est += 360.0

        return speed_est, heading_est

    def build_fused_tracks(self, association_df: pd.DataFrame,
                           datasets: Dict[str, pd.DataFrame],
                           min_confidence: str = "suspected") -> Dict[str, pd.DataFrame]:
        """
        根据关联结果构建融合航迹

        策略：
        1. 将 confirmed/suspected 关联对归并为同一个 system_track_id
        2. 对每个 system_track_id + time_sec，收集所有关联的源观测
        3. 加权融合位置与速度
        4. 未关联的孤立航迹保留为独立 system_track_id

        返回: {system_track_id: fused_df}
        """
        # 空输入保护
        if association_df.empty or "level" not in association_df.columns:
            print("  警告: 无有效关联结果，所有航迹作为孤立航迹输出")
            assoc_valid = pd.DataFrame(columns=["src_a", "tid_a", "src_b", "tid_b", "level"])
        else:
            # 过滤低置信度关联
            conf_map = {"confirmed": 2, "suspected": 1, "pending": 0}
            min_level = conf_map.get(min_confidence, 1)
            assoc_valid = association_df[association_df["level"].map(conf_map) >= min_level].copy()

        # 构建等价类：用并查集将关联对合并为同一 system_track_id
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # 节点标识: (source, target_id)
        for _, row in assoc_valid.iterrows():
            a = (row["src_a"], str(row["tid_a"]))
            b = (row["src_b"], str(row["tid_b"]))
            union(a, b)

        # 所有源中的航迹也要加入 parent，确保孤立航迹也能被找到
        all_tracks = set()
        for src, df in datasets.items():
            for tid in df["target_id"].unique():
                all_tracks.add((src, str(tid)))
        for node in all_tracks:
            find(node)

        # 按根节点分组
        groups: Dict[Tuple, List[Tuple]] = defaultdict(list)
        for node in all_tracks:
            root = find(node)
            groups[root].append(node)

        # 为每个 group 分配 system_track_id
        fused_tracks: Dict[str, pd.DataFrame] = {}
        for idx, (root, members) in enumerate(groups.items(), start=1):
            stid = f"ST{idx:04d}"
            # 收集该组所有观测
            obs_list = []
            for src, tid in members:
                df_src = datasets.get(src)
                if df_src is None:
                    continue
                sub = df_src[df_src["target_id"] == tid].copy()
                if sub.empty:
                    continue
                sub["system_track_id"] = stid
                sub["member_source"] = src
                sub["member_tid"] = tid
                obs_list.append(sub)

            if not obs_list:
                continue

            combined = pd.concat(obs_list, ignore_index=True)

            # 孤立航迹（只有一个源）直接透传，跳过复杂融合
            if len(members) == 1:
                src = members[0][0]
                fused_df = combined.copy()
                fused_df["sources"] = src
                fused_df["source_count"] = 1
                # 保留核心列，统一列名
                keep_cols = ["time_sec", "e", "n", "u", "speed", "heading", "lat", "lon",
                             "alt_geom", "alt_pressure", "system_track_id", "sources", "source_count"]
                keep_cols = [c for c in keep_cols if c in fused_df.columns]
                fused_df = fused_df[keep_cols].sort_values("time_sec").reset_index(drop=True)
                fused_tracks[stid] = fused_df
                continue

            # 多源关联航迹：按时间窗口分箱后融合（避免浮点 time_sec 产生过多小组）
            combined["time_bin"] = combined["time_sec"].round(0)
            fused_rows = []
            for t, grp in combined.groupby("time_bin"):
                pos_obs = []
                vel_obs = []
                for _, r in grp.iterrows():
                    src = r["member_source"]
                    if not pd.isna(r.get("e")) and not pd.isna(r.get("n")):
                        pos_obs.append((src, float(r["e"]), float(r["n"]), float(r.get("u", 0))))
                    if not pd.isna(r.get("speed")) and not pd.isna(r.get("heading")):
                        vel_obs.append((src, float(r["speed"]), float(r["heading"])))

                e_f, n_f, u_f = self.fuse_position(pos_obs)
                sp_f, hd_f = self.fuse_velocity(vel_obs)

                row_dict = {
                    "system_track_id": stid,
                    "time_sec": t,
                    "e": e_f,
                    "n": n_f,
                    "u": u_f,
                    "speed": sp_f,
                    "heading": hd_f,
                    "sources": ",".join(grp["member_source"].unique()),
                    "source_count": grp["member_source"].nunique(),
                }
                fused_rows.append(row_dict)

            if fused_rows:
                fused_df = pd.DataFrame(fused_rows).sort_values("time_sec").reset_index(drop=True)
                fused_tracks[stid] = fused_df

        return fused_tracks
