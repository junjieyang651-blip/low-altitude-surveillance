"""
多源异构航迹关联引擎

创新点：
1. 速度自适应时间窗 —— 高速目标缩小窗口减少错关联，低速目标放宽窗口避免漏关联
2. 多特征加权相似度 —— 空间+速度+航向+高度四维联合度量，比单一最近邻更鲁棒
3. 三级关联确认机制 —— "确定/疑似/待确认"，配合航迹连续性后校验，降低虚警
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

from utils.constants import ASSOCIATION
from utils.coords import velocity_to_enu, heading_from_velocity


class TrackState:
    """
    单条航迹在某一时刻的状态快照，用于关联计算
    """
    def __init__(self, source: str, target_id: str, time_sec: float,
                 e: float, n: float, u: float,
                 speed: float = np.nan, heading: float = np.nan,
                 climb_rate: float = np.nan):
        self.source = source
        self.target_id = target_id
        self.time_sec = time_sec
        self.e = e
        self.n = n
        self.u = u
        self.speed = speed
        self.heading = heading
        self.climb_rate = climb_rate

    @property
    def position(self) -> np.ndarray:
        return np.array([self.e, self.n, self.u])

    def velocity_vector(self) -> np.ndarray:
        """返回 ENU 速度矢量 [ve, vn, vu]"""
        if math.isnan(self.speed) or math.isnan(self.heading):
            return np.array([np.nan, np.nan, np.nan])
        ve, vn, vu = velocity_to_enu(self.speed, self.heading, self.climb_rate if not math.isnan(self.climb_rate) else 0.0)
        return np.array([ve, vn, vu])


class TrackAssociator:
    def __init__(self):
        self.cfg = ASSOCIATION
        # 关联历史: (src_a, tid_a, src_b, tid_b) -> deque of bool (最近 N 帧是否关联)
        self.confirm_history: Dict[Tuple, deque] = defaultdict(lambda: deque(maxlen=self.cfg["confirm_frames"]))

    def _adaptive_time_window(self, speed_ms: float) -> float:
        """
        速度自适应时间窗
        v 越大 → 窗口越小（高速目标位移快，错关联代价大）
        v 越小 → 窗口越大（低速目标位移慢，容易漏关联）
        """
        if math.isnan(speed_ms) or speed_ms <= 0:
            v_eff = 10.0  # 默认假定 10 m/s
        else:
            v_eff = speed_ms
        dt = self.cfg["adaptive_window_factor"] / v_eff
        return max(self.cfg["min_window_sec"], min(self.cfg["max_window_sec"], dt))

    def _compute_similarity(self, s1: TrackState, s2: TrackState) -> Tuple[float, Dict]:
        """
        计算两条航迹状态的相似度得分 (0~1, 越大越相似)
        同时返回各分量详情用于调试与可视化
        """
        # 时间差
        dt = abs(s1.time_sec - s2.time_sec)
        tw = max(self._adaptive_time_window(s1.speed), self._adaptive_time_window(s2.speed))
        if dt > tw:
            return 0.0, {"reason": "time_window_exceeded", "dt": dt, "window": tw}

        # 空间距离 (ENU, m)
        de = s1.e - s2.e
        dn = s1.n - s2.n
        du = s1.u - s2.u
        dist_3d = math.sqrt(de ** 2 + dn ** 2 + du ** 2)
        if dist_3d > self.cfg["max_spatial_distance_m"]:
            return 0.0, {"reason": "spatial_too_far", "dist_3d": dist_3d}

        # 高度差单独惩罚（垂直方向误差通常更大，但物理上高度应更接近）
        d_alt = abs(s1.u - s2.u)

        # 速度差
        v1 = s1.velocity_vector()
        v2 = s2.velocity_vector()
        speed_diff = np.nan
        speed_sim = 0.5  # 默认中性
        if not np.isnan(v1).any() and not np.isnan(v2).any():
            speed_diff = np.linalg.norm(v1 - v2)
            if speed_diff > self.cfg["max_speed_diff_ms"]:
                return 0.0, {"reason": "speed_mismatch", "speed_diff": speed_diff}
            speed_sim = 1.0 - (speed_diff / self.cfg["max_speed_diff_ms"])

        # 航向差（考虑 360° 环绕）
        heading_sim = 0.5
        if not math.isnan(s1.heading) and not math.isnan(s2.heading):
            dh = abs(s1.heading - s2.heading)
            dh = min(dh, 360.0 - dh)
            if dh > self.cfg["max_heading_diff_deg"]:
                return 0.0, {"reason": "heading_mismatch", "dh": dh}
            heading_sim = 1.0 - (dh / self.cfg["max_heading_diff_deg"])

        # 空间相似度（高斯衰减）
        sigma = self.cfg["max_spatial_distance_m"] / 3.0
        spatial_sim = math.exp(-(dist_3d ** 2) / (2 * sigma ** 2))

        # 高度相似度
        alt_sim = 1.0
        if d_alt > self.cfg["max_altitude_diff_m"]:
            alt_sim = 0.0
        else:
            alt_sim = 1.0 - (d_alt / self.cfg["max_altitude_diff_m"])

        # 综合加权相似度（权重可工程调参）
        w_spatial = 0.35
        w_speed = 0.25
        w_heading = 0.20
        w_alt = 0.20

        score = (w_spatial * spatial_sim +
                 w_speed * speed_sim +
                 w_heading * heading_sim +
                 w_alt * alt_sim)

        details = {
            "dt": dt, "window": tw,
            "dist_3d": dist_3d, "d_alt": d_alt,
            "speed_diff": speed_diff,
            "spatial_sim": spatial_sim, "speed_sim": speed_sim,
            "heading_sim": heading_sim, "alt_sim": alt_sim,
            "score": score,
        }
        return score, details

    def associate_frame(self, states_a: List[TrackState],
                        states_b: List[TrackState]) -> List[Tuple[TrackState, TrackState, float, str]]:
        """
        对两个源在相近时刻的状态列表进行单帧关联
        返回: [(state_a, state_b, score, level), ...]
        level 为初步判决（未经连续帧确认）
        """
        # 计算相似度矩阵
        pairs = []
        for sa in states_a:
            for sb in states_b:
                score, details = self._compute_similarity(sa, sb)
                if score > 0.0:
                    pairs.append((sa, sb, score, details))

        if not pairs:
            return []

        # 按分数降序，贪心匹配（一个航迹只关联一次）
        pairs.sort(key=lambda x: x[2], reverse=True)

        matched_a = set()
        matched_b = set()
        results = []

        for sa, sb, score, details in pairs:
            key_a = (sa.source, sa.target_id)
            key_b = (sb.source, sb.target_id)
            if key_a in matched_a or key_b in matched_b:
                continue

            # 初步分级
            if score >= 0.85:
                level = "confirmed"
            elif score >= 0.60:
                level = "suspected"
            else:
                level = "pending"

            results.append((sa, sb, score, level))
            matched_a.add(key_a)
            matched_b.add(key_b)

        return results

    def update_confirmation(self, src_a: str, tid_a: str, src_b: str, tid_b: str, matched: bool) -> str:
        """
        基于连续帧的历史更新关联置信等级
        返回最终等级: confirmed | suspected | pending
        """
        key = (src_a, tid_a, src_b, tid_b)
        hist = self.confirm_history[key]
        hist.append(matched)

        if not matched:
            if len(hist) >= self.cfg["confirm_frames"] and sum(hist) == 0:
                # 连续多帧未匹配，降级
                return "pending"
            return "suspected" if any(hist) else "pending"

        # 当前帧匹配
        if len(hist) >= self.cfg["confirm_frames"] and all(hist):
            return "confirmed"
        elif sum(hist) >= max(1, len(hist) // 2):
            return "suspected"
        return "pending"

    def associate_tracks_global(self, datasets: Dict[str, pd.DataFrame],
                                time_step: float = 1.0) -> pd.DataFrame:
        """
        全局航迹关联流水线（高性能版）

        策略：
        1. 排除 spectrum（无目标位置，仅用于识别验证）
        2. 预计算每条轨迹的时间范围 + 空间包围盒
        3. 时间重叠 + 空间包围盒粗筛 → 大幅减少候选对
        4. 对候选对在重叠时间段内向量化计算相似度
        5. 贪心匹配确保一对一

        输入: {source_name: df_with_enu}
        输出: 关联结果表
        """
        # 只关联有目标位置的源
        spatial_sources = [s for s in datasets.keys() if s != "spectrum"]
        if len(spatial_sources) < 2:
            return pd.DataFrame()

        # ---- 预计算：为每条轨迹建立索引 ----
        track_index = {}  # (src, tid) -> {t_min, t_max, e_c, n_c, df}
        min_track_points = 5  # 最少点数才参与关联（过滤噪声/短轨迹）
        for src in spatial_sources:
            df = datasets[src]
            if "e" not in df.columns or "n" not in df.columns:
                continue
            for tid, grp in df.groupby("target_id"):
                if len(grp) < min_track_points:
                    continue
                t_min = grp["time_sec"].min()
                t_max = grp["time_sec"].max()
                # 轨迹中心点 + 平均高度（用于粗筛）
                e_c = grp["e"].mean()
                n_c = grp["n"].mean()
                u_c = grp["u"].mean() if "u" in grp.columns else 0.0
                track_index[(src, str(tid))] = {
                    "t_min": t_min, "t_max": t_max,
                    "e_c": e_c, "n_c": n_c, "u_c": u_c,
                    "df": grp[["time_sec", "e", "n", "u", "speed", "heading", "climb_rate"]].copy(),
                }

        records = []

        # ---- 两两源之间关联 ----
        for i in range(len(spatial_sources)):
            for j in range(i + 1, len(spatial_sources)):
                sa, sb = spatial_sources[i], spatial_sources[j]
                tracks_a = [(k, v) for k, v in track_index.items() if k[0] == sa]
                tracks_b = [(k, v) for k, v in track_index.items() if k[0] == sb]

                if not tracks_a or not tracks_b:
                    continue

                print(f"  关联 {sa}({len(tracks_a)} 条) ↔ {sb}({len(tracks_b)} 条)...")

                # 粗筛：时间重叠 + 中心点距离阈值
                # 根据源对动态调整：remote_id↔radar 目标多，阈值更紧
                if sa == "remote_id" and sb == "radar":
                    coarse_dist_thresh = 200.0
                else:
                    coarse_dist_thresh = 300.0
                candidates = []
                for (src_a, tid_a), info_a in tracks_a:
                    for (src_b, tid_b), info_b in tracks_b:
                        # 时间是否重叠（允许窗口容差）
                        tw_max = self.cfg["max_window_sec"]
                        if info_a["t_max"] + tw_max < info_b["t_min"] - tw_max:
                            continue
                        if info_b["t_max"] + tw_max < info_a["t_min"] - tw_max:
                            continue
                        # 中心点平面距离粗筛
                        de = info_a["e_c"] - info_b["e_c"]
                        dn = info_a["n_c"] - info_b["n_c"]
                        dist_2d = math.sqrt(de ** 2 + dn ** 2)
                        # NaN 检查：如果 e_c/n_c 是 NaN，直接过滤
                        if math.isnan(dist_2d) or dist_2d > coarse_dist_thresh:
                            continue
                        candidates.append(((src_a, tid_a), (src_b, tid_b), info_a, info_b))

                print(f"    粗筛后候选对: {len(candidates)}")

                # 细匹配：在重叠时间段内计算最优相似度
                pair_scores = []
                for (src_a, tid_a), (src_b, tid_b), info_a, info_b in candidates:
                    df_a = info_a["df"]
                    df_b = info_b["df"]

                    # 找重叠时间范围
                    t_overlap_min = max(df_a["time_sec"].min(), df_b["time_sec"].min())
                    t_overlap_max = min(df_a["time_sec"].max(), df_b["time_sec"].max())
                    if t_overlap_min > t_overlap_max:
                        continue

                    # 以较稀疏的源为基准，在较密的目标中找时间最近的点
                    # 判断哪个源更稀疏（点数少）
                    if len(df_a) <= len(df_b):
                        base_df, match_df = df_a, df_b
                        base_src, match_src = src_a, src_b
                        base_tid, match_tid = tid_a, tid_b
                    else:
                        base_df, match_df = df_b, df_a
                        base_src, match_src = src_b, src_a
                        base_tid, match_tid = tid_b, tid_a

                    best_score = 0.0
                    best_details = None
                    best_t = None

                    # 对 base 的每个代表点（最多取 3 个：起点、中点、终点），在 match 中找时间最近的
                    base_times = base_df["time_sec"].values
                    if len(base_times) > 3:
                        base_times = np.array([base_times[0], base_times[len(base_times)//2], base_times[-1]])

                    for t_base in base_times:
                        # 找 match 中时间最近的点
                        idx_match = (match_df["time_sec"] - t_base).abs().argmin()
                        row_match = match_df.iloc[idx_match]
                        row_base = base_df.iloc[(base_df["time_sec"] - t_base).abs().argmin()]

                        # 快速预筛：时间差 > 60s 或距离 > 500m 跳过
                        dt = abs(float(row_base["time_sec"]) - float(row_match["time_sec"]))
                        if dt > 60.0:
                            continue
                        de = float(row_base["e"]) - float(row_match["e"])
                        dn = float(row_base["n"]) - float(row_match["n"])
                        dist = math.sqrt(de ** 2 + dn ** 2)
                        if dist > 500.0:
                            continue

                        sta = TrackState(
                            source=base_src, target_id=base_tid,
                            time_sec=float(row_base["time_sec"]),
                            e=float(row_base["e"]), n=float(row_base["n"]), u=float(row_base.get("u", 0)),
                            speed=float(row_base["speed"]) if not pd.isna(row_base.get("speed")) else np.nan,
                            heading=float(row_base["heading"]) if not pd.isna(row_base.get("heading")) else np.nan,
                            climb_rate=float(row_base["climb_rate"]) if not pd.isna(row_base.get("climb_rate")) else np.nan,
                        )
                        stb = TrackState(
                            source=match_src, target_id=match_tid,
                            time_sec=float(row_match["time_sec"]),
                            e=float(row_match["e"]), n=float(row_match["n"]), u=float(row_match.get("u", 0)),
                            speed=float(row_match["speed"]) if not pd.isna(row_match.get("speed")) else np.nan,
                            heading=float(row_match["heading"]) if not pd.isna(row_match.get("heading")) else np.nan,
                            climb_rate=float(row_match["climb_rate"]) if not pd.isna(row_match.get("climb_rate")) else np.nan,
                        )
                        score, details = self._compute_similarity(sta, stb)
                        if score > best_score:
                            best_score = score
                            best_details = details
                            best_t = (float(row_base["time_sec"]) + float(row_match["time_sec"])) / 2.0

                    if best_score > 0:
                        level = "confirmed" if best_score >= 0.85 else ("suspected" if best_score >= 0.60 else "pending")
                        pair_scores.append((src_a, tid_a, src_b, tid_b, best_score, level, best_details, best_t))


                # 贪心匹配：按分数降序，一对一
                pair_scores.sort(key=lambda x: x[4], reverse=True)
                matched_a = set()
                matched_b = set()

                for src_a, tid_a, src_b, tid_b, score, level, details, t in pair_scores:
                    if (src_a, tid_a) in matched_a or (src_b, tid_b) in matched_b:
                        continue
                    matched_a.add((src_a, tid_a))
                    matched_b.add((src_b, tid_b))

                    records.append({
                        "time_sec": t,
                        "src_a": src_a,
                        "tid_a": tid_a,
                        "src_b": src_b,
                        "tid_b": tid_b,
                        "score": score,
                        "level": level,
                        "dist_3d_m": details.get("dist_3d"),
                        "d_alt_m": details.get("d_alt"),
                        "speed_diff_ms": details.get("speed_diff"),
                        "heading_diff_deg": details.get("heading_diff_deg"),
                        "time_window_sec": details.get("window"),
                    })

        return pd.DataFrame(records)
