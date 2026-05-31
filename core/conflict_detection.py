"""
冲突检测与风险预警引擎

创新点：
1. 速度方向修正人工势场 —— 传统势场只考虑距离，引入相对速度夹角修正势场强度，
   使高速接近目标的威胁感知的更敏锐，同向并行目标威胁感知更合理。
2. DCPA/TCPA 经典航空模型作为基准，与势场法交叉验证。
3. 分级预警 + 解脱建议 —— 不仅报风险，还输出建议航向调整角，体现平台实用价值。
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from utils.constants import CONFLICT, NO_FLY_ZONES
from utils.coords import haversine_distance, wgs84_to_enu


class ConflictDetector:
    def __init__(self):
        self.cfg = CONFLICT

    def _dcpa_tcpa(self, p1: np.ndarray, v1: np.ndarray,
                   p2: np.ndarray, v2: np.ndarray) -> Tuple[float, float, float]:
        """
        计算 DCPA (最近会遇距离) 和 TCPA (到达最近点时间)
        返回: (dcpa, tcpa, relative_speed)
        p, v 为 ENU 坐标/速度 (m, m/s)
        """
        dp = p2 - p1
        dv = v2 - v1
        dv_norm_sq = np.dot(dv, dv)
        rel_speed = math.sqrt(dv_norm_sq)
        current_dist = np.linalg.norm(dp)

        # 相对速度极低（< 1 m/s）：视为准静止，若当前距离已大于警告阈值，
        # 则返回 tcpa=inf 避免触发大量虚假 critical 告警
        if rel_speed < 1.0:
            if current_dist > self.cfg["dcpa_warning_m"]:
                return current_dist, float('inf'), rel_speed
            return current_dist, 0.0, rel_speed

        tcpa = -np.dot(dp, dv) / dv_norm_sq
        if tcpa < 0:
            # 已经在远离，取当前距离为 DCPA，TCPA=0 表示"现在"已最接近
            dcpa = current_dist
            tcpa = 0.0
        else:
            closest_p1 = p1 + v1 * tcpa
            closest_p2 = p2 + v2 * tcpa
            dcpa = np.linalg.norm(closest_p2 - closest_p1)
        return dcpa, tcpa, rel_speed

    def _modified_potential(self, p1: np.ndarray, v1: np.ndarray,
                            p2: np.ndarray, v2: np.ndarray) -> Tuple[float, float]:
        """
        改进型人工势场：
        U = k * exp(-d^2 / 2R^2) * (1 + alpha * cos_theta)
        其中 cos_theta 为相对速度方向与位置连线的夹角余弦：
        - 迎头接近（cos_theta ≈ -1）→ 势能增强
        - 同向远离（cos_theta ≈ 1）→ 势能快速衰减

        返回: (potential_value, normalized_risk_score 0~1)
        """
        d_vec = p2 - p1
        d = np.linalg.norm(d_vec)
        R = self.cfg["potential_radius_m"]
        if d > R * 2:
            return 0.0, 0.0

        dv = v2 - v1
        v_rel_norm = np.linalg.norm(dv)

        cos_theta = 0.0
        if d > 1e-3 and v_rel_norm > 1e-3:
            cos_theta = np.dot(dv, d_vec) / (v_rel_norm * d)
            # cos_theta ∈ [-1, 1]；迎头接近时 dv 指向 p2，d_vec 也指向 p2，
            # 但这里我们需要的是"接近程度"：
            # 若两者相向运动，dv = v2-v1 与 d_vec = p2-p1 方向相同（v2 朝 p1 来，v1 朝 p2 去）
            # 为简化，取 -cos_theta 作为接近因子：
            # cos_theta = -1（反向）→ 接近因子 = 1（增强）
            # cos_theta = 1（同向）→ 接近因子 = -1（减弱）
        approach_factor = 1.0 + self.cfg["potential_k_speed"] * (-cos_theta)
        approach_factor = max(0.1, approach_factor)

        base = math.exp(-(d ** 2) / (2 * (R / 3.0) ** 2))
        potential = base * approach_factor

        # 归一化风险分 (0~1)
        risk = min(potential, 1.0)
        return potential, risk

    def _resolution_advice(self, p1: np.ndarray, v1: np.ndarray,
                           p2: np.ndarray, v2: np.ndarray) -> Optional[float]:
        """
        基于势场梯度给出建议航向调整角（度）
        建议 p1 的目标向右转以远离 p2 的势场
        返回建议右转角度，None 表示无冲突不需调整
        """
        _, risk = self._modified_potential(p1, v1, p2, v2)
        if risk < 0.3:
            return None
        # 计算从 p1 指向 p2 的方位角
        de = p2[0] - p1[0]
        dn = p2[1] - p1[1]
        bearing_to_threat = math.degrees(math.atan2(de, dn))
        if bearing_to_threat < 0:
            bearing_to_threat += 360.0
        # 建议右转 30~60 度远离威胁
        advice = 30.0 + risk * 30.0  # risk=0.3→30°, risk=1.0→60°
        return advice

    def detect_frame(self, tracks_at_t: Dict[str, pd.Series],
                     ref_lat: float, ref_lon: float) -> List[Dict]:
        """
        对单一时刻的多目标状态进行冲突检测
        tracks_at_t: {track_id: Series with e,n,u,speed,heading, ...}
        """
        events = []
        ids = list(tracks_at_t.keys())

        # 预计算 ENU 位置和速度矢量
        states = {}
        for tid, row in tracks_at_t.items():
            e = row.get("e")
            n = row.get("n")
            u = row.get("u", 0.0)
            if pd.isna(e) or pd.isna(n):
                continue
            p = np.array([float(e), float(n), float(u)])
            speed = row.get("speed", 0.0)
            heading = row.get("heading", 0.0)
            if pd.isna(speed) or pd.isna(heading):
                v = np.array([0.0, 0.0, 0.0])
            else:
                h = math.radians(float(heading))
                v = np.array([float(speed) * math.sin(h),
                              float(speed) * math.cos(h),
                              0.0])
            states[tid] = (p, v)

        ids = list(states.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                tid_a, tid_b = ids[i], ids[j]
                p1, v1 = states[tid_a]
                p2, v2 = states[tid_b]

                dcpa, tcpa, rel_speed = self._dcpa_tcpa(p1, v1, p2, v2)
                current_dist = np.linalg.norm(p2 - p1)

                # 过滤准静止/低速目标对：数据集中存在大量地面静止杂波与停放目标，
                # 它们之间不应产生大量冲突告警，否则会造成虚警淹没。
                if rel_speed < 5.0 and current_dist < self.cfg["dcpa_warning_m"] * 2:
                    continue

                pot, risk = self._modified_potential(p1, v1, p2, v2)

                # 分级判决（TCPA=inf 时不触发时间相关阈值）
                level = "normal"
                severity = "low"
                tcpa_finite = tcpa != float('inf')
                if dcpa < self.cfg["dcpa_critical_m"] and tcpa_finite and tcpa < self.cfg["tcpa_critical_sec"]:
                    level = "critical"
                    severity = "critical"
                elif dcpa < self.cfg["dcpa_warning_m"] and tcpa_finite and tcpa < self.cfg["tcpa_warning_sec"]:
                    level = "warning"
                    severity = "high"
                elif risk > 0.5:
                    level = "caution"
                    severity = "medium"

                if level == "normal":
                    continue

                # 解脱建议（给 tid_a）
                advice = self._resolution_advice(p1, v1, p2, v2)

                events.append({
                    "time_sec": tracks_at_t[tid_a].get("time_sec", 0),
                    "track_a": tid_a,
                    "track_b": tid_b,
                    "dcpa_m": round(dcpa, 1),
                    "tcpa_s": round(tcpa, 1) if tcpa_finite else None,
                    "rel_speed_ms": round(rel_speed, 2),
                    "potential_risk": round(risk, 3),
                    "level": level,
                    "severity": severity,
                    "advice_turn_deg": round(advice, 1) if advice is not None else None,
                })
        return events

    def detect_all(self, fused_tracks: Dict[str, pd.DataFrame],
                   time_step: float = 1.0,
                   ref_lat: float = 0.0, ref_lon: float = 0.0) -> pd.DataFrame:
        """
        对全部融合航迹按时间帧滑动检测冲突
        """
        # 合并所有航迹到统一表
        all_frames = []
        for stid, df in fused_tracks.items():
            if df.empty:
                continue
            sub = df.copy()
            sub["track_id"] = stid
            all_frames.append(sub)
        if not all_frames:
            return pd.DataFrame()
        master = pd.concat(all_frames, ignore_index=True)

        # 按时间帧聚合
        master["time_frame"] = (master["time_sec"] / time_step).round() * time_step
        events = []
        for t, grp in master.groupby("time_frame"):
            tracks_at_t = {}
            for _, row in grp.iterrows():
                tid = row["track_id"]
                tracks_at_t[tid] = row
            frame_events = self.detect_frame(tracks_at_t, ref_lat, ref_lon)
            for ev in frame_events:
                ev["time_sec"] = t
            events.extend(frame_events)

        return pd.DataFrame(events)

    def check_geofence(self, track_df: pd.DataFrame, track_id: str) -> List[Dict]:
        """
        地理围栏入侵检测
        """
        events = []
        df = track_df.sort_values("time_sec").copy()
        for _, row in df.iterrows():
            lat = row.get("lat")
            lon = row.get("lon")
            alt = row.get("alt_geom") if not pd.isna(row.get("alt_geom")) else row.get("u")
            if pd.isna(lat) or pd.isna(lon):
                continue
            for cx, cy, radius, alt_min, alt_max, name in NO_FLY_ZONES:
                d = haversine_distance(lat, lon, cy, cx)
                if d < radius:
                    if not pd.isna(alt) and alt_min <= alt <= alt_max:
                        events.append({
                            "time_sec": row["time_sec"],
                            "track_id": track_id,
                            "type": "geofence_intrusion",
                            "fence_name": name,
                            "distance_to_center_m": round(d, 1),
                            "level": "warning",
                            "severity": "high",
                            "description": f"侵入 {name}，距中心 {d:.0f}m",
                        })
        return events
