"""
异常检测双引擎

创新点：
1. 规则引擎 —— 基于航空器物理极限检测"不可能运动"，100%可解释，100%置信度
2. 统计引擎 —— 基于航迹局部统计特征检测"行为偏离"，捕捉规则外的软异常
3. 信号完整性监控 —— 检测合作目标（ADS-B/Remote ID）信号异常丢失

两者互补，避免纯深度学习的黑盒与不可解释问题。
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from utils.constants import AIRCRAFT_LIMITS, get_category_by_limits


class RuleBasedAnomalyEngine:
    """
    基于物理约束的硬异常检测
    """

    ANOMALY_TYPES = {
        "overspeed": "超速",
        "stall_speed": "低于失速速度",
        "excessive_climb": "爬升率超限",
        "excessive_descent": "下降率超限",
        "excessive_turn": "转弯率超限",
        "altitude_ceiling": "高度超限",
        "impossible_accel": "加速度超限",
    }

    def detect_track(self, track_df: pd.DataFrame) -> List[Dict]:
        """
        对单条航迹逐点检测物理异常
        返回异常事件列表
        """
        events = []
        df = track_df.sort_values("time_sec").copy()
        if len(df) < 2:
            return events

        # 推断机型类别以选用对应约束
        alt_mean = df["alt_geom"].mean() if "alt_geom" in df.columns else df["u"].mean()
        spd_mean = df["speed"].mean()
        # 用首点爬升率估算
        climb_est = 0.0
        if "climb_rate" in df.columns and not df["climb_rate"].isna().all():
            climb_est = df["climb_rate"].mean()
        elif "u" in df.columns:
            dt = df["time_sec"].diff().mean()
            if dt > 0:
                climb_est = df["u"].diff().abs().mean() / dt

        cat = get_category_by_limits(spd_mean if not pd.isna(spd_mean) else 0,
                                     alt_mean if not pd.isna(alt_mean) else 0,
                                     climb_est)
        limits = AIRCRAFT_LIMITS.get(cat, AIRCRAFT_LIMITS["unknown"])

        for i in range(len(df)):
            row = df.iloc[i]
            t = row["time_sec"]
            speed = row.get("speed")
            alt = row.get("alt_geom") if not pd.isna(row.get("alt_geom")) else row.get("u")
            climb = row.get("climb_rate")

            # 速度约束
            if not pd.isna(speed):
                if speed > limits["max_speed_ms"]:
                    events.append({
                        "time_sec": t, "type": "overspeed",
                        "description": f"速度 {speed:.1f} m/s 超过机型上限 {limits['max_speed_ms']:.1f}",
                        "confidence": 1.0, "severity": "high",
                    })
                elif speed < limits["min_speed_ms"] and speed > 0:
                    events.append({
                        "time_sec": t, "type": "stall_speed",
                        "description": f"速度 {speed:.1f} m/s 低于机型下限 {limits['min_speed_ms']:.1f}",
                        "confidence": 0.90, "severity": "medium",
                    })

            # 高度约束
            if not pd.isna(alt) and alt > limits["max_altitude_m"]:
                events.append({
                    "time_sec": t, "type": "altitude_ceiling",
                    "description": f"高度 {alt:.1f} m 超过机型上限 {limits['max_altitude_m']:.1f}",
                    "confidence": 1.0, "severity": "medium",
                })

            # 爬升/下降率约束
            if not pd.isna(climb):
                if climb > limits["max_climb_rate_ms"]:
                    events.append({
                        "time_sec": t, "type": "excessive_climb",
                        "description": f"爬升率 {climb:.1f} m/s 超限 {limits['max_climb_rate_ms']:.1f}",
                        "confidence": 1.0, "severity": "high",
                    })
                elif climb < -limits["max_descent_rate_ms"]:
                    events.append({
                        "time_sec": t, "type": "excessive_descent",
                        "description": f"下降率 {abs(climb):.1f} m/s 超限 {limits['max_descent_rate_ms']:.1f}",
                        "confidence": 1.0, "severity": "high",
                    })

        # 转弯率约束（需要前后点计算）
        # 要求时间间隔 >= 2s，避免高频采样噪声导致虚假高转弯率
        if "heading" in df.columns and len(df) >= 3:
            headings = df["heading"].values
            times = df["time_sec"].values
            for i in range(1, len(df) - 1):
                h_prev = headings[i - 1]
                h_next = headings[i + 1]
                if pd.isna(h_prev) or pd.isna(h_next):
                    continue
                dh = abs(h_next - h_prev)
                dh = min(dh, 360.0 - dh)
                dt = times[i + 1] - times[i - 1]
                # 过滤高频噪声：时间间隔太小则不计算
                if dt >= 2.0:
                    turn_rate = dh / dt  # deg/s
                    if turn_rate > limits["max_turn_rate_degs"]:
                        events.append({
                            "time_sec": times[i], "type": "excessive_turn",
                            "description": f"转弯率 {turn_rate:.1f}°/s 超限 {limits['max_turn_rate_degs']:.1f}",
                            "confidence": 1.0, "severity": "high",
                        })

        # 加速度约束（水平面）
        if "speed" in df.columns and len(df) >= 2:
            speeds = df["speed"].values
            times = df["time_sec"].values
            for i in range(1, len(df)):
                if pd.isna(speeds[i]) or pd.isna(speeds[i - 1]):
                    continue
                dv = speeds[i] - speeds[i - 1]
                dt = times[i] - times[i - 1]
                if dt > 0:
                    accel = abs(dv) / dt
                    # 通用物理极限：一般航空器水平加速度很少超过 3g (~30 m/s²)
                    if accel > 30.0:
                        events.append({
                            "time_sec": times[i], "type": "impossible_accel",
                            "description": f"水平加速度 {accel:.1f} m/s² 超出物理极限",
                            "confidence": 1.0, "severity": "critical",
                        })

        return events


class StatisticalAnomalyEngine:
    """
    基于统计的行为偏离检测
    """

    def detect_track(self, track_df: pd.DataFrame,
                     z_thresh: float = 3.0,
                     window: int = 5) -> List[Dict]:
        """
        使用滑动窗口 Z-Score 检测局部偏离
        """
        events = []
        df = track_df.sort_values("time_sec").copy()
        if len(df) < window * 2:
            return events

        numeric_cols = ["speed", "heading", "climb_rate", "u"]
        for col in numeric_cols:
            if col not in df.columns or df[col].isna().all():
                continue
            vals = df[col].astype(float).values
            for i in range(window, len(df) - window):
                local = vals[i - window:i + window + 1]
                local_valid = local[~np.isnan(local)]
                if len(local_valid) < window:
                    continue
                mean_loc = np.mean(local_valid)
                std_loc = np.std(local_valid)
                if std_loc < 1e-6:
                    continue
                z = (vals[i] - mean_loc) / std_loc
                if abs(z) > z_thresh:
                    events.append({
                        "time_sec": df.iloc[i]["time_sec"],
                        "type": f"statistical_deviation_{col}",
                        "description": f"{col} 局部Z-Score={z:.2f} 偏离正常范围",
                        "confidence": min(abs(z) / (z_thresh * 2), 0.99),
                        "severity": "medium" if abs(z) < z_thresh * 1.5 else "high",
                    })
        return events


class SignalIntegrityMonitor:
    """
    监测合作目标信号完整性：
    - ADS-B 航迹突然中断
    - Remote ID 航迹突然中断
    - 融合航迹中合作源消失，仅剩雷达源持续跟踪（可能的敌我识别异常）
    """

    def detect_loss(self, track_df: pd.DataFrame,
                    source_col: str = "source",
                    gap_thresh_sec: float = 60.0) -> List[Dict]:
        """
        检测单条航迹内的时间缺失 gap
        """
        events = []
        df = track_df.sort_values("time_sec").copy()
        dt = df["time_sec"].diff().values
        for i in range(1, len(df)):
            if dt[i] > gap_thresh_sec:
                events.append({
                    "time_sec": df.iloc[i]["time_sec"],
                    "type": "signal_loss",
                    "description": f"信号中断 {dt[i]:.1f} 秒，超过阈值 {gap_thresh_sec}",
                    "confidence": 0.85,
                    "severity": "high" if dt[i] > gap_thresh_sec * 2 else "medium",
                })
        return events

    def detect_cooperative_dropout(self, fused_track: pd.DataFrame) -> List[Dict]:
        """
        检测融合航迹中合作源（adsb/remote_id）突然消失，仅余非合作源的情况
        """
        events = []
        if "sources" not in fused_track.columns or fused_track.empty:
            return events

        df = fused_track.sort_values("time_sec").copy()
        prev_coop = False
        for _, row in df.iterrows():
            srcs = str(row.get("sources", ""))
            has_coop = any(s in srcs for s in ["adsb", "remote_id"])
            if prev_coop and not has_coop:
                events.append({
                    "time_sec": row["time_sec"],
                    "type": "cooperative_dropout",
                    "description": "合作目标信号丢失，仅剩非合作雷达跟踪，存在潜在威胁",
                    "confidence": 0.80,
                    "severity": "critical",
                })
            prev_coop = has_coop
        return events


class AnomalyDetector:
    """
    统一异常检测接口：整合规则+统计+信号完整性
    """

    def __init__(self):
        self.rule_engine = RuleBasedAnomalyEngine()
        self.stat_engine = StatisticalAnomalyEngine()
        self.sig_monitor = SignalIntegrityMonitor()

    def detect(self, track_df: pd.DataFrame,
               track_id: str,
               is_fused: bool = False) -> pd.DataFrame:
        """
        对单条航迹执行全量异常检测，返回异常事件 DataFrame
        """
        all_events = []

        # 1. 规则引擎
        all_events.extend(self.rule_engine.detect_track(track_df))

        # 2. 统计引擎
        all_events.extend(self.stat_engine.detect_track(track_df))

        # 3. 信号完整性
        all_events.extend(self.sig_monitor.detect_loss(track_df))
        if is_fused:
            all_events.extend(self.sig_monitor.detect_cooperative_dropout(track_df))

        if not all_events:
            return pd.DataFrame()

        ev_df = pd.DataFrame(all_events)
        ev_df["track_id"] = track_id
        # 去重：同类型、同时刻（±1s）只保留最严重的一条
        ev_df = ev_df.sort_values("severity", key=lambda x: x.map({"low": 0, "medium": 1, "high": 2, "critical": 3}),
                                   ascending=False)
        ev_df["time_round"] = ev_df["time_sec"].round(0)
        ev_df = ev_df.drop_duplicates(subset=["track_id", "type", "time_round"], keep="first")
        ev_df = ev_df.drop(columns=["time_round"])
        return ev_df.sort_values("time_sec").reset_index(drop=True)

    def detect_all(self, datasets: Dict[str, pd.DataFrame],
                   fused_tracks: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        批量检测所有航迹
        """
        results = []
        # 原始航迹
        for src, df in datasets.items():
            if df.empty or "target_id" not in df.columns:
                continue
            for tid, grp in df.groupby("target_id"):
                ev = self.detect(grp, track_id=f"{src}:{tid}", is_fused=False)
                if not ev.empty:
                    results.append(ev)
        # 融合航迹
        if fused_tracks:
            for stid, fdf in fused_tracks.items():
                ev = self.detect(fdf, track_id=stid, is_fused=True)
                if not ev.empty:
                    results.append(ev)

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)
