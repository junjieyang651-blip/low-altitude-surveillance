"""
目标分类引擎 — 轻量、可解释、分层决策

不堆砌深度学习，而是结合数据源特性、运动学特征、频谱先验进行多层规则推理。
符合低空监视工程落地需求：实时、可解释、低算力。
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Optional

from utils.constants import CLASSIFICATION


class TargetClassifier:
    """
    分类逻辑：
    1. 若频谱直接识别出 UAV 机型 → 直接判定为对应无人机类别（置信度最高）
    2. 若 ADS-B/Remote ID 有合作标识 → 结合高度+速度做二次校验
    3. 纯雷达目标 → 基于运动学特征推断
    """

    def __init__(self):
        self.uav_alt_thr = CLASSIFICATION["uav_altitude_threshold_m"]
        self.uav_spd_thr = CLASSIFICATION["uav_speed_threshold_ms"]

    def classify_from_spectrum(self, spectrum_df: pd.DataFrame) -> Dict[str, str]:
        """
        从频谱数据提取机型识别结果
        返回: {uav_id: category_label}
        """
        labels = {}
        if spectrum_df.empty or "uav_model_text" not in spectrum_df.columns:
            return labels
        for uid, grp in spectrum_df.groupby("target_id"):
            models = grp["uav_model_text"].dropna().unique()
            if len(models) > 0:
                # 取出现频次最高的机型
                best = grp["uav_model_text"].value_counts().idxmax()
                best = str(best).lower()
                if any(k in best for k in ["mini", "air", "fpv", "tello", "spark"]):
                    labels[str(uid)] = "uav_consumer"
                elif any(k in best for k in ["m300", "m350", "m30", "t40", "t50", "industry", "enterprise"]):
                    labels[str(uid)] = "uav_industrial"
                else:
                    labels[str(uid)] = "uav_unknown"
        return labels

    def classify_track(self, track_df: pd.DataFrame,
                       spectrum_labels: Optional[Dict[str, str]] = None,
                       source_hint: Optional[str] = None) -> Dict:
        """
        对单条航迹进行分类
        track_df 需包含：lat/lon/alt_geom 或 u, speed, heading, climb_rate

        返回字典：
        {
            "category": "ga" | "uav_consumer" | "uav_industrial" | "unknown",
            "confidence": float,
            "reason": str,
            "source_evidence": ["spectrum", "altitude_rule", ...]
        }
        """
        spectrum_labels = spectrum_labels or {}
        evidence = []

        # 尝试匹配频谱先验（通过 target_id 模糊匹配）
        tid = str(track_df["target_id"].iloc[0]) if "target_id" in track_df.columns else None
        if tid and tid in spectrum_labels:
            cat = spectrum_labels[tid]
            evidence.append("spectrum")
            return {
                "category": cat,
                "confidence": CLASSIFICATION["spectrum_confidence"],
                "reason": f"频谱直接识别为 {cat}",
                "source_evidence": evidence,
            }

        # 运动学统计
        alt_mean = track_df["alt_geom"].mean() if "alt_geom" in track_df.columns else (
            track_df["u"].mean() if "u" in track_df.columns else np.nan)
        speed_mean = track_df["speed"].mean() if "speed" in track_df.columns else np.nan
        speed_max = track_df["speed"].max() if "speed" in track_df.columns else np.nan

        # 数据源先验
        if source_hint == "adsb":
            # ADS-B 几乎全是通航飞机，但需校验是否也可能是大型工业无人机
            evidence.append("adsb_source")
            if not math.isnan(speed_max) and speed_max > 80:
                evidence.append("speed_rule")
                return {
                    "category": "ga",
                    "confidence": 0.90,
                    "reason": "ADS-B源且高速，判定为通航飞机",
                    "source_evidence": evidence,
                }

        if source_hint == "remote_id":
            # Remote ID 是无人机专属
            evidence.append("remote_id_source")
            if not math.isnan(alt_mean) and alt_mean < 120:
                evidence.append("altitude_rule")
                return {
                    "category": "uav_consumer",
                    "confidence": 0.85,
                    "reason": "Remote ID源且低高度，判定为消费级无人机",
                    "source_evidence": evidence,
                }
            else:
                evidence.append("altitude_rule")
                return {
                    "category": "uav_industrial",
                    "confidence": 0.80,
                    "reason": "Remote ID源且中高高度，判定为行业级无人机",
                    "source_evidence": evidence,
                }

        # 纯运动学推断（主要用于雷达/频谱无法直接定位的目标）
        if not math.isnan(speed_mean) and not math.isnan(alt_mean):
            if speed_mean > self.uav_spd_thr or alt_mean > self.uav_alt_thr:
                evidence.append("speed_rule")
                cat = "ga"
                reason = f"速度{speed_mean:.1f}m/s或高度{alt_mean:.1f}m超出典型无人机范围，判为通航"
            else:
                evidence.append("altitude_rule")
                cat = "uav_consumer"
                reason = f"低高度({alt_mean:.1f}m)低速度({speed_mean:.1f}m/s)，判为消费级无人机"
            return {
                "category": cat,
                "confidence": 0.70,
                "reason": reason,
                "source_evidence": evidence,
            }

        return {
            "category": "unknown",
            "confidence": 0.0,
            "reason": "信息不足，无法分类",
            "source_evidence": evidence,
        }

    def classify_all(self, datasets: Dict[str, pd.DataFrame],
                     fused_tracks: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        对所有航迹（单源原始 + 融合后）进行分类，输出分类结果表
        """
        spectrum_labels = self.classify_from_spectrum(datasets.get("spectrum", pd.DataFrame()))

        records = []
        # 先对原始单源航迹分类
        for src, df in datasets.items():
            if df.empty or "target_id" not in df.columns:
                continue
            for tid, grp in df.groupby("target_id"):
                result = self.classify_track(grp, spectrum_labels=spectrum_labels, source_hint=src)
                records.append({
                    "track_id": f"{src}:{tid}",
                    "source": src,
                    "original_tid": str(tid),
                    "category": result["category"],
                    "confidence": result["confidence"],
                    "reason": result["reason"],
                })

        # 对融合航迹分类（综合各源证据）
        if fused_tracks:
            for stid, fdf in fused_tracks.items():
                # 融合航迹没有单一 source_hint，基于运动学推断
                result = self.classify_track(fdf, spectrum_labels=spectrum_labels, source_hint=None)
                records.append({
                    "track_id": stid,
                    "source": "fused",
                    "original_tid": stid,
                    "category": result["category"],
                    "confidence": min(result["confidence"] + 0.05, 0.99),
                    "reason": "融合航迹," + result["reason"],
                })

        return pd.DataFrame(records)
