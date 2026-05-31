"""
D-S 证据理论冲突处理模块

基于 Dempster-Shafer 证据理论实现传感器信息的不确定性融合与冲突检测。
核心功能：
1. 将每个传感器的检测结果建模为基本概率分配函数 (BPA/mass function)
2. 通过Dempster组合规则融合多源证据
3. 检测高冲突度（K值）情况，采用修正融合策略
4. 用于判定：目标是否真实存在、传感器是否可信

参考文献:
  [1] Shafer, "A Mathematical Theory of Evidence", Princeton, 1976
  [2] Yager, "On the Dempster-Shafer framework and new combination rules", 1987
  [3] Murphy, "Combining belief functions when evidence conflicts", 2000
"""

import numpy as np
from typing import Dict, List, Tuple


# 辨识框架 (Frame of Discernment)
# Θ = {Target_Real, Target_False, Unknown}
FRAME = {"real", "false"}  # 目标真实存在 / 虚假目标


class MassFunction:
    """基本概率分配 (Basic Probability Assignment)"""

    def __init__(self):
        # 焦元 → 质量值
        # 键: frozenset, 值: float
        self.masses: Dict[frozenset, float] = {}

    def assign(self, hypothesis: frozenset, mass: float):
        """为焦元分配质量"""
        self.masses[hypothesis] = mass

    @property
    def belief(self) -> Dict[frozenset, float]:
        """信度函数 Bel(A)"""
        bel = {}
        for A in self.masses:
            bel[A] = sum(m for B, m in self.masses.items() if B <= A)
        return bel

    @property
    def plausibility(self) -> Dict[frozenset, float]:
        """似真度 Pl(A)"""
        pl = {}
        for A in self.masses:
            pl[A] = sum(m for B, m in self.masses.items() if A & B)
        return pl

    def normalize(self):
        """归一化质量函数"""
        total = sum(self.masses.values())
        if total > 0:
            self.masses = {k: v / total for k, v in self.masses.items()}


class DSEvidenceEngine:
    """D-S证据融合引擎"""

    # 传感器可靠性先验
    SENSOR_RELIABILITY = {
        "adsb": 0.90,       # ADS-B可靠性高（机载设备）
        "remote_id": 0.85,  # Remote ID可靠性较高
        "radar": 0.75,      # 雷达存在杂波虚警
        "spectrum": 0.55,   # 频谱检测误报率高
    }

    def __init__(self, conflict_threshold: float = 0.7):
        """
        Args:
            conflict_threshold: 冲突系数K超过此值时触发冲突处理
        """
        self.conflict_threshold = conflict_threshold

    def build_mass_from_detection(self, source: str, detected: bool,
                                  signal_quality: float = 1.0) -> MassFunction:
        """
        根据传感器检测结果构建BPA

        Args:
            source: 传感器类型 (adsb/remote_id/radar/spectrum)
            detected: 是否检测到目标
            signal_quality: 信号质量 [0,1]，影响证据强度
        """
        m = MassFunction()
        reliability = self.SENSOR_RELIABILITY.get(source, 0.6)
        # 综合可靠性 = 先验可靠性 × 信号质量
        effective_rel = reliability * signal_quality

        real_hyp = frozenset({"real"})
        false_hyp = frozenset({"false"})
        theta = frozenset({"real", "false"})  # 全集表示不确定

        if detected:
            # 检测到目标 → 支持"真实"
            m.assign(real_hyp, effective_rel * 0.8)
            m.assign(false_hyp, (1 - effective_rel) * 0.1)
            m.assign(theta, 1.0 - effective_rel * 0.8 - (1 - effective_rel) * 0.1)
        else:
            # 未检测到 → 支持"虚假"（但可能是遮挡/盲区）
            m.assign(real_hyp, (1 - effective_rel) * 0.1)
            m.assign(false_hyp, effective_rel * 0.5)
            m.assign(theta, 1.0 - (1 - effective_rel) * 0.1 - effective_rel * 0.5)

        m.normalize()
        return m

    def build_mass_from_consistency(self, source: str,
                                    position_error: float,
                                    error_threshold: float = 500.0) -> MassFunction:
        """
        根据传感器观测一致性构建BPA

        如果某传感器与其他源差距过大（position_error > threshold），
        则其支持"虚假"的证据增加。

        Args:
            source: 传感器类型
            position_error: 与融合位置的偏差 (m)
            error_threshold: 偏差阈值
        """
        m = MassFunction()
        reliability = self.SENSOR_RELIABILITY.get(source, 0.6)

        real_hyp = frozenset({"real"})
        false_hyp = frozenset({"false"})
        theta = frozenset({"real", "false"})

        # 偏差越大 → 越可能是虚警
        conflict_degree = min(position_error / error_threshold, 1.0)

        m.assign(real_hyp, reliability * (1 - conflict_degree) * 0.7)
        m.assign(false_hyp, conflict_degree * 0.6)
        m.assign(theta, 1.0 - reliability * (1 - conflict_degree) * 0.7 - conflict_degree * 0.6)
        m.normalize()
        return m

    def dempster_combine(self, m1: MassFunction, m2: MassFunction) -> Tuple[MassFunction, float]:
        """
        Dempster组合规则

        Returns:
            (combined_mass, conflict_K)
            conflict_K: 冲突系数，0=完全一致，1=完全冲突
        """
        combined = MassFunction()
        K = 0.0  # 冲突系数

        for A, ma in m1.masses.items():
            for B, mb in m2.masses.items():
                intersection = A & B
                if intersection:
                    key = frozenset(intersection)
                    combined.masses[key] = combined.masses.get(key, 0.0) + ma * mb
                else:
                    K += ma * mb

        # 归一化（Dempster规则）
        if K < 1.0:
            for key in combined.masses:
                combined.masses[key] /= (1.0 - K)

        return combined, K

    def murphy_combine(self, mass_list: List[MassFunction]) -> Tuple[MassFunction, float]:
        """
        Murphy平均证据组合（处理高冲突场景）

        当K > threshold时使用此方法：先对所有BPA取平均，再自组合n-1次

        Args:
            mass_list: 多个传感器的BPA列表

        Returns:
            (combined_mass, max_conflict_K)
        """
        if not mass_list:
            m_empty = MassFunction()
            m_empty.assign(frozenset({"real", "false"}), 1.0)
            return m_empty, 0.0

        if len(mass_list) == 1:
            return mass_list[0], 0.0

        # 计算平均BPA
        all_keys = set()
        for m in mass_list:
            all_keys.update(m.masses.keys())

        avg_mass = MassFunction()
        for key in all_keys:
            avg_val = sum(m.masses.get(key, 0.0) for m in mass_list) / len(mass_list)
            if avg_val > 0:
                avg_mass.assign(key, avg_val)
        avg_mass.normalize()

        # 自组合 n-1 次
        result = avg_mass
        max_K = 0.0
        for _ in range(len(mass_list) - 1):
            result, K = self.dempster_combine(result, avg_mass)
            max_K = max(max_K, K)

        return result, max_K

    def fuse_evidence(self, mass_list: List[MassFunction]) -> Dict[str, float]:
        """
        智能融合：根据冲突程度选择策略

        Returns:
            {
                "real_belief": float,    # 目标真实存在的信度
                "false_belief": float,   # 虚假目标的信度
                "uncertainty": float,    # 不确定度
                "conflict_K": float,     # 最大冲突系数
                "strategy": str,         # 使用的策略
            }
        """
        if not mass_list:
            return {"real_belief": 0.5, "false_belief": 0.0,
                    "uncertainty": 0.5, "conflict_K": 0.0, "strategy": "none"}

        if len(mass_list) == 1:
            m = mass_list[0]
            real_hyp = frozenset({"real"})
            false_hyp = frozenset({"false"})
            return {
                "real_belief": m.masses.get(real_hyp, 0.0),
                "false_belief": m.masses.get(false_hyp, 0.0),
                "uncertainty": m.masses.get(frozenset({"real", "false"}), 0.0),
                "conflict_K": 0.0,
                "strategy": "single",
            }

        # 先尝试 Dempster 组合，检测冲突
        result = mass_list[0]
        max_K = 0.0
        for i in range(1, len(mass_list)):
            result, K = self.dempster_combine(result, mass_list[i])
            max_K = max(max_K, K)

        strategy = "dempster"

        # 如果冲突过高，改用Murphy方法
        if max_K > self.conflict_threshold:
            result, max_K = self.murphy_combine(mass_list)
            strategy = "murphy"

        real_hyp = frozenset({"real"})
        false_hyp = frozenset({"false"})
        theta = frozenset({"real", "false"})

        return {
            "real_belief": result.masses.get(real_hyp, 0.0),
            "false_belief": result.masses.get(false_hyp, 0.0),
            "uncertainty": result.masses.get(theta, 0.0),
            "conflict_K": max_K,
            "strategy": strategy,
        }

    def evaluate_track_credibility(self, observations: List[Dict]) -> Dict:
        """
        评估航迹可信度

        对于一条系统航迹上的多源观测，综合评估其是否为真实目标。

        Args:
            observations: [{"source": str, "detected": bool, "signal_quality": float}, ...]

        Returns:
            {"credibility": float, "conflict_K": float, "decision": str}
        """
        mass_list = []
        for obs in observations:
            m = self.build_mass_from_detection(
                obs["source"],
                obs.get("detected", True),
                obs.get("signal_quality", 1.0)
            )
            mass_list.append(m)

        result = self.fuse_evidence(mass_list)

        # 决策规则
        if result["real_belief"] > 0.6:
            decision = "confirmed_real"
        elif result["false_belief"] > 0.5:
            decision = "suspected_false"
        elif result["conflict_K"] > self.conflict_threshold:
            decision = "high_conflict"
        else:
            decision = "uncertain"

        return {
            "credibility": result["real_belief"],
            "false_prob": result["false_belief"],
            "uncertainty": result["uncertainty"],
            "conflict_K": result["conflict_K"],
            "strategy": result["strategy"],
            "decision": decision,
        }
