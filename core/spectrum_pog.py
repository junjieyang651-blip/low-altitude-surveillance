"""
频谱概率占据网格 (Probability Occupancy Grid, POG)

将频谱检测数据建模为空间概率热力图，替代简单的"点位置+权重0.3"方案。
核心思想：
1. 频谱检测只能提供方位角(AOA)和粗略距离估计，无法精确定位
2. 将检测结果映射为空间网格上的占据概率分布
3. 与雷达/ADS-B精确轨迹做时空重叠度匹配（非点距离匹配）

参考文献:
  [1] Elfes, "Using Occupancy Grids for Mobile Robot Perception", Computer, 1989
  [2] Thrun, "Learning Occupancy Grid Maps with Forward Sensor Models", 2003
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter


class ProbabilityOccupancyGrid:
    """概率占据网格"""

    def __init__(self, grid_size_m: float = 100.0,
                 extent_m: float = 10000.0,
                 aoa_sigma_deg: float = 15.0,
                 range_sigma_m: float = 2000.0):
        """
        Args:
            grid_size_m: 网格单元尺寸 (m)
            extent_m: 网格覆盖范围半径 (m)
            aoa_sigma_deg: AOA估计标准差 (度)
            range_sigma_m: 距离估计标准差 (m)
        """
        self.grid_size = grid_size_m
        self.extent = extent_m
        self.aoa_sigma = np.radians(aoa_sigma_deg)
        self.range_sigma = range_sigma_m

        # 网格维度
        self.n_cells = int(2 * extent_m / grid_size_m)
        # 占据概率网格 (log-odds表示)
        self.log_odds = np.zeros((self.n_cells, self.n_cells))
        # 先验 log-odds (0.5概率对应log-odds=0)
        self.l0 = 0.0

    def _enu_to_grid(self, e: float, n: float) -> Tuple[int, int]:
        """ENU坐标→网格索引"""
        col = int((e + self.extent) / self.grid_size)
        row = int((n + self.extent) / self.grid_size)
        col = max(0, min(col, self.n_cells - 1))
        row = max(0, min(row, self.n_cells - 1))
        return row, col

    def _grid_to_enu(self, row: int, col: int) -> Tuple[float, float]:
        """网格索引→ENU坐标（网格中心）"""
        e = col * self.grid_size - self.extent + self.grid_size / 2
        n = row * self.grid_size - self.extent + self.grid_size / 2
        return e, n

    def update_from_spectrum_detection(self, sensor_e: float, sensor_n: float,
                                        aoa_deg: float, signal_strength: float,
                                        estimated_range: float = None):
        """
        根据频谱检测更新占据网格

        将检测信息转换为扇形概率分布：
        - 方位方向: 高斯分布，中心=AOA，σ=aoa_sigma
        - 径向方向: 高斯分布（如有距离估计）或均匀衰减

        Args:
            sensor_e, sensor_n: 传感器位置 (ENU)
            aoa_deg: 到达角估计 (度, 北偏东为正)
            signal_strength: 信号强度 [0,1]，影响更新强度
            estimated_range: 估计距离 (m), 可选
        """
        aoa_rad = np.radians(aoa_deg)
        strength = min(max(signal_strength, 0.1), 1.0)

        # 计算受影响的网格区域（扇形区域）
        max_range = estimated_range * 2 if estimated_range else self.extent * 0.5
        min_range = 50.0  # 最小距离

        # 遍历可能受影响的网格（通过采样扇形区域）
        n_range_samples = int(max_range / self.grid_size)
        n_angle_samples = max(int(4 * self.aoa_sigma / np.radians(2)), 10)

        for ri in range(1, min(n_range_samples, 50)):  # 限制计算量
            r = min_range + ri * self.grid_size
            for ai in range(-n_angle_samples, n_angle_samples + 1):
                angle = aoa_rad + ai * self.aoa_sigma / n_angle_samples

                # 计算该网格点的ENU坐标
                e_point = sensor_e + r * np.sin(angle)
                n_point = sensor_n + r * np.cos(angle)

                row, col = self._enu_to_grid(e_point, n_point)
                if row < 0 or row >= self.n_cells or col < 0 or col >= self.n_cells:
                    continue

                # 计算概率贡献
                # 方位维度: 高斯
                angle_diff = ai * self.aoa_sigma / n_angle_samples
                p_angle = np.exp(-0.5 * (angle_diff / self.aoa_sigma) ** 2)

                # 距离维度
                if estimated_range:
                    range_diff = r - estimated_range
                    p_range = np.exp(-0.5 * (range_diff / self.range_sigma) ** 2)
                else:
                    # 无距离估计时用1/r²衰减
                    p_range = min_range ** 2 / (r ** 2 + 1)

                # 综合概率
                p_detection = strength * p_angle * p_range

                # Log-odds更新
                if p_detection > 0.01:
                    l_update = np.log(p_detection / (1 - min(p_detection, 0.99) + 1e-10))
                    self.log_odds[row, col] += l_update * 0.3  # 衰减因子

        # 限制log-odds范围
        self.log_odds = np.clip(self.log_odds, -5.0, 5.0)

    def get_probability_map(self) -> np.ndarray:
        """获取占据概率图 [0,1]"""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def get_high_probability_cells(self, threshold: float = 0.6) -> List[Dict]:
        """获取高概率占据单元"""
        prob_map = self.get_probability_map()
        cells = []
        rows, cols = np.where(prob_map > threshold)
        for r, c in zip(rows, cols):
            e, n = self._grid_to_enu(r, c)
            cells.append({
                "e": e, "n": n,
                "probability": float(prob_map[r, c]),
                "grid_row": int(r), "grid_col": int(c)
            })
        return cells

    def match_with_track(self, track_points: List[Tuple[float, float]]) -> float:
        """
        计算轨迹与占据网格的时空重叠度

        Args:
            track_points: [(e, n), ...] 轨迹点序列

        Returns:
            overlap_score: [0,1] 重叠度分数
        """
        if not track_points:
            return 0.0

        prob_map = self.get_probability_map()
        scores = []

        for e, n in track_points:
            row, col = self._enu_to_grid(e, n)
            if 0 <= row < self.n_cells and 0 <= col < self.n_cells:
                scores.append(prob_map[row, col])
            else:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def decay(self, factor: float = 0.95):
        """时间衰减：降低历史检测的影响"""
        self.log_odds *= factor

    def reset(self):
        """重置网格"""
        self.log_odds = np.zeros((self.n_cells, self.n_cells))


class SpectrumPOGProcessor:
    """频谱数据POG处理器"""

    def __init__(self, grid_size_m: float = 200.0, extent_m: float = 8000.0):
        self.grid = ProbabilityOccupancyGrid(
            grid_size_m=grid_size_m,
            extent_m=extent_m,
            aoa_sigma_deg=15.0,
            range_sigma_m=1500.0
        )
        # 传感器默认位置（假设在原点附近）
        self.sensor_positions = {}

    def process_spectrum_data(self, spectrum_df: pd.DataFrame,
                              time_window: float = 10.0) -> Dict:
        """
        处理频谱数据生成POG

        Args:
            spectrum_df: 频谱检测数据 (含target_id, time_sec, frequency等)
            time_window: 时间窗口 (s)

        Returns:
            {
                "grid": ProbabilityOccupancyGrid,
                "high_prob_cells": [...],
                "coverage_ratio": float,
            }
        """
        self.grid.reset()

        if spectrum_df.empty:
            return {"grid": self.grid, "high_prob_cells": [], "coverage_ratio": 0.0}

        # 按时间窗口处理
        t_min = spectrum_df["time_sec"].min()
        t_max = spectrum_df["time_sec"].max()

        # 使用频率+信号强度估计方位（简化模型）
        for _, row in spectrum_df.iterrows():
            # 频谱检测的"方位"从target_id hash推导（模拟AOA）
            tid_hash = hash(str(row.get("target_id", ""))) % 360
            aoa = float(tid_hash)

            # 信号强度归一化
            sig_str = row.get("signal_strength", 0.5)
            if pd.isna(sig_str):
                sig_str = 0.3
            sig_str = min(max(float(sig_str), 0.1), 1.0)

            # 从频率估计粗略距离（频率越高通常意味着越近）
            freq = row.get("frequency", 2400)
            if pd.isna(freq):
                freq = 2400
            est_range = max(500.0, 5000.0 - float(freq) * 0.5)

            # 更新POG
            sensor_e, sensor_n = 0.0, 0.0
            self.grid.update_from_spectrum_detection(
                sensor_e, sensor_n,
                aoa, sig_str, est_range
            )

        # 获取结果
        high_prob = self.grid.get_high_probability_cells(threshold=0.55)
        prob_map = self.grid.get_probability_map()
        coverage = float(np.sum(prob_map > 0.5)) / max(prob_map.size, 1)

        return {
            "grid": self.grid,
            "high_prob_cells": high_prob,
            "coverage_ratio": coverage,
            "n_detections": len(spectrum_df),
        }

    def compute_track_spectrum_overlap(self, track_df: pd.DataFrame) -> float:
        """
        计算雷达/ADS-B轨迹与频谱POG的重叠度

        用于航迹关联：如果重叠度高，说明频谱检测与精确轨迹一致

        Args:
            track_df: 含 'e', 'n' 列的轨迹DataFrame

        Returns:
            overlap_score: [0, 1]
        """
        if track_df.empty or "e" not in track_df.columns:
            return 0.0

        points = list(zip(
            track_df["e"].dropna().values,
            track_df["n"].dropna().values
        ))
        return self.grid.match_with_track(points)
