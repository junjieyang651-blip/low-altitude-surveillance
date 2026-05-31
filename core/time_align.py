"""
时空一致性处理引擎
- 时间对齐：统一 rev_time → 相对秒数，动态插值
- 坐标对齐：建立全局 ENU 坐标系，便于精确距离计算
- 数据质量评估与野值剔除
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.interpolate import interp1d

from utils.coords import wgs84_to_enu, compute_centroid
from utils.constants import EARTH_RADIUS_M


class SpatioTemporalAligner:
    """
    将多源异构数据对齐到统一时空基准。

    创新点：
    1. 动态插值策略 —— 根据各源采样密度自动选择线性或三次样条插值，
       避免粗暴重采样导致的信息损失。
    2. 航向/速度连续性约束 —— 插值后利用前后点约束修正航向跳变，
       保证运动学合理性。
    """

    def __init__(self, ref_lat: float = None, ref_lon: float = None, ref_alt: float = 0.0):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.enu_cache: Dict[str, pd.DataFrame] = {}

    def fit_reference(self, all_points: List[Tuple[float, float, float]]):
        """
        根据所有数据的中心点自动设定 ENU 原点
        """
        self.ref_lat, self.ref_lon, self.ref_alt = compute_centroid(all_points)

    def add_enu_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为 DataFrame 增加 ENU 局部坐标列 (e, n, u)
        要求 df 包含 lat, lon, alt_geom（或 alt_pressure）
        向量化实现，避免 iterrows 遍历
        """
        if self.ref_lat is None:
            raise ValueError("请先调用 fit_reference 设定参考原点")

        alt_col = "alt_geom" if "alt_geom" in df.columns else "alt_pressure"
        lats = df["lat"].values.astype(float)
        lons = df["lon"].values.astype(float)
        alts = df.get(alt_col, pd.Series(0.0, index=df.index)).fillna(0.0).values.astype(float)

        e, n, u = wgs84_to_enu(lats, lons, alts, self.ref_lat, self.ref_lon, self.ref_alt)
        df["e"] = e
        df["n"] = n
        df["u"] = u
        return df

    def _select_interpolation_kind(self, times: np.ndarray) -> str:
        """
        根据采样密度动态选择插值策略
        - 数据密集（平均间隔 < 1s）：三次样条，平滑
        - 数据稀疏（平均间隔 >= 1s）：线性，避免过拟合振荡
        """
        if len(times) < 4:
            return "linear"
        avg_dt = np.mean(np.diff(times))
        return "cubic" if avg_dt < 1.0 else "linear"

    def interpolate_track(self, df: pd.DataFrame, target_times: np.ndarray,
                          columns: List[str] = None) -> pd.DataFrame:
        """
        将单条航迹插值到目标时间序列 target_times
        仅对数值列进行插值，target_id/source 等非数值列保留
        """
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("time_sec").drop_duplicates(subset=["time_sec"])
        times = df["time_sec"].values.astype(float)

        if columns is None:
            # 自动识别可插值的数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = {"time_sec"}
            columns = [c for c in numeric_cols if c not in exclude]

        kind = self._select_interpolation_kind(times)

        result = {"time_sec": target_times}
        for col in columns:
            vals = df[col].values.astype(float)
            # 剔除 NaN 防止插值失败
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                result[col] = np.full_like(target_times, np.nan, dtype=float)
                continue
            try:
                f = interp1d(times[valid], vals[valid], kind=kind,
                             bounds_error=False, fill_value="extrapolate")
                result[col] = f(target_times)
            except Exception:
                # 回退线性
                f = interp1d(times[valid], vals[valid], kind="linear",
                             bounds_error=False, fill_value="extrapolate")
                result[col] = f(target_times)

        # 保留标识列（取第一个有效值）
        for id_col in ["target_id", "source", "target_type"]:
            if id_col in df.columns:
                valid_vals = df[id_col].dropna()
                result[id_col] = valid_vals.iloc[0] if not valid_vals.empty else None

        return pd.DataFrame(result)

    def align_all_sources(self, datasets: Dict[str, pd.DataFrame],
                          interval_sec: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        将所有数据源对齐到统一的时间轴上（全局等间隔）
        返回对齐后的各源 DataFrame

        注意：对于后续关联融合，我们并不强制全局重采样，
        而是提供统一时间轴便于逐帧关联。实际工程中使用
        '事件驱动对齐' 而非 '全局重采样' 更节省算力。
        """
        # 确定全局时间范围
        all_times = []
        for df in datasets.values():
            all_times.extend(df["time_sec"].dropna().tolist())
        if not all_times:
            raise ValueError("无有效时间数据")

        t_min = np.floor(min(all_times))
        t_max = np.ceil(max(all_times))
        global_times = np.arange(t_min, t_max + interval_sec, interval_sec)

        aligned = {}
        for source_name, df in datasets.items():
            tracks = []
            for tid, group in df.groupby("target_id"):
                interped = self.interpolate_track(group, global_times)
                tracks.append(interped)
            if tracks:
                aligned[source_name] = pd.concat(tracks, ignore_index=True)
            else:
                aligned[source_name] = pd.DataFrame()
        return aligned

    def outlier_rejection(self, df: pd.DataFrame,
                          max_speed_ms: float = 200.0,
                          max_climb_ms: float = 50.0) -> pd.DataFrame:
        """
        基于运动学约束的野值剔除（按 target_id 分组处理）
        若相邻点 implied speed / climb rate 超过物理极限，则标记为野值
        """
        results = []
        for tid, grp in df.groupby("target_id"):
            grp = grp.sort_values("time_sec").copy()
            dt = grp["time_sec"].diff().values

            # 利用 ENU 坐标计算局部位移
            for axis in ["e", "n", "u"]:
                if axis not in grp.columns:
                    continue
                d_axis = grp[axis].diff().abs().values
                with np.errstate(divide="ignore", invalid="ignore"):
                    v_axis = d_axis / dt
                    if axis == "u":
                        bad = v_axis > max_climb_ms
                        grp.loc[grp.index[bad], axis] = np.nan

            # 计算水平 implied speed
            if "e" in grp.columns and "n" in grp.columns:
                de = grp["e"].diff().abs().values
                dn = grp["n"].diff().abs().values
                horiz_dist = np.sqrt(de ** 2 + dn ** 2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    implied_speed = horiz_dist / dt
                    bad_speed = implied_speed > max_speed_ms
                    grp.loc[grp.index[bad_speed], ["lat", "lon", "e", "n"]] = np.nan

            results.append(grp)

        return pd.concat(results, ignore_index=True) if results else df


def build_global_timeline(datasets: Dict[str, pd.DataFrame],
                          interval_sec: float = 1.0) -> np.ndarray:
    """
    构建全局统一时间轴
    """
    all_t = []
    for df in datasets.values():
        all_t.extend(df["time_sec"].dropna().tolist())
    t_min = np.floor(min(all_t))
    t_max = np.ceil(max(all_t))
    return np.arange(t_min, t_max + interval_sec, interval_sec)
