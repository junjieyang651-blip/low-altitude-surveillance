"""
数据读取、清洗与统一格式化
将 4 类异构数据读取为标准化 DataFrame，便于后续模块处理
"""

import os
import re
from typing import Dict, Optional
import pandas as pd
import numpy as np


def parse_rev_time_to_seconds(rev_time_str) -> Optional[float]:
    """
    将 rev_time 转换为从0开始的秒数
    支持格式:
      - "MM:SS.m" / "MM:SS.ms" (原始说明)
      - "HH:MM:SS.microseconds" (实际数据常见格式)
      - "HH:MM:SS" (无小数)
    """
    if pd.isna(rev_time_str):
        return None
    s = str(rev_time_str).strip()

    # 尝试匹配 HH:MM:SS.microseconds / HH:MM:SS.ms / HH:MM:SS
    m = re.match(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$", s)
    if m:
        hours = int(m.group(1))
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        frac = 0.0
        if m.group(4):
            frac_str = m.group(4)
            # .800000 -> 0.8 秒 (微秒转秒)
            # .9 -> 0.9 秒
            # 统一按秒的小数处理
            frac = float("0." + frac_str)
        return hours * 3600.0 + minutes * 60.0 + seconds + frac

    # 尝试匹配 MM:SS.m / MMM:SS.m
    m = re.match(r"^(\d+):(\d{2})(?:\.(\d+))?$", s)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        frac = 0.0
        if m.group(3):
            frac = float("0." + m.group(3))
        return minutes * 60.0 + seconds + frac

    # 回退：尝试纯秒数
    try:
        return float(s)
    except ValueError:
        return None


def load_adsb(path: str) -> pd.DataFrame:
    """读取 ADS-B 数据并标准化列名"""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    # 统一时间
    df["time_sec"] = df["rev_time"].apply(parse_rev_time_to_seconds)
    # 标准化列名
    df = df.rename(columns={
        "icao": "target_id",
        "lng": "lon",
        "lat": "lat",
        "altitude": "alt_pressure",
        "height": "alt_geom",
        "heading": "heading",
        "speed": "speed",
        "ver_speed": "climb_rate",
    })
    df["source"] = "adsb"
    df["target_type"] = "cooperative"
    # 基本清洗
    df = df.dropna(subset=["time_sec", "lat", "lon"])
    df = df.sort_values(["target_id", "time_sec"]).reset_index(drop=True)
    return df


def load_remote_id(path: str) -> pd.DataFrame:
    """读取 Remote ID 数据并标准化"""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df["time_sec"] = df["rev_time"].apply(parse_rev_time_to_seconds)
    df = df.rename(columns={
        "uas_id": "target_id",
        "longitude": "lon",
        "latitude": "lat",
        "altitude_pressure": "alt_pressure",
        "altitude_geodetic": "alt_geom",
        "height": "alt_rel",
        "direction": "heading",
        "speed_horizontal": "speed",
        "speed_vertical": "climb_rate",
    })
    df["source"] = "remote_id"
    df["target_type"] = "cooperative"
    df = df.dropna(subset=["time_sec", "lat", "lon"])
    df = df.sort_values(["target_id", "time_sec"]).reset_index(drop=True)
    return df


def load_radar(path: str) -> pd.DataFrame:
    """读取雷达航迹数据并标准化"""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df["time_sec"] = df["rev_time"].apply(parse_rev_time_to_seconds)
    # 雷达数据已包含经纬度高程，直接使用
    df = df.rename(columns={
        "track_number": "target_id",
        "longitude": "lon",
        "latitude": "lat",
        "height": "alt_geom",
        "heading_angle": "heading",
    })
    # 计算合速度
    if {"velocity_x", "velocity_y", "velocity_z"}.issubset(df.columns):
        df["speed"] = np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2)
        df["climb_rate"] = df["velocity_z"]
    else:
        df["speed"] = np.nan
        df["climb_rate"] = np.nan
    df["source"] = "radar"
    df["target_type"] = "non_cooperative"  # 默认非合作，后续关联后可修正
    df = df.dropna(subset=["time_sec", "lat", "lon"])
    df = df.sort_values(["target_id", "time_sec"]).reset_index(drop=True)
    return df


def load_spectrum(path: str) -> pd.DataFrame:
    """读取频谱检测数据并标准化"""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df["time_sec"] = df["rev_time"].apply(parse_rev_time_to_seconds)
    df = df.rename(columns={
        "uav_id": "target_id",
        "sensor_longitude": "lon",
        "sensor_latitude": "lat",
        "sensor_altitude": "alt_geom",
    })
    # 频谱数据只有传感器位置和频率、机型信息，没有目标位置
    # 这里将传感器位置作为"探测事件位置"用于时间对齐与可视化
    # 真实目标位置需要通过测向交叉定位（至少需要两个传感器）——本数据集单传感器无法定位
    # 因此频谱主要用于目标识别与辅助验证，不直接参与航迹融合
    df["source"] = "spectrum"
    df["target_type"] = "non_cooperative"
    df["speed"] = np.nan
    df["climb_rate"] = np.nan
    df["heading"] = np.nan
    df = df.dropna(subset=["time_sec"])
    df = df.sort_values(["target_id", "time_sec"]).reset_index(drop=True)
    return df


def load_all_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    一键读取全部4类数据源
    返回: {"adsb": df, "remote_id": df, "radar": df, "spectrum": df}
    """
    datasets = {}
    mapping = {
        "adsb": ("ad_adsb", load_adsb),
        "remote_id": ("ad_remote_id", load_remote_id),
        "radar": ("ad_radar", load_radar),
        "spectrum": ("ad_spectrum_detect", load_spectrum),
    }
    for key, (prefix, loader) in mapping.items():
        found = False
        for ext in [".xlsx", ".csv"]:
            path = os.path.join(data_dir, prefix + ext)
            if os.path.exists(path):
                datasets[key] = loader(path)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"未找到 {prefix}.xlsx/.csv 于 {data_dir}")
    return datasets


def compute_data_gaps(df: pd.DataFrame, max_gap_sec: float = 5.0) -> pd.DataFrame:
    """
    分析单源数据的时间缺失 gaps，返回缺失段信息
    用于评估数据质量和插值策略选择
    """
    df = df.sort_values("time_sec").copy()
    df["dt"] = df["time_sec"].diff()
    gaps = df[df["dt"] > max_gap_sec].copy()
    return gaps[["target_id", "time_sec", "dt"]]
