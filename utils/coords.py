"""
坐标转换工具集
支持：WGS84 ↔ ENU(东北天) 局部坐标系、极坐标辅助运算
所有计算使用 double 精度，满足低空监视精度需求
向量化支持 numpy 数组输入
"""

import math
from typing import Tuple, List
import numpy as np
from utils.constants import EARTH_RADIUS_M


def wgs84_to_enu(lat, lon, alt,
                 ref_lat: float, ref_lon: float, ref_alt: float):
    """
    将 WGS84 (lat, lon, alt) 转换为以 ref 点为原点的 ENU 局部坐标系 (m)
    E: East (东), N: North (北), U: Up (天/上)
    支持标量或 numpy 数组输入
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)

    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    dalt = alt - ref_alt

    lat_rad = math.radians(ref_lat)

    # 局部子午线曲率半径
    R_n = EARTH_RADIUS_M

    north = dlat * R_n
    east = dlon * R_n * math.cos(lat_rad)
    up = dalt

    # 保持输入维度：标量输入返回标量，数组输入返回数组
    if lat.ndim == 0:
        return float(east), float(north), float(up)
    return east, north, up


def enu_to_wgs84(e, n, u,
                 ref_lat: float, ref_lon: float, ref_alt: float):
    """
    ENU → WGS84 反变换
    支持标量或 numpy 数组输入
    """
    e = np.asarray(e, dtype=float)
    n = np.asarray(n, dtype=float)
    u = np.asarray(u, dtype=float)

    lat_rad = math.radians(ref_lat)
    R_n = EARTH_RADIUS_M

    dlat = n / R_n
    dlon = e / (R_n * math.cos(lat_rad))

    lat = ref_lat + np.degrees(dlat)
    lon = ref_lon + np.degrees(dlon)
    alt = ref_alt + u

    if e.ndim == 0:
        return float(lat), float(lon), float(alt)
    return lat, lon, alt


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    基于 haversine 公式的两经纬度点间距离 (m)
    适用于大尺度粗略估计，精确计算请用 ENU 转换后求模
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def spherical_to_cartesian(r: float, azimuth_deg: float, elevation_deg: float) -> Tuple[float, float, float]:
    """
    雷达极坐标 (斜距, 方位角, 俯仰角) → 局部 ENU 直角坐标
    方位角：正北为0，顺时针增大
    俯仰角：水平面为0，向上为正
    返回: (east, north, up)
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # 局部坐标系：x-East, y-North, z-Up
    # 斜距在水平面投影 = r * cos(el)
    # north = proj * cos(az)
    # east  = proj * sin(az)
    # up    = r * sin(el)
    proj = r * math.cos(el)
    north = proj * math.cos(az)
    east = proj * math.sin(az)
    up = r * math.sin(el)

    return east, north, up


def velocity_to_enu(speed_ms: float, heading_deg: float, climb_rate_ms: float = 0.0) -> Tuple[float, float, float]:
    """
    将速度大小、航向、爬升率 转换为 ENU 速度矢量
    heading: 正北为0，顺时针增大
    返回: (ve, vn, vu)
    """
    h = math.radians(heading_deg)
    vn = speed_ms * math.cos(h)
    ve = speed_ms * math.sin(h)
    vu = climb_rate_ms
    return ve, vn, vu


def heading_from_velocity(ve: float, vn: float) -> float:
    """
    由 ENU 速度分量反算航向角 (0-360)
    """
    h = math.degrees(math.atan2(ve, vn))
    if h < 0:
        h += 360.0
    return h


def compute_centroid(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    计算点集的几何中心 (lat, lon, alt)
    用于确定局部 ENU 坐标系原点
    """
    if not points:
        return 0.0, 0.0, 0.0
    n = len(points)
    avg_lat = sum(p[0] for p in points) / n
    avg_lon = sum(p[1] for p in points) / n
    avg_alt = sum(p[2] for p in points) / n
    return avg_lat, avg_lon, avg_alt
