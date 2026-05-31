"""
物理常数、阈值配置与机型动力学约束
所有参数集中管理，便于调参与工程验证
"""

import math

# ---------------------------
# 地球物理常数
# ---------------------------
EARTH_RADIUS_M = 6371000.0  # 地球平均半径 (m)

# ---------------------------
# 各源定位精度先验（用于融合加权，单位：m）
# 数值越小表示精度越高，权重越大
# ---------------------------
SOURCE_ACCURACY = {
    "adsb": {"horiz": 10.0, "alt": 15.0, "vel": 1.0},      # ADS-B GNSS精度
    "remote_id": {"horiz": 30.0, "alt": 30.0, "vel": 2.0}, # Remote ID 一般精度
    "radar": {"horiz": 50.0, "alt": 80.0, "vel": 3.0},     # 低空雷达精度
    "spectrum": {"horiz": 200.0, "alt": 500.0, "vel": 10.0}, # 频谱仅RF测向，位置很粗
}

# ---------------------------
# 关联算法阈值
# ---------------------------
ASSOCIATION = {
    "max_spatial_distance_m": 800.0,      # 最大空间距离门限（进一步放宽，传感器定位误差）
    "max_speed_diff_ms": 50.0,            # 最大速度差门限（大幅放宽，允许机动差异）
    "max_heading_diff_deg": 90.0,         # 最大航向差门限（大幅放宽，考虑盘旋等机动）
    "max_altitude_diff_m": 500.0,         # 最大高度差门限（放宽，传感器高度误差）
    "adaptive_window_factor": 5.0,        # 自适应窗口系数 k: Δt = k / v
    "min_window_sec": 120.0,              # 最小时间窗（大幅放宽，允许跨传感器时间差异）
    "max_window_sec": 180.0,              # 最大时间窗
    "mahalanobis_threshold": 9.21,        # 马氏距离门限（对应95%置信椭圆，chi2(2)）
    "confirm_frames": 3,                  # 连续确认帧数才升为"确定关联"
}

# ---------------------------
# 冲突检测阈值
# ---------------------------
CONFLICT = {
    "dcpa_warning_m": 500.0,              # DCPA 警告阈值
    "dcpa_critical_m": 200.0,             # DCPA 紧急阈值
    "tcpa_warning_sec": 60.0,             # TCPA 警告阈值
    "tcpa_critical_sec": 20.0,            # TCPA 紧急阈值
    "potential_radius_m": 2000.0,         # 势场作用半径
    "potential_k_speed": 2.0,             # 速度方向修正系数
}

# ---------------------------
# 异常检测：机型动力学约束（硬物理极限）
# 超出即视为"不可能运动"，100%异常置信度
# ---------------------------
AIRCRAFT_LIMITS = {
    # 通航飞机 (ADS-B 典型)
    "ga": {
        "max_speed_ms": 150.0,
        "min_speed_ms": 30.0,
        "max_climb_rate_ms": 15.0,
        "max_descent_rate_ms": 20.0,
        "max_turn_rate_degs": 15.0,
        "max_altitude_m": 6000.0,
    },
    # 消费级无人机 (Remote ID 典型)
    "uav_consumer": {
        "max_speed_ms": 25.0,
        "min_speed_ms": 0.0,
        "max_climb_rate_ms": 8.0,
        "max_descent_rate_ms": 6.0,
        "max_turn_rate_degs": 60.0,
        "max_altitude_m": 500.0,
    },
    # 行业级无人机
    "uav_industrial": {
        "max_speed_ms": 40.0,
        "min_speed_ms": 0.0,
        "max_climb_rate_ms": 10.0,
        "max_descent_rate_ms": 8.0,
        "max_turn_rate_degs": 45.0,
        "max_altitude_m": 1000.0,
    },
    # 未知/非合作目标（雷达/频谱）—— 采用最宽松约束
    "unknown": {
        "max_speed_ms": 300.0,
        "min_speed_ms": 0.0,
        "max_climb_rate_ms": 50.0,
        "max_descent_rate_ms": 50.0,
        "max_turn_rate_degs": 180.0,
        "max_altitude_m": 10000.0,
    },
}

# ---------------------------
# 分类规则阈值（轻量可解释）
# ---------------------------
CLASSIFICATION = {
    "uav_altitude_threshold_m": 600.0,    # 低于此高度倾向无人机
    "uav_speed_threshold_ms": 50.0,       # 低于此速度倾向无人机
    "spectrum_confidence": 0.95,          # 频谱直接识别的置信度
}

# ---------------------------
# 地理围栏（示例：可扩展为多个多边形）
# ---------------------------
NO_FLY_ZONES = [
    # (中心经度, 中心纬度, 半径m, 最低高度m, 最高高度m, 名称)
    (113.8, 22.6, 3000.0, 0.0, 1000.0, "机场净空区-A"),
]


def get_category_by_limits(speed, altitude, climb_rate):
    """
    根据运动特征反推最可能的机型类别，用于选择对应动力学约束
    返回: 'ga' | 'uav_consumer' | 'uav_industrial' | 'unknown'
    """
    if altitude > 1000.0 or speed > 80.0:
        return "ga"
    if speed < 25.0 and altitude < 300.0:
        return "uav_consumer"
    if altitude < 800.0:
        return "uav_industrial"
    return "unknown"
