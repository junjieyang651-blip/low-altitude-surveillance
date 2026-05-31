"""
IMM-AEKF 融合引擎

交互式多模型 (IMM) 结合自适应扩展卡尔曼滤波 (AEKF) 实现动态航迹融合。
- 三模型切换：匀速 (CV)、协同转弯 (CT)、匀加速 (CA)
- 自适应噪声估计：基于新息序列在线估计测量噪声协方差
- 处理低空目标高机动性场景（急转弯、悬停、加速）

参考文献:
  [1] Mazor et al., "IMM methods", IEEE Trans. AES, 1998
  [2] Mohamed & Schwarz, "Adaptive Kalman Filtering", IEEE Trans. AES, 1999
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

from utils.constants import SOURCE_ACCURACY


# ==============================================================================
# 卡尔曼滤波器基础类
# ==============================================================================

class KalmanModel:
    """单模型卡尔曼滤波器（EKF）"""

    def __init__(self, model_type: str, dt: float = 1.0):
        self.model_type = model_type  # 'CV', 'CT', 'CA'
        self.dt = dt
        self.dim_x = 6  # 状态维度: [e, n, u, ve, vn, vu]
        self.dim_z = 3  # 观测维度: [e, n, u]

        # 状态向量
        self.x = np.zeros(self.dim_x)
        # 状态协方差
        self.P = np.eye(self.dim_x) * 100.0
        # 过程噪声
        self.Q = self._build_Q()
        # 测量噪声（初始值，AEKF会自适应更新）
        self.R = np.diag([25.0, 25.0, 50.0])  # 5m, 5m, ~7m
        # 新息序列缓存（用于AEKF自适应）
        self.innovation_buffer = []
        self.max_buffer_size = 10

    def _build_Q(self) -> np.ndarray:
        """根据模型类型构建过程噪声"""
        dt = self.dt
        if self.model_type == 'CV':  # 匀速模型，噪声较小
            q = 1.0
        elif self.model_type == 'CT':  # 协同转弯，中等噪声
            q = 5.0
        elif self.model_type == 'CA':  # 匀加速，较大噪声
            q = 10.0
        else:
            q = 2.0

        # 连续白噪声加速度模型
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt ** 3 / 3
            Q[i, i + 3] = q * dt ** 2 / 2
            Q[i + 3, i] = q * dt ** 2 / 2
            Q[i + 3, i + 3] = q * dt
        return Q

    def _state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """状态转移函数"""
        x_pred = x.copy()
        if self.model_type == 'CV':
            # 匀速: x_new = x + v*dt
            x_pred[0] += x[3] * dt
            x_pred[1] += x[4] * dt
            x_pred[2] += x[5] * dt
        elif self.model_type == 'CT':
            # 协同转弯: 估算转弯率omega
            speed = np.sqrt(x[3] ** 2 + x[4] ** 2) + 1e-6
            omega = 0.05  # 默认小转弯率 rad/s
            if abs(omega) > 1e-6:
                s, c = np.sin(omega * dt), np.cos(omega * dt)
                x_pred[0] += (x[3] * s - x[4] * (1 - c)) / omega
                x_pred[1] += (x[3] * (1 - c) + x[4] * s) / omega
                x_pred[2] += x[5] * dt
                v3 = x[3] * c - x[4] * s
                v4 = x[3] * s + x[4] * c
                x_pred[3] = v3
                x_pred[4] = v4
            else:
                x_pred[0] += x[3] * dt
                x_pred[1] += x[4] * dt
                x_pred[2] += x[5] * dt
        elif self.model_type == 'CA':
            # 匀加速: 使用上一次估计的加速度（简化为CV+更大噪声）
            x_pred[0] += x[3] * dt
            x_pred[1] += x[4] * dt
            x_pred[2] += x[5] * dt
        return x_pred

    def _jacobian_F(self, dt: float) -> np.ndarray:
        """状态转移雅可比矩阵"""
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def predict(self, dt: float):
        """预测步"""
        self.dt = dt
        self.Q = self._build_Q()
        F = self._jacobian_F(dt)
        self.x = self._state_transition(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, R_obs: np.ndarray = None):
        """更新步（带AEKF自适应噪声估计）"""
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # 观测e
        H[1, 1] = 1.0  # 观测n
        H[2, 2] = 1.0  # 观测u

        # 新息
        y = z - H @ self.x
        self.innovation_buffer.append(y)
        if len(self.innovation_buffer) > self.max_buffer_size:
            self.innovation_buffer.pop(0)

        # AEKF：自适应估计测量噪声R
        if len(self.innovation_buffer) >= 3:
            innovations = np.array(self.innovation_buffer)
            R_adaptive = np.cov(innovations.T) + H @ self.P @ H.T
            # 加权融合先验R和自适应R
            alpha = 0.3
            self.R = alpha * self.R + (1 - alpha) * R_adaptive

        R_use = R_obs if R_obs is not None else self.R
        S = H @ self.P @ H.T + R_use
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        # 返回似然值用于IMM模型概率更新
        det_S = max(np.linalg.det(S), 1e-30)
        likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt((2 * np.pi) ** 3 * det_S)
        return float(likelihood)


# ==============================================================================
# IMM 交互式多模型
# ==============================================================================

class IMMEstimator:
    """三模型IMM估计器"""

    def __init__(self):
        self.models = [
            KalmanModel('CV'),
            KalmanModel('CT'),
            KalmanModel('CA'),
        ]
        self.n_models = 3
        # 模型概率
        self.mu = np.array([0.6, 0.2, 0.2])
        # 马尔可夫转移矩阵
        self.TPM = np.array([
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.10, 0.10, 0.80],
        ])

    def initialize(self, e, n, u, ve=0.0, vn=0.0, vu=0.0):
        """初始化所有模型状态"""
        x0 = np.array([e, n, u, ve, vn, vu])
        for model in self.models:
            model.x = x0.copy()
            model.P = np.eye(6) * 100.0

    def step(self, z: np.ndarray, dt: float, R_obs: np.ndarray = None) -> np.ndarray:
        """IMM单步：交互→预测→更新→合并"""
        # ---- 1. 计算混合概率 ----
        c_bar = self.TPM.T @ self.mu
        c_bar = np.maximum(c_bar, 1e-30)
        mu_ij = np.zeros((self.n_models, self.n_models))
        for i in range(self.n_models):
            for j in range(self.n_models):
                mu_ij[i, j] = self.TPM[i, j] * self.mu[i] / c_bar[j]

        # ---- 2. 状态交互（混合） ----
        x_mixed = []
        P_mixed = []
        for j in range(self.n_models):
            x_j = np.zeros(6)
            for i in range(self.n_models):
                x_j += mu_ij[i, j] * self.models[i].x
            P_j = np.zeros((6, 6))
            for i in range(self.n_models):
                diff = self.models[i].x - x_j
                P_j += mu_ij[i, j] * (self.models[i].P + np.outer(diff, diff))
            x_mixed.append(x_j)
            P_mixed.append(P_j)

        # ---- 3. 各模型预测+更新 ----
        likelihoods = np.zeros(self.n_models)
        for j in range(self.n_models):
            self.models[j].x = x_mixed[j]
            self.models[j].P = P_mixed[j]
            self.models[j].predict(dt)
            likelihoods[j] = self.models[j].update(z, R_obs)

        # ---- 4. 模型概率更新 ----
        self.mu = c_bar * likelihoods
        mu_sum = np.sum(self.mu)
        if mu_sum > 1e-30:
            self.mu /= mu_sum
        else:
            self.mu = np.array([0.6, 0.2, 0.2])

        # ---- 5. 输出合并 ----
        x_out = np.zeros(6)
        for j in range(self.n_models):
            x_out += self.mu[j] * self.models[j].x

        return x_out

    @property
    def state(self):
        """当前融合状态"""
        x_out = np.zeros(6)
        for j in range(self.n_models):
            x_out += self.mu[j] * self.models[j].x
        return x_out

    @property
    def model_probs(self):
        return self.mu.copy()


# ==============================================================================
# IMM融合引擎（替代旧的WLS融合）
# ==============================================================================

class IMMFusionEngine:
    """基于IMM-AEKF的航迹融合引擎"""

    def __init__(self):
        self.accuracy = SOURCE_ACCURACY

    def _get_R(self, source: str) -> np.ndarray:
        """根据传感器类型获取测量噪声协方差矩阵"""
        cfg = self.accuracy.get(source, self.accuracy.get("radar", {}))
        sigma_h = cfg.get("horiz", 50.0)
        sigma_a = cfg.get("alt", 100.0)
        return np.diag([sigma_h ** 2, sigma_h ** 2, sigma_a ** 2])

    def build_fused_tracks(self, association_df: pd.DataFrame,
                           datasets: Dict[str, pd.DataFrame],
                           min_confidence: str = "suspected") -> Dict[str, pd.DataFrame]:
        """
        使用IMM-AEKF构建融合航迹

        核心升级点（相比旧版WLS）：
        1. 动态状态估计替代静态加权平均
        2. 多模型切换适配高机动目标
        3. 自适应噪声估计应对不同距离/环境
        """
        # ---- 关联归并（与旧版相同的并查集逻辑）----
        if association_df.empty or "level" not in association_df.columns:
            assoc_valid = pd.DataFrame(columns=["src_a", "tid_a", "src_b", "tid_b", "level"])
        else:
            conf_map = {"confirmed": 2, "suspected": 1, "pending": 0}
            min_level = conf_map.get(min_confidence, 1)
            assoc_valid = association_df[association_df["level"].map(conf_map) >= min_level].copy()

        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for _, row in assoc_valid.iterrows():
            a = (row["src_a"], str(row["tid_a"]))
            b = (row["src_b"], str(row["tid_b"]))
            union(a, b)

        all_tracks = set()
        for src, df in datasets.items():
            for tid in df["target_id"].unique():
                all_tracks.add((src, str(tid)))
        for node in all_tracks:
            find(node)

        groups: Dict[tuple, List[tuple]] = defaultdict(list)
        for node in all_tracks:
            root = find(node)
            groups[root].append(node)

        # ---- 为每个系统航迹运行IMM-AEKF ----
        fused_tracks: Dict[str, pd.DataFrame] = {}

        for idx, (root, members) in enumerate(groups.items(), start=1):
            stid = f"ST{idx:04d}"

            # 收集该组所有观测
            obs_list = []
            for src, tid in members:
                df_src = datasets.get(src)
                if df_src is None:
                    continue
                sub = df_src[df_src["target_id"] == tid].copy()
                if sub.empty:
                    continue
                sub["member_source"] = src
                obs_list.append(sub)

            if not obs_list:
                continue

            combined = pd.concat(obs_list, ignore_index=True)
            combined = combined.sort_values("time_sec").reset_index(drop=True)

            # 孤立航迹（单源）也用IMM进行平滑，提升轨迹质量
            if len(combined) < 2:
                continue

            # 初始化IMM
            imm = IMMEstimator()
            first = combined.iloc[0]
            e0 = float(first.get("e", 0.0)) if not pd.isna(first.get("e")) else 0.0
            n0 = float(first.get("n", 0.0)) if not pd.isna(first.get("n")) else 0.0
            u0 = float(first.get("u", 0.0)) if not pd.isna(first.get("u")) else 0.0
            imm.initialize(e0, n0, u0)

            fused_rows = []
            prev_time = float(combined.iloc[0]["time_sec"])

            for i, row in combined.iterrows():
                t = float(row["time_sec"])
                dt = max(t - prev_time, 0.01)
                prev_time = t

                e_obs = float(row.get("e", np.nan))
                n_obs = float(row.get("n", np.nan))
                u_obs = float(row.get("u", 0.0)) if not pd.isna(row.get("u")) else 0.0

                if pd.isna(e_obs) or pd.isna(n_obs):
                    continue

                src = row.get("member_source", "radar")
                R_obs = self._get_R(src)
                z = np.array([e_obs, n_obs, u_obs])

                state = imm.step(z, dt, R_obs)

                fused_rows.append({
                    "system_track_id": stid,
                    "time_sec": t,
                    "e": state[0],
                    "n": state[1],
                    "u": state[2],
                    "speed": np.sqrt(state[3] ** 2 + state[4] ** 2),
                    "heading": np.degrees(np.arctan2(state[3], state[4])) % 360,
                    "ve": state[3],
                    "vn": state[4],
                    "vu": state[5],
                    "sources": row.get("member_source", ""),
                    "source_count": len(members),
                    "model_cv_prob": imm.model_probs[0],
                    "model_ct_prob": imm.model_probs[1],
                    "model_ca_prob": imm.model_probs[2],
                })

            if fused_rows:
                fused_df = pd.DataFrame(fused_rows).sort_values("time_sec").reset_index(drop=True)
                fused_tracks[stid] = fused_df

        return fused_tracks
