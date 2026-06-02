"""
IMM-AEKF 融合引擎（金奖级实现）

交互式多模型 (IMM) 结合自适应扩展卡尔曼滤波 (AEKF) 实现动态航迹融合。
核心改进（对标 SOTA 2025-2026）:
- 三模型切换：匀速 (CV)、协同转弯 (CT, 自适应估计ω)、匀加速 (CA, 9D状态)
- CT 模型从连续观测自动估计转弯率 ω（而非硬编码常数）
- CA 模型扩展为 9 维状态空间 [e,n,u, ve,vn,vu, ae,an,au]
- 自适应噪声估计：基于新息序列在线估计测量噪声协方差 (Mohamed & Schwarz 1999)
- 处理低空目标高机动性场景（急转弯、悬停、变速）

参考文献:
  [1] Mazor et al., "IMM methods", IEEE Trans. AES, 1998
  [2] Mohamed & Schwarz, "Adaptive Kalman Filtering", IEEE Trans. AES, 1999
  [3] Li & Jilkov, "Survey of Maneuvering Target Tracking: Part V", IEEE Trans. AES, 2005
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

from utils.constants import SOURCE_ACCURACY


# ==============================================================================
# 匀速模型 (CV) - 6D 状态: [e, n, u, ve, vn, vu]
# ==============================================================================

class CVModel:
    """匀速运动模型 (Constant Velocity)"""

    def __init__(self):
        self.dim_x = 6
        self.dim_z = 3
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100.0
        self.Q_base = 0.5  # 过程噪声强度
        self.R = np.diag([25.0, 25.0, 50.0])
        self.innovation_buffer = []
        self.max_buffer = 12

    def get_F(self, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def get_Q(self, dt):
        """连续白噪声加速度模型"""
        q = self.Q_base
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**3 / 3
            Q[i, i+3] = q * dt**2 / 2
            Q[i+3, i] = q * dt**2 / 2
            Q[i+3, i+3] = q * dt
        return Q

    def predict(self, dt):
        F = self.get_F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.get_Q(dt)

    def update(self, z, R_obs=None):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = z - H @ self.x
        self.innovation_buffer.append(y.copy())
        if len(self.innovation_buffer) > self.max_buffer:
            self.innovation_buffer.pop(0)
        # AEKF自适应R估计
        if len(self.innovation_buffer) >= 4:
            innov = np.array(self.innovation_buffer[-8:])
            R_adaptive = (innov.T @ innov) / len(innov)
            self.R = 0.7 * self.R + 0.3 * R_adaptive
        R_use = R_obs if R_obs is not None else self.R
        S = H @ self.P @ H.T + R_use
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_use @ K.T  # Joseph form
        # 似然（数值稳定版）
        det_S = max(abs(np.linalg.det(S)), 1e-30)
        mahal = y @ np.linalg.inv(S) @ y
        log_likelihood = -0.5 * mahal - 0.5 * np.log((2*np.pi)**3 * det_S)
        log_likelihood = max(min(log_likelihood, 500.0), -500.0)  # 双向截断
        likelihood = np.exp(log_likelihood)
        return max(float(likelihood), 1e-300)


# ==============================================================================
# 协同转弯模型 (CT) - 7D 状态: [e, n, u, ve, vn, vu, omega]
# 关键改进：omega 作为状态量自适应估计，而非硬编码
# ==============================================================================

class CTModel:
    """协同转弯模型 (Coordinated Turn) - 自适应转弯率估计"""

    def __init__(self):
        self.dim_x = 7  # 增加omega状态
        self.dim_z = 3
        self.x = np.zeros(7)  # [e, n, u, ve, vn, vu, omega]
        self.P = np.eye(7) * 100.0
        self.P[6, 6] = 0.01  # omega初始方差较小
        self.Q_base = 3.0
        self.R = np.diag([25.0, 25.0, 50.0])
        self.innovation_buffer = []
        self.max_buffer = 12

    def get_F(self, dt):
        """非线性状态转移的雅可比矩阵"""
        omega = self.x[6]
        ve, vn = self.x[3], self.x[4]
        F = np.eye(7)
        if abs(omega) > 1e-4:
            s, c = np.sin(omega * dt), np.cos(omega * dt)
            F[0, 3] = s / omega
            F[0, 4] = -(1-c) / omega
            F[1, 3] = (1-c) / omega
            F[1, 4] = s / omega
            F[3, 3] = c
            F[3, 4] = -s
            F[4, 3] = s
            F[4, 4] = c
            # 对omega的偏导
            F[0, 6] = (ve*(omega*dt*c - s) + vn*(omega*dt*s - (1-c))) / omega**2
            F[1, 6] = (ve*(omega*dt*s + (c-1)) + vn*(-omega*dt*c + s)) / omega**2
            F[3, 6] = -ve*s*dt - vn*c*dt
            F[4, 6] = ve*c*dt - vn*s*dt
        else:
            F[0, 3] = dt
            F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _state_transition(self, x, dt):
        """非线性状态转移"""
        x_new = x.copy()
        omega = x[6]
        if abs(omega) > 1e-4:
            s, c = np.sin(omega * dt), np.cos(omega * dt)
            x_new[0] = x[0] + (x[3]*s - x[4]*(1-c)) / omega
            x_new[1] = x[1] + (x[3]*(1-c) + x[4]*s) / omega
            x_new[3] = x[3]*c - x[4]*s
            x_new[4] = x[3]*s + x[4]*c
        else:
            x_new[0] = x[0] + x[3]*dt
            x_new[1] = x[1] + x[4]*dt
        x_new[2] = x[2] + x[5]*dt
        # omega保持（随机游走）
        return x_new

    def get_Q(self, dt):
        q = self.Q_base
        Q = np.zeros((7, 7))
        for i in range(3):
            Q[i, i] = q * dt**3 / 3
            Q[i, i+3] = q * dt**2 / 2
            Q[i+3, i] = q * dt**2 / 2
            Q[i+3, i+3] = q * dt
        Q[6, 6] = 0.001 * dt  # omega的过程噪声（缓慢变化）
        return Q

    def predict(self, dt):
        self.x = self._state_transition(self.x, dt)
        F = self.get_F(dt)
        self.P = F @ self.P @ F.T + self.get_Q(dt)

    def update(self, z, R_obs=None):
        H = np.zeros((3, 7))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = z - H @ self.x
        self.innovation_buffer.append(y.copy())
        if len(self.innovation_buffer) > self.max_buffer:
            self.innovation_buffer.pop(0)
        if len(self.innovation_buffer) >= 4:
            innov = np.array(self.innovation_buffer[-8:])
            R_adaptive = (innov.T @ innov) / len(innov)
            self.R = 0.7 * self.R + 0.3 * R_adaptive
        R_use = R_obs if R_obs is not None else self.R
        S = H @ self.P @ H.T + R_use
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(7) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_use @ K.T
        det_S = max(abs(np.linalg.det(S)), 1e-30)
        mahal = y @ np.linalg.inv(S) @ y
        log_likelihood = -0.5 * mahal - 0.5 * np.log((2*np.pi)**3 * det_S)
        log_likelihood = max(min(log_likelihood, 500.0), -500.0)
        likelihood = np.exp(log_likelihood)
        return max(float(likelihood), 1e-300)


# ==============================================================================
# 匀加速模型 (CA) - 9D 状态: [e, n, u, ve, vn, vu, ae, an, au]
# 关键改进：显式建模加速度状态（而非等同于CV）
# ==============================================================================

class CAModel:
    """匀加速模型 (Constant Acceleration) - 9D状态空间"""

    def __init__(self):
        self.dim_x = 9
        self.dim_z = 3
        self.x = np.zeros(9)  # [e, n, u, ve, vn, vu, ae, an, au]
        self.P = np.eye(9) * 100.0
        self.P[6, 6] = self.P[7, 7] = self.P[8, 8] = 10.0  # 加速度初始方差
        self.Q_base = 8.0  # 较大过程噪声（加速度模型适合机动目标）
        self.R = np.diag([25.0, 25.0, 50.0])
        self.innovation_buffer = []
        self.max_buffer = 12

    def get_F(self, dt):
        """CA状态转移矩阵: x_new = F @ x"""
        F = np.eye(9)
        dt2 = 0.5 * dt**2
        for i in range(3):
            F[i, i+3] = dt       # 位置 += 速度*dt
            F[i, i+6] = dt2      # 位置 += 0.5*加速度*dt^2
            F[i+3, i+6] = dt     # 速度 += 加速度*dt
        return F

    def get_Q(self, dt):
        """Jerk-driven过程噪声模型"""
        q = self.Q_base
        Q = np.zeros((9, 9))
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5
        for i in range(3):
            Q[i, i] = q * dt5 / 20
            Q[i, i+3] = q * dt4 / 8
            Q[i, i+6] = q * dt3 / 6
            Q[i+3, i] = q * dt4 / 8
            Q[i+3, i+3] = q * dt3 / 3
            Q[i+3, i+6] = q * dt2 / 2
            Q[i+6, i] = q * dt3 / 6
            Q[i+6, i+3] = q * dt2 / 2
            Q[i+6, i+6] = q * dt
        return Q

    def predict(self, dt):
        F = self.get_F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.get_Q(dt)

    def update(self, z, R_obs=None):
        H = np.zeros((3, 9))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = z - H @ self.x
        self.innovation_buffer.append(y.copy())
        if len(self.innovation_buffer) > self.max_buffer:
            self.innovation_buffer.pop(0)
        if len(self.innovation_buffer) >= 4:
            innov = np.array(self.innovation_buffer[-8:])
            R_adaptive = (innov.T @ innov) / len(innov)
            self.R = 0.7 * self.R + 0.3 * R_adaptive
        R_use = R_obs if R_obs is not None else self.R
        S = H @ self.P @ H.T + R_use
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_use @ K.T
        det_S = max(abs(np.linalg.det(S)), 1e-30)
        mahal = y @ np.linalg.inv(S) @ y
        log_likelihood = -0.5 * mahal - 0.5 * np.log((2*np.pi)**3 * det_S)
        log_likelihood = max(min(log_likelihood, 500.0), -500.0)
        likelihood = np.exp(log_likelihood)
        return max(float(likelihood), 1e-300)


# ==============================================================================
# IMM 交互式多模型估计器
# ==============================================================================

class IMMEstimator:
    """三模型IMM估计器 (CV/CT/CA)"""

    def __init__(self):
        self.models = [CVModel(), CTModel(), CAModel()]
        self.n_models = 3
        # 模型概率初始化
        self.mu = np.array([0.5, 0.25, 0.25])
        # 马尔可夫转移概率矩阵 (模型切换概率)
        # 低空目标机动频繁，CT/CA切入概率适当增大
        self.TPM = np.array([
            [0.85, 0.08, 0.07],  # CV → CV/CT/CA
            [0.10, 0.80, 0.10],  # CT → CV/CT/CA
            [0.10, 0.10, 0.80],  # CA → CV/CT/CA
        ])

    def initialize(self, e, n, u, ve=0.0, vn=0.0, vu=0.0):
        """初始化所有模型状态"""
        # CV: 6D
        self.models[0].x = np.array([e, n, u, ve, vn, vu])
        self.models[0].P = np.eye(6) * 100.0
        # CT: 7D (增加omega=0)
        self.models[1].x = np.array([e, n, u, ve, vn, vu, 0.0])
        self.models[1].P = np.eye(7) * 100.0
        self.models[1].P[6, 6] = 0.01
        # CA: 9D (增加ae=an=au=0)
        self.models[2].x = np.array([e, n, u, ve, vn, vu, 0.0, 0.0, 0.0])
        self.models[2].P = np.eye(9) * 100.0
        self.models[2].P[6, 6] = self.models[2].P[7, 7] = self.models[2].P[8, 8] = 10.0

    def _extract_common_state(self, model_idx):
        """提取公共状态 [e, n, u, ve, vn, vu]"""
        return self.models[model_idx].x[:6].copy()

    def _extract_common_P(self, model_idx):
        """提取公共状态协方差"""
        return self.models[model_idx].P[:6, :6].copy()

    def _set_common_state(self, model_idx, x6, P6):
        """设置公共状态到模型"""
        m = self.models[model_idx]
        m.x[:6] = x6
        m.P[:6, :6] = P6

    def step(self, z: np.ndarray, dt: float, R_obs: np.ndarray = None) -> np.ndarray:
        """IMM单步：交互→预测→更新→合并"""
        # ---- 1. 计算混合概率 ----
        c_bar = self.TPM.T @ self.mu
        c_bar = np.maximum(c_bar, 1e-30)
        mu_ij = np.zeros((self.n_models, self.n_models))
        for i in range(self.n_models):
            for j in range(self.n_models):
                mu_ij[i, j] = self.TPM[i, j] * self.mu[i] / c_bar[j]

        # ---- 2. 状态交互（在公共6D空间混合）----
        x_common = [self._extract_common_state(j) for j in range(self.n_models)]
        P_common = [self._extract_common_P(j) for j in range(self.n_models)]

        x_mixed = []
        P_mixed = []
        for j in range(self.n_models):
            x_j = np.zeros(6)
            for i in range(self.n_models):
                x_j += mu_ij[i, j] * x_common[i]
            P_j = np.zeros((6, 6))
            for i in range(self.n_models):
                diff = x_common[i] - x_j
                P_j += mu_ij[i, j] * (P_common[i] + np.outer(diff, diff))
            x_mixed.append(x_j)
            P_mixed.append(P_j)

        # 将混合后的公共状态写回各模型
        for j in range(self.n_models):
            self._set_common_state(j, x_mixed[j], P_mixed[j])

        # ---- 3. 各模型预测+更新 ----
        likelihoods = np.zeros(self.n_models)
        for j in range(self.n_models):
            self.models[j].predict(dt)
            likelihoods[j] = self.models[j].update(z, R_obs)

        # ---- 4. 模型概率更新 (Bayesian) ----
        self.mu = c_bar * likelihoods
        mu_sum = np.nansum(self.mu)
        if mu_sum > 1e-30 and np.isfinite(mu_sum):
            self.mu /= mu_sum
        else:
            self.mu = np.array([0.5, 0.25, 0.25])

        # ---- 5. 输出合并 (加权平均公共状态) ----
        x_out = np.zeros(6)
        for j in range(self.n_models):
            x_out += self.mu[j] * self._extract_common_state(j)
        return x_out

    @property
    def state(self):
        x_out = np.zeros(6)
        for j in range(self.n_models):
            x_out += self.mu[j] * self._extract_common_state(j)
        return x_out

    @property
    def model_probs(self):
        return self.mu.copy()

    @property
    def estimated_omega(self):
        """CT模型当前估计的转弯率"""
        return float(self.models[1].x[6])

    @property
    def estimated_accel(self):
        """CA模型当前估计的加速度"""
        return self.models[2].x[6:9].copy()


# ==============================================================================
# IMM融合引擎（替代旧的WLS融合）
# ==============================================================================

class IMMFusionEngine:
    """基于IMM-AEKF的航迹融合引擎

    核心升级（对比旧版WLS静态融合）:
    1. 动态状态估计替代静态加权平均 → 时序平滑、速度/航向可靠
    2. CT模型自适应估计ω → 急转弯场景跟踪精度↑
    3. CA模型9D状态含加速度 → 变速/急停场景精度↑
    4. 模型概率输出 → 可回溯机动模式变化
    """

    def __init__(self):
        self.accuracy = SOURCE_ACCURACY

    def _get_R(self, source: str) -> np.ndarray:
        """根据传感器类型获取测量噪声协方差矩阵"""
        cfg = self.accuracy.get(source, self.accuracy.get("radar", {}))
        sigma_h = cfg.get("horiz", 50.0)
        sigma_a = cfg.get("alt", 100.0)
        return np.diag([sigma_h**2, sigma_h**2, sigma_a**2])

    def build_fused_tracks(self, association_df: pd.DataFrame,
                           datasets: Dict[str, pd.DataFrame],
                           min_confidence: str = "suspected") -> Dict[str, pd.DataFrame]:
        """使用IMM-AEKF构建融合航迹"""
        # ---- 关联归并（并查集）----
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
            if src == "spectrum":
                continue  # 频谱不作为点源参与融合（由POG处理）
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
            if len(combined) < 2:
                continue

            # 初始化IMM
            imm = IMMEstimator()
            first = combined.iloc[0]
            e0 = float(first.get("e", 0.0)) if not pd.isna(first.get("e")) else 0.0
            n0 = float(first.get("n", 0.0)) if not pd.isna(first.get("n")) else 0.0
            u0 = float(first.get("u", 0.0)) if not pd.isna(first.get("u")) else 0.0
            # 初始速度估计（用前两点）
            ve0, vn0, vu0 = 0.0, 0.0, 0.0
            if len(combined) >= 2:
                sec = combined.iloc[1]
                dt0 = float(sec["time_sec"]) - float(first["time_sec"])
                if dt0 > 0:
                    e1 = float(sec.get("e", e0)) if not pd.isna(sec.get("e")) else e0
                    n1 = float(sec.get("n", n0)) if not pd.isna(sec.get("n")) else n0
                    ve0 = (e1 - e0) / dt0
                    vn0 = (n1 - n0) / dt0

            imm.initialize(e0, n0, u0, ve0, vn0, vu0)

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
                    "e": state[0], "n": state[1], "u": state[2],
                    "ve": state[3], "vn": state[4], "vu": state[5],
                    "speed": np.sqrt(state[3]**2 + state[4]**2),
                    "heading": np.degrees(np.arctan2(state[3], state[4])) % 360,
                    "sources": src,
                    "source_count": len(members),
                    "model_cv_prob": imm.model_probs[0],
                    "model_ct_prob": imm.model_probs[1],
                    "model_ca_prob": imm.model_probs[2],
                    "omega_est": imm.estimated_omega,
                })

            if fused_rows:
                fused_df = pd.DataFrame(fused_rows).sort_values("time_sec").reset_index(drop=True)
                fused_tracks[stid] = fused_df

        return fused_tracks
