# 低空目标多元融合监视系统

> **多源异构数据融合 · IMM-AEKF动态滤波 · D-S证据理论 · 频谱POG网格 · 复杂网络拓扑**

---

## 项目简介

本系统面向低空空域管理需求，将 **ADS-B、Remote ID、低空雷达、频谱检测** 四类异构传感器数据进行深度融合，实现低空目标的"看得见、辨得清、管得住"。系统采用十步流水线架构，完成从原始数据到融合航迹、态势感知、空域分析的完整闭环。

**在线演示**：[GitHub Pages 可视化平台](https://junjieyang651-blip.github.io/low-altitude-surveillance/)

---

## 核心技术亮点

| 技术模块 | 方法 | 核心创新 |
|---------|------|---------|
| **IMM-AEKF融合** | CV-6D / CT-7D / CA-9D 异维状态空间切换 | 转弯率ω作为第7维状态自适应估计，RMSE改进60-84% |
| **D-S证据理论** | Sigmoid映射BPA + K冲突系数自动切换 | 低冲突用Dempster规则，高冲突用Murphy平均证据法 |
| **频谱POG** | Log-Odds累积 + 多站交叉定位 | 四级虚警过滤（单站孤立/频率异常/多径闪烁/时空不一致） |
| **网络拓扑** | 标签传播社区检测 + 单点失效脆弱性 | 动态容量阈值(DCT) + 拥堵指数(CI)量化空域负载 |
| **航迹维持** | 位置预测门控 + 特征匹配 | 处理雷达ID跳变导致的航迹断裂 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        十步流水线架构                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ① 数据加载 → ② 时空对齐(WGS84→ENU) → ③ 航迹关联(粗筛+细匹配)     │
│       ↓                                                             │
│  ④ IMM-AEKF动态融合 → ⑤ 目标分类 → ⑥ 双引擎异常检测               │
│       ↓                                                             │
│  ⑦ 冲突预警(DCPA/TCPA) + 流量分析                                  │
│       ↓                                                             │
│  ⑧ 频谱POG空间建模 → ⑨ D-S证据可信度评估 → ⑩ 复杂网络拓扑分析     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 实验结果

### 数据规模

| 数据源 | 记录数 | 目标数 | 特点 |
|--------|--------|--------|------|
| ADS-B | 12,667 | 17 | 合作式广播，位置精度~30m |
| Remote ID | 112,909 | 502 | 无人机远程识别 |
| 低空雷达 | 32,263 | 500 | 非合作式探测 |
| 频谱检测 | 155,953 | 2,657 | 仅存在性检测，无精确位置 |
| **合计** | **313,792** | **3,676** | 覆盖时间59分钟 |

### 融合精度（RMSE量化对比）

| 数据源 | 单源观测RMSE | 融合后RMSE | 改进率 |
|--------|-------------|-----------|--------|
| ADS-B | 36,447 m | 14,778 m | **60%** |
| Remote ID | 30,899 m | 14,778 m | **52%** |
| 雷达 | 90,915 m | 14,778 m | **84%** |

> 融合后轨迹不确定性显著低于任何单一数据源，雷达改进最大(84%)

### 核心指标

| 指标 | 数值 |
|------|------|
| 融合航迹数 | 455 条 |
| 峰值活跃目标 | 714 个 |
| 航迹维持重关联 | 22对断裂 / 3确认 |
| 异常事件 | 59,866 条 |
| 冲突预警 | 6,380 条 |
| 航线走廊 | 73 条 |
| 虚警识别 | 频率异常21,550次 / 单站孤立19,121次 |

### 网络拓扑分析

| 指标 | 数值 |
|------|------|
| 网络节点 | 1,418 |
| 网络边 | 4,015 |
| 社区数 | 171 |
| 全局拥堵指数 | 0.030 |
| 拥堵节点(CI>1) | 4 个 |
| 最高CI节点 | G+053_+026 (CI=4.764) |

---

## 项目结构

```
├── main.py                  # 主流水线（十步）
├── core/                    # 核心算法模块
│   ├── time_align.py        # 时空对齐与ENU坐标转换
│   ├── association.py       # 粗筛+细匹配航迹关联 + 航迹维持
│   ├── imm_fusion.py       # ★ IMM-AEKF动态融合（CV/CT/CA异维）
│   ├── ds_evidence.py      # ★ D-S证据理论冲突处理
│   ├── spectrum_pog.py     # ★ 频谱概率占据网格 + 虚警过滤
│   ├── network_topology.py # ★ 复杂网络拓扑 + 空域容量分析
│   ├── classification.py    # 规则引擎目标分类
│   ├── anomaly_detection.py # 双引擎异常检测
│   ├── conflict_detection.py# DCPA/TCPA冲突预警
│   └── flow_analysis.py     # 时域流量与走廊识别
├── utils/                   # 工具函数
│   ├── coords.py            # WGS84↔ENU坐标转换
│   └── io_utils.py          # 四源异构数据加载
├── web/                     # Web可视化平台
│   ├── app.py               # Flask服务
│   └── templates/index.html # Leaflet+ECharts前端
├── docs/                    # GitHub Pages静态站点
│   ├── index.html           # 可视化首页
│   └── data/                # 静态JSON数据
├── shuju/                   # 原始数据（4个xlsx）
├── output/                  # 输出结果
│   ├── figures/             # 论文配图
│   ├── final/              # 最终成果汇总
│   └── *.csv / *.json       # 中间结果
├── export_static.py         # 静态数据导出（GitHub Pages）
├── gen_figures.py           # 配图生成
└── export_results.py        # 成果导出
```

---

## 快速开始

### 环境要求

- Python 3.12+
- 依赖：`pip install -r requirements.txt`

### 运行主流水线

```bash
python main.py --data ./shuju --output ./output
```

### 启动Web可视化

```bash
python web/app.py --port 5000
# 浏览器访问 http://127.0.0.1:5000
```

### 导出GitHub Pages数据

```bash
python export_static.py
```

---

## 技术栈

- **语言**：Python 3.14
- **数据处理**：Pandas, NumPy, SciPy
- **网络分析**：NetworkX
- **可视化**：Flask, Leaflet.js, ECharts
- **坐标系统**：WGS84 ↔ ENU（东北天局部坐标）

---

## 与传统方法对比

| 环节 | 传统方法 | 本系统方法 | 优势 |
|------|---------|-----------|------|
| 航迹融合 | WLS加权最小二乘 | IMM-AEKF动态滤波 | 适应转弯/加速等机动，RMSE降低60-84% |
| 冲突处理 | 并查集简单合并 | D-S证据理论 | 量化冲突，识别虚假目标 |
| 频谱处理 | 简单赋权0.3做点融合 | POG网格+多站交叉定位+虚警过滤 | 符合频谱物理特性，过滤40,000+虚警 |
| 流量分析 | 网格计数 | 复杂网络+社区检测+拥堵指数 | 发现结构性风险，量化容量瓶颈 |

---

## 参考文献

1. Mazor E, et al. *Interacting multiple model methods in target tracking: a survey*. IEEE TAES, 1998.
2. Mohamed A H, Schwarz K P. *Adaptive Kalman filtering for INS/GPS*. Journal of Geodesy, 1999.
3. Shafer G. *A Mathematical Theory of Evidence*. Princeton University Press, 1976.
4. Murphy C K. *Combining belief functions when evidence conflicts*. Decision Support Systems, 2000.
5. Elfes A. *Using occupancy grids for mobile robot perception and navigation*. Computer, 1989.
6. Newman M E J. *Networks: An Introduction*. Oxford University Press, 2010.
7. Raghavan U N, et al. *Near linear time algorithm to detect community structures*. Physical Review E, 2007.
