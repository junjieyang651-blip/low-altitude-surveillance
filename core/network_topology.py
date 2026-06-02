"""
低空复杂网络拓扑分析

将低空交通流建模为有向加权图，通过复杂网络分析方法识别关键节点和结构特征。
核心功能：
1. 构建低空交通网络图（节点=空间网格/航路点，边=航迹连接）
2. 计算节点度中心度 (Degree Centrality)
3. 计算介数中心性 (Betweenness Centrality) —— 识别核心风险节点
4. 网络社区检测 —— 识别独立航路群
5. 网络脆弱性分析 —— 评估单点失效影响

参考文献:
  [1] Newman, "Networks: An Introduction", Oxford, 2010
  [2] Barrat et al., "The architecture of complex weighted networks", 2004
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import heapq


class AirspaceNetworkGraph:
    """低空交通网络图"""

    def __init__(self, grid_size_m: float = 500.0):
        """
        Args:
            grid_size_m: 空间网格尺寸 (m)，每个网格中心为一个网络节点
        """
        self.grid_size = grid_size_m
        # 邻接表: node_id -> {neighbor_id: weight}
        self.adjacency: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # 节点属性
        self.node_attrs: Dict[str, Dict] = {}
        # 节点通过目标数
        self.node_traffic: Dict[str, int] = defaultdict(int)

    def _enu_to_node_id(self, e: float, n: float) -> str:
        """ENU坐标→节点ID"""
        gi = int(round(e / self.grid_size))
        gj = int(round(n / self.grid_size))
        return f"G{gi:+04d}_{gj:+04d}"

    def _node_id_to_enu(self, node_id: str) -> Tuple[float, float]:
        """节点ID→ENU中心坐标"""
        parts = node_id[1:].split("_")
        gi = int(parts[0])
        gj = int(parts[1])
        return gi * self.grid_size, gj * self.grid_size

    def build_from_tracks(self, fused_tracks: Dict[str, pd.DataFrame]):
        """
        从融合航迹构建网络

        每条航迹按时间顺序经过的网格节点形成有向边。
        边权重 = 经过该边的航迹数量。
        """
        for stid, track_df in fused_tracks.items():
            if "e" not in track_df.columns or "n" not in track_df.columns:
                continue

            sorted_df = track_df.sort_values("time_sec")
            prev_node = None

            for _, row in sorted_df.iterrows():
                e, n = row.get("e", np.nan), row.get("n", np.nan)
                if pd.isna(e) or pd.isna(n):
                    continue

                node_id = self._enu_to_node_id(e, n)
                self.node_traffic[node_id] += 1

                # 记录节点属性
                if node_id not in self.node_attrs:
                    self.node_attrs[node_id] = {
                        "e": e, "n": n,
                        "first_seen": row.get("time_sec", 0),
                        "tracks_passing": set()
                    }
                self.node_attrs[node_id]["tracks_passing"].add(stid)

                # 构建有向边
                if prev_node and prev_node != node_id:
                    self.adjacency[prev_node][node_id] += 1.0
                    # 确保反向节点也在图中
                    if node_id not in self.adjacency:
                        self.adjacency[node_id] = defaultdict(float)

                prev_node = node_id

    @property
    def nodes(self) -> List[str]:
        """所有节点"""
        all_nodes = set(self.adjacency.keys())
        for neighbors in self.adjacency.values():
            all_nodes.update(neighbors.keys())
        return list(all_nodes)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return sum(len(nb) for nb in self.adjacency.values())

    def degree_centrality(self) -> Dict[str, float]:
        """
        度中心度 (Degree Centrality)

        DC(v) = (入度 + 出度) / (2 * (N-1))
        衡量节点的直接连接数
        """
        nodes = self.nodes
        N = len(nodes)
        if N <= 1:
            return {n: 0.0 for n in nodes}

        out_degree = defaultdict(int)
        in_degree = defaultdict(int)

        for src, neighbors in self.adjacency.items():
            out_degree[src] += len(neighbors)
            for dst in neighbors:
                in_degree[dst] += 1

        centrality = {}
        for node in nodes:
            total_degree = out_degree.get(node, 0) + in_degree.get(node, 0)
            centrality[node] = total_degree / (2.0 * (N - 1)) if N > 1 else 0.0

        return centrality

    def weighted_degree_centrality(self) -> Dict[str, float]:
        """
        加权度中心度 (Weighted Degree / Strength)

        考虑边权重（通过流量）的节点重要性
        """
        strength = defaultdict(float)

        for src, neighbors in self.adjacency.items():
            for dst, weight in neighbors.items():
                strength[src] += weight
                strength[dst] += weight

        max_s = max(strength.values()) if strength else 1.0
        return {node: s / max_s for node, s in strength.items()}

    def betweenness_centrality(self, sample_size: int = 200) -> Dict[str, float]:
        """
        介数中心性 (Betweenness Centrality)

        BC(v) = Σ(σ_st(v) / σ_st) , s≠v≠t

        识别处于多条最短路径上的"桥梁"节点 → 核心风险节点

        Args:
            sample_size: 采样节点数（大图优化）
        """
        nodes = self.nodes
        N = len(nodes)
        if N <= 2:
            return {n: 0.0 for n in nodes}

        betweenness = defaultdict(float)

        # 对于大图，采样部分源节点
        source_nodes = nodes[:min(sample_size, N)]

        for s in source_nodes:
            # Dijkstra (BFS on unweighted) 计算最短路径
            dist = {s: 0.0}
            paths_count = {s: 1}
            predecessors = defaultdict(list)
            visited_order = []
            heap = [(0.0, s)]

            while heap:
                d, u = heapq.heappop(heap)
                if d > dist.get(u, float('inf')):
                    continue
                visited_order.append(u)

                for v, w in self.adjacency.get(u, {}).items():
                    new_dist = d + (1.0 / (w + 0.1))  # 权重越大→距离越短
                    if v not in dist or new_dist < dist[v]:
                        dist[v] = new_dist
                        paths_count[v] = paths_count[u]
                        predecessors[v] = [u]
                        heapq.heappush(heap, (new_dist, v))
                    elif abs(new_dist - dist.get(v, float('inf'))) < 1e-10:
                        paths_count[v] += paths_count[u]
                        predecessors[v].append(u)

            # 反向累积介数
            delta = defaultdict(float)
            for v in reversed(visited_order):
                for u in predecessors.get(v, []):
                    if paths_count.get(v, 0) > 0:
                        frac = paths_count[u] / paths_count[v]
                        delta[u] += frac * (1 + delta[v])
                if v != s:
                    betweenness[v] += delta[v]

        # 归一化
        scale = 1.0 / ((N - 1) * (N - 2)) if N > 2 else 1.0
        # 采样修正
        if len(source_nodes) < N:
            scale *= N / len(source_nodes)

        return {node: betweenness.get(node, 0.0) * scale for node in nodes}

    def identify_critical_nodes(self, top_k: int = 10) -> List[Dict]:
        """
        识别核心风险节点

        综合度中心度、介数中心性、流量强度排名
        """
        dc = self.degree_centrality()
        bc = self.betweenness_centrality()
        wdc = self.weighted_degree_centrality()

        nodes = self.nodes
        scores = []

        for node in nodes:
            # 综合评分 = 0.3*DC + 0.4*BC + 0.3*WDC
            score = 0.3 * dc.get(node, 0) + 0.4 * bc.get(node, 0) + 0.3 * wdc.get(node, 0)
            e, n = self._node_id_to_enu(node)
            n_tracks = len(self.node_attrs.get(node, {}).get("tracks_passing", set()))
            scores.append({
                "node_id": node,
                "e": e, "n": n,
                "composite_score": score,
                "degree_centrality": dc.get(node, 0),
                "betweenness_centrality": bc.get(node, 0),
                "weighted_degree": wdc.get(node, 0),
                "traffic_count": self.node_traffic.get(node, 0),
                "tracks_passing": n_tracks,
            })

        scores.sort(key=lambda x: x["composite_score"], reverse=True)
        return scores[:top_k]

    def community_detection(self, max_iter: int = 50) -> Dict[str, int]:
        """
        社区检测（标签传播算法, Label Propagation）

        识别低空交通网络中的独立航路群/社区结构。
        每个社区代表一组相互紧密连接的航路。

        Returns:
            {node_id: community_id}
        """
        nodes = self.nodes
        if not nodes:
            return {}

        # 初始化：每个节点为独立社区
        labels = {node: i for i, node in enumerate(nodes)}

        for iteration in range(max_iter):
            changed = False
            # 随机顺序遍历
            order = np.random.permutation(len(nodes))
            for idx in order:
                node = nodes[idx]
                # 收集邻居标签（加权投票）
                neighbor_labels = defaultdict(float)
                for nb, w in self.adjacency.get(node, {}).items():
                    neighbor_labels[labels.get(nb, -1)] += w
                # 反向邻居
                for src, nbs in self.adjacency.items():
                    if node in nbs:
                        neighbor_labels[labels.get(src, -1)] += nbs[node]

                if neighbor_labels:
                    best_label = max(neighbor_labels, key=neighbor_labels.get)
                    if best_label != labels[node]:
                        labels[node] = best_label
                        changed = True

            if not changed:
                break

        # 重新编号社区ID
        unique_labels = list(set(labels.values()))
        remap = {old: new for new, old in enumerate(unique_labels)}
        return {node: remap[labels[node]] for node in nodes}

    def vulnerability_analysis(self, top_k: int = 5) -> List[Dict]:
        """
        网络脆弱性分析

        评估移除某个节点后对网络连通性的影响（单点失效分析）。
        对于低空经济安全监管：脆弱节点 = 一旦该区域失效，大量航路中断。

        Returns:
            脆弱节点列表（按影响度降序）
        """
        nodes = self.nodes
        N = len(nodes)
        if N < 5:
            return []

        # 原始网络连通分量数
        original_components = self._count_components()

        # 基于介数选取候选节点（避免全遍历太慢）
        bc = self.betweenness_centrality(sample_size=min(100, N))
        candidates = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:min(30, N)]

        results = []
        for node_id, _ in candidates:
            # 模拟移除该节点后的连通性
            new_components = self._count_components(exclude_node=node_id)
            delta = new_components - original_components
            # 该节点承载的流量
            traffic = self.node_traffic.get(node_id, 0)

            e, n = self._node_id_to_enu(node_id)
            results.append({
                "node_id": node_id,
                "e": e, "n": n,
                "components_increase": delta,
                "traffic_load": traffic,
                "vulnerability_score": delta * 0.5 + (traffic / max(sum(self.node_traffic.values()), 1)) * 0.5,
            })

        results.sort(key=lambda x: x["vulnerability_score"], reverse=True)
        return results[:top_k]

    def _count_components(self, exclude_node: str = None) -> int:
        """计算连通分量数（可排除指定节点）"""
        nodes = set(self.nodes)
        if exclude_node:
            nodes.discard(exclude_node)

        visited = set()
        components = 0

        for start in nodes:
            if start in visited:
                continue
            components += 1
            # BFS
            queue = [start]
            while queue:
                curr = queue.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                for nb in self.adjacency.get(curr, {}):
                    if nb != exclude_node and nb not in visited:
                        queue.append(nb)
                for src, nbs in self.adjacency.items():
                    if src != exclude_node and curr in nbs and src not in visited:
                        queue.append(src)

        return components

    def network_statistics(self) -> Dict:
        """网络全局统计"""
        nodes = self.nodes
        N = len(nodes)
        E = self.n_edges

        avg_degree = 2.0 * E / N if N > 0 else 0
        density = E / (N * (N - 1)) if N > 1 else 0

        # 聚类系数（采样计算）
        clustering_coeffs = []
        for node in nodes[:100]:
            neighbors = set(self.adjacency.get(node, {}).keys())
            for src, nb in self.adjacency.items():
                if node in nb:
                    neighbors.add(src)
            k = len(neighbors)
            if k < 2:
                clustering_coeffs.append(0.0)
                continue
            links = 0
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    ni, nj = neighbor_list[i], neighbor_list[j]
                    if nj in self.adjacency.get(ni, {}) or ni in self.adjacency.get(nj, {}):
                        links += 1
            clustering_coeffs.append(2.0 * links / (k * (k - 1)) if k > 1 else 0)

        avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0

        # 社区检测
        communities = self.community_detection()
        n_communities = len(set(communities.values())) if communities else 0

        return {
            "n_nodes": N,
            "n_edges": E,
            "average_degree": round(avg_degree, 2),
            "density": round(density, 6),
            "average_clustering_coefficient": round(float(avg_clustering), 4),
            "n_communities": n_communities,
        }

    def generate_report(self, fused_tracks: Dict[str, pd.DataFrame]) -> Dict:
        """
        生成完整网络分析报告

        包含：全局统计 + 核心风险节点 + 社区结构 + 脆弱性分析
        """
        self.build_from_tracks(fused_tracks)

        stats = self.network_statistics()
        critical_nodes = self.identify_critical_nodes(top_k=15)
        vulnerable = self.vulnerability_analysis(top_k=10)

        return {
            "network_statistics": stats,
            "critical_risk_nodes": critical_nodes,
            "vulnerability_analysis": vulnerable,
            "total_tracks_analyzed": len(fused_tracks),
        }
