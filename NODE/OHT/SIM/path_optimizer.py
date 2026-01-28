"""
OHT 최적 경로 알고리즘 모듈
- 실시간 교통 상황 반영 동적 경로 탐색
- 정체/혼잡 구간 우회 기능
- A* 알고리즘 기반 최적 경로 탐색

사용법:
    from path_optimizer import PathOptimizer

    optimizer = PathOptimizer(nodes, edges, graph)
    optimizer.update_traffic(rail_edge_map, hid_zones, vehicles)
    path = optimizer.find_optimal_path(start, end)
"""

import heapq
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class PathStrategy(Enum):
    """경로 탐색 전략"""
    SHORTEST = "shortest"           # 최단 거리
    FASTEST = "fastest"             # 최소 시간 (교통 반영)
    SAFEST = "safest"               # 정체 회피 우선
    BALANCED = "balanced"           # 균형 (거리 + 교통)


@dataclass
class EdgeTraffic:
    """엣지 교통 정보"""
    edge_id: str
    from_node: int
    to_node: int
    base_length: float = 100.0      # 기본 거리 (mm)

    # 실시간 교통 정보
    vehicle_count: int = 0          # 현재 차량 수
    jam_count: int = 0              # 정체 차량 수
    avg_velocity: float = 150.0     # 평균 속도 (m/min)
    density: float = 0.0            # 밀도 (%)
    in_count: int = 0               # 진입 수
    out_count: int = 0              # 진출 수

    @property
    def congestion_level(self) -> float:
        """혼잡도 (0~1, 높을수록 혼잡)"""
        # 밀도, 정체 차량, In/Out 비율 고려
        density_factor = min(1.0, self.density / 100.0)
        jam_factor = min(1.0, self.jam_count / 3.0)  # 3대 이상 JAM이면 최대

        in_out_ratio = 1.0
        if self.out_count > 0:
            in_out_ratio = min(2.0, self.in_count / self.out_count)
        elif self.in_count > 0:
            in_out_ratio = 2.0
        in_out_factor = (in_out_ratio - 1.0) / 1.0  # 0~1 정규화

        return min(1.0, density_factor * 0.4 + jam_factor * 0.4 + in_out_factor * 0.2)

    @property
    def estimated_travel_time(self) -> float:
        """예상 통과 시간 (초)"""
        if self.avg_velocity <= 0:
            return float('inf')
        # 거리(mm) / 속도(m/min) * 60 / 1000 = 초
        base_time = (self.base_length / self.avg_velocity) * 60.0 / 1000.0
        # 혼잡도에 따른 지연 추가
        delay_factor = 1.0 + self.congestion_level * 3.0  # 최대 4배 지연
        return base_time * delay_factor

    @property
    def is_blocked(self) -> bool:
        """통행 불가 여부"""
        return self.jam_count >= 2 or self.density > 90


@dataclass
class ZoneTraffic:
    """Zone 교통 정보"""
    zone_id: int
    vehicle_count: int = 0
    vehicle_max: int = 10
    vehicle_precaution: int = 7
    status: str = "NORMAL"

    @property
    def congestion_level(self) -> float:
        """Zone 혼잡도 (0~1)"""
        if self.vehicle_max <= 0:
            return 0.0
        return min(1.0, self.vehicle_count / self.vehicle_max)

    @property
    def is_full(self) -> bool:
        return self.status == "FULL" or self.vehicle_count >= self.vehicle_max


class PathOptimizer:
    """OHT 최적 경로 탐색기"""

    def __init__(self, nodes: Dict, edges: List, graph: Dict):
        """
        Args:
            nodes: 노드 정보 {node_id: Node}
            edges: 엣지 리스트 [(from, to, distance), ...]
            graph: 인접 리스트 {node_id: [(neighbor, distance), ...]}
        """
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

        # 교통 정보
        self.edge_traffic: Dict[Tuple[int, int], EdgeTraffic] = {}
        self.zone_traffic: Dict[int, ZoneTraffic] = {}

        # 노드 -> Zone 매핑 (Zone 내 노드 판별용)
        self.node_to_zone: Dict[int, int] = {}

        # 캐시
        self._heuristic_cache: Dict[Tuple[int, int], float] = {}

        # 기본 설정
        self.strategy = PathStrategy.BALANCED
        self.max_alternatives = 3  # 최대 대안 경로 수

        print(f"[PathOptimizer] 초기화: 노드 {len(nodes)}개, 엣지 {len(edges)}개")

    def update_traffic(self, rail_edge_map: Dict = None,
                       hid_zones: Dict = None,
                       vehicles: Dict = None,
                       lane_to_zone: Dict = None):
        """실시간 교통 정보 업데이트

        Args:
            rail_edge_map: RailEdge 객체 맵 {edge_id: RailEdge}
            hid_zones: HIDZone 객체 맵 {zone_id: HIDZone}
            vehicles: Vehicle 객체 맵 {vehicle_id: Vehicle}
            lane_to_zone: Lane -> Zone 매핑 {(from, to): zone_id}
        """
        # RailEdge 정보 업데이트
        if rail_edge_map:
            for edge_id, rail_edge in rail_edge_map.items():
                key = (rail_edge.fromNodeId, rail_edge.toNodeId)

                # 정체 차량 수 계산
                jam_count = 0
                if vehicles:
                    for v in vehicles.values():
                        if v.currentNode == rail_edge.fromNodeId and v.nextNode == rail_edge.toNodeId:
                            if hasattr(v, 'udpState') and v.udpState.state.value == 7:  # JAM
                                jam_count += 1

                self.edge_traffic[key] = EdgeTraffic(
                    edge_id=edge_id,
                    from_node=rail_edge.fromNodeId,
                    to_node=rail_edge.toNodeId,
                    base_length=rail_edge.length,
                    vehicle_count=len(rail_edge.vhlIdMap) if hasattr(rail_edge, 'vhlIdMap') else 0,
                    jam_count=jam_count,
                    avg_velocity=rail_edge.getAvgVelocity() if hasattr(rail_edge, 'getAvgVelocity') else 150.0,
                    density=rail_edge.getDensity() if hasattr(rail_edge, 'getDensity') else 0.0,
                    in_count=rail_edge.inCount if hasattr(rail_edge, 'inCount') else 0,
                    out_count=rail_edge.outCount if hasattr(rail_edge, 'outCount') else 0
                )

        # Zone 정보 업데이트
        if hid_zones:
            for zone_id, zone in hid_zones.items():
                self.zone_traffic[zone_id] = ZoneTraffic(
                    zone_id=zone_id,
                    vehicle_count=zone.vehicleCount if hasattr(zone, 'vehicleCount') else 0,
                    vehicle_max=zone.vehicleMax if hasattr(zone, 'vehicleMax') else 10,
                    vehicle_precaution=zone.vehiclePrecaution if hasattr(zone, 'vehiclePrecaution') else 7,
                    status=zone.status if hasattr(zone, 'status') else "NORMAL"
                )

        # Lane -> Zone 매핑 저장
        if lane_to_zone:
            for (from_n, to_n), zone_id in lane_to_zone.items():
                self.node_to_zone[from_n] = zone_id
                self.node_to_zone[to_n] = zone_id

    def _heuristic(self, node: int, goal: int) -> float:
        """A* 휴리스틱 함수 (유클리드 거리)"""
        cache_key = (node, goal)
        if cache_key in self._heuristic_cache:
            return self._heuristic_cache[cache_key]

        n1 = self.nodes.get(node)
        n2 = self.nodes.get(goal)

        if not n1 or not n2:
            return 0.0

        # 유클리드 거리
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        dist = math.sqrt(dx * dx + dy * dy)

        self._heuristic_cache[cache_key] = dist
        return dist

    def _get_edge_cost(self, from_node: int, to_node: int,
                       base_distance: float) -> float:
        """엣지 비용 계산 (전략에 따라 다름)"""
        key = (from_node, to_node)
        traffic = self.edge_traffic.get(key)

        if self.strategy == PathStrategy.SHORTEST:
            return base_distance

        if not traffic:
            return base_distance

        # 통행 불가 엣지는 매우 높은 비용
        if traffic.is_blocked:
            return base_distance * 100.0

        if self.strategy == PathStrategy.FASTEST:
            # 예상 통과 시간 기반
            return traffic.estimated_travel_time * 1000  # 초 -> 적절한 스케일

        elif self.strategy == PathStrategy.SAFEST:
            # 혼잡도 회피 우선
            congestion_penalty = 1.0 + traffic.congestion_level * 5.0
            return base_distance * congestion_penalty

        else:  # BALANCED
            # 거리 + 혼잡도 균형
            congestion_penalty = 1.0 + traffic.congestion_level * 2.0
            return base_distance * congestion_penalty

    def _get_zone_penalty(self, node: int) -> float:
        """노드가 속한 Zone의 혼잡 페널티"""
        zone_id = self.node_to_zone.get(node)
        if not zone_id:
            return 0.0

        zone = self.zone_traffic.get(zone_id)
        if not zone:
            return 0.0

        if zone.is_full:
            return 1000.0  # 포화 Zone 회피

        return zone.congestion_level * 100.0

    def find_optimal_path(self, start: int, end: int,
                          strategy: PathStrategy = None,
                          blocked_edges: Set[Tuple[int, int]] = None) -> List[int]:
        """최적 경로 탐색 (A* 알고리즘)

        Args:
            start: 출발 노드
            end: 도착 노드
            strategy: 경로 탐색 전략 (None이면 기본값 사용)
            blocked_edges: 회피할 엣지 집합 {(from, to), ...}

        Returns:
            경로 노드 리스트 [start, ..., end] 또는 빈 리스트
        """
        if start == end:
            return [start]

        if start not in self.nodes or end not in self.nodes:
            return []

        if strategy:
            self.strategy = strategy

        blocked = blocked_edges or set()

        # A* 알고리즘
        # (f_score, g_score, node, path)
        open_set = [(0, 0, start, [start])]
        closed_set = set()
        g_scores = {start: 0}

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current == end:
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            # 이웃 노드 탐색
            for neighbor, base_dist in self.graph.get(current, []):
                if neighbor in closed_set:
                    continue

                # 차단된 엣지 회피
                if (current, neighbor) in blocked:
                    continue

                # 엣지 비용 계산
                edge_cost = self._get_edge_cost(current, neighbor, base_dist)

                # Zone 페널티 추가
                zone_penalty = self._get_zone_penalty(neighbor)

                tentative_g = g_score + edge_cost + zone_penalty

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = self._heuristic(neighbor, end)
                    f_score = tentative_g + h_score
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))

        return []  # 경로 없음

    def find_alternative_paths(self, start: int, end: int,
                               primary_path: List[int] = None,
                               count: int = 3) -> List[List[int]]:
        """대안 경로 탐색

        Args:
            start: 출발 노드
            end: 도착 노드
            primary_path: 주 경로 (회피 대상)
            count: 대안 경로 수

        Returns:
            대안 경로 리스트
        """
        alternatives = []
        blocked_edges: Set[Tuple[int, int]] = set()

        # 주 경로가 없으면 먼저 찾기
        if not primary_path:
            primary_path = self.find_optimal_path(start, end)

        if not primary_path:
            return []

        alternatives.append(primary_path)

        # 주 경로의 엣지를 하나씩 차단하면서 대안 찾기
        for i in range(len(primary_path) - 1):
            if len(alternatives) >= count:
                break

            edge = (primary_path[i], primary_path[i + 1])
            blocked_edges.add(edge)

            alt_path = self.find_optimal_path(start, end, blocked_edges=blocked_edges)

            if alt_path and alt_path not in alternatives:
                alternatives.append(alt_path)

            # 다음 반복을 위해 차단 해제
            blocked_edges.discard(edge)

        return alternatives

    def should_reroute(self, vehicle_id: str, current_path: List[int],
                       current_index: int) -> Tuple[bool, str]:
        """경로 재탐색 필요 여부 판단

        Args:
            vehicle_id: 차량 ID
            current_path: 현재 경로
            current_index: 현재 경로 인덱스

        Returns:
            (재탐색 필요 여부, 이유)
        """
        if not current_path or current_index >= len(current_path) - 1:
            return False, ""

        # 남은 경로에서 막힌 구간 확인
        for i in range(current_index, len(current_path) - 1):
            from_n = current_path[i]
            to_n = current_path[i + 1]

            traffic = self.edge_traffic.get((from_n, to_n))
            if traffic and traffic.is_blocked:
                return True, f"엣지 {from_n}->{to_n} 막힘 (JAM: {traffic.jam_count}, 밀도: {traffic.density:.1f}%)"

            # Zone 확인
            zone_id = self.node_to_zone.get(to_n)
            if zone_id:
                zone = self.zone_traffic.get(zone_id)
                if zone and zone.is_full:
                    return True, f"Zone {zone_id} 포화"

        return False, ""

    def get_rerouted_path(self, start: int, end: int,
                          avoid_path: List[int] = None) -> Optional[List[int]]:
        """우회 경로 탐색

        Args:
            start: 현재 위치 (출발 노드)
            end: 목적지
            avoid_path: 회피할 기존 경로

        Returns:
            우회 경로 또는 None
        """
        blocked_edges = set()

        # 기존 경로의 막힌 구간 차단
        if avoid_path:
            for i in range(len(avoid_path) - 1):
                from_n = avoid_path[i]
                to_n = avoid_path[i + 1]

                traffic = self.edge_traffic.get((from_n, to_n))
                if traffic and traffic.is_blocked:
                    blocked_edges.add((from_n, to_n))

        # 현재 막힌 모든 엣지 차단
        for key, traffic in self.edge_traffic.items():
            if traffic.is_blocked:
                blocked_edges.add(key)

        # SAFEST 전략으로 우회 경로 탐색
        return self.find_optimal_path(start, end,
                                      strategy=PathStrategy.SAFEST,
                                      blocked_edges=blocked_edges)

    def get_path_info(self, path: List[int]) -> Dict:
        """경로 정보 반환

        Args:
            path: 경로 노드 리스트

        Returns:
            경로 정보 딕셔너리
        """
        if not path:
            return {"valid": False}

        total_distance = 0
        total_time = 0
        max_congestion = 0
        blocked_count = 0

        for i in range(len(path) - 1):
            from_n = path[i]
            to_n = path[i + 1]

            # 기본 거리
            for neighbor, dist in self.graph.get(from_n, []):
                if neighbor == to_n:
                    total_distance += dist
                    break

            # 교통 정보
            traffic = self.edge_traffic.get((from_n, to_n))
            if traffic:
                total_time += traffic.estimated_travel_time
                max_congestion = max(max_congestion, traffic.congestion_level)
                if traffic.is_blocked:
                    blocked_count += 1

        return {
            "valid": True,
            "length": len(path),
            "total_distance": round(total_distance, 2),
            "estimated_time": round(total_time, 2),
            "max_congestion": round(max_congestion, 3),
            "blocked_edges": blocked_count,
            "path": path
        }

    def clear_cache(self):
        """캐시 초기화"""
        self._heuristic_cache.clear()


# ============================================================
# 시뮬레이터 통합 헬퍼 함수
# ============================================================

def create_optimizer_from_engine(engine) -> PathOptimizer:
    """SimulationEngine에서 PathOptimizer 생성

    Args:
        engine: SimulationEngine 인스턴스

    Returns:
        PathOptimizer 인스턴스
    """
    optimizer = PathOptimizer(
        nodes=engine.nodes,
        edges=engine.edges,
        graph=engine.graph
    )
    return optimizer


def update_optimizer_from_engine(optimizer: PathOptimizer, engine):
    """SimulationEngine의 실시간 데이터로 PathOptimizer 업데이트

    Args:
        optimizer: PathOptimizer 인스턴스
        engine: SimulationEngine 인스턴스
    """
    # Lane -> Zone 매핑 병합
    lane_to_zone = {}
    if hasattr(engine, 'in_lane_to_zone'):
        lane_to_zone.update(engine.in_lane_to_zone)
    if hasattr(engine, 'out_lane_to_zone'):
        lane_to_zone.update(engine.out_lane_to_zone)

    optimizer.update_traffic(
        rail_edge_map=getattr(engine, 'rail_edge_map', None),
        hid_zones=getattr(engine, 'hid_zones', None),
        vehicles=getattr(engine, 'vehicles', None),
        lane_to_zone=lane_to_zone
    )


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("PathOptimizer 모듈 테스트")

    # 간단한 그래프로 테스트
    nodes = {
        1: type('Node', (), {'x': 0, 'y': 0})(),
        2: type('Node', (), {'x': 100, 'y': 0})(),
        3: type('Node', (), {'x': 200, 'y': 0})(),
        4: type('Node', (), {'x': 100, 'y': 100})(),
        5: type('Node', (), {'x': 200, 'y': 100})(),
    }

    edges = [
        (1, 2, 100), (2, 3, 100),
        (1, 4, 141), (4, 5, 100),
        (2, 4, 100), (3, 5, 100),
        (4, 3, 141)
    ]

    graph = defaultdict(list)
    for from_n, to_n, dist in edges:
        graph[from_n].append((to_n, dist))
        graph[to_n].append((from_n, dist))  # 양방향

    optimizer = PathOptimizer(nodes, edges, dict(graph))

    # 최적 경로 탐색
    path = optimizer.find_optimal_path(1, 5)
    print(f"1 -> 5 최적 경로: {path}")

    info = optimizer.get_path_info(path)
    print(f"경로 정보: {info}")

    # 대안 경로 탐색
    alternatives = optimizer.find_alternative_paths(1, 5, count=3)
    print(f"대안 경로들: {alternatives}")

    print("\n테스트 완료!")