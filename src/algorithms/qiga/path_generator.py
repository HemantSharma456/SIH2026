"""Graph-constrained candidate path generator for QIGA optimization.

Generates up to K feasible, distinct, loopless candidate routes between
an origin and destination using Yen's algorithm integrated with DynamicCostEngine.
"""

from dataclasses import dataclass, field
import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

from src.algorithms.baselines.dijkstra import DijkstraRouter
from src.algorithms.base import RouteResult
from src.traffic.cost_engine import CostWeights, DynamicCostEngine, TrafficState, INFINITY_COST


@dataclass
class CandidateRoute:
    """Represents a feasible, continuous route candidate for QIGA optimization."""
    candidate_id: int
    origin: Any
    destination: Any
    route: List[Any]                           # Sequence of node IDs
    edge_keys: List[Any]                       # Sequence of edge keys for MultiDiGraph
    total_distance_meters: float               # Physical route length in meters
    total_travel_time_sec: float               # Estimated traversal time in seconds
    total_cost: float                          # Evaluated impedance
    is_valid: bool = True                      # Topological and traffic validity
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the candidate route."""
        return len(self.route)

    @property
    def total_distance_km(self) -> float:
        """Total distance in kilometers."""
        return self.total_distance_meters / 1000.0

    @property
    def total_travel_time_min(self) -> float:
        """Total travel time in minutes."""
        return self.total_travel_time_sec / 60.0


def _evaluate_path_metrics(
    G: Union[nx.DiGraph, nx.MultiDiGraph],
    route_nodes: List[Any],
    route_keys: List[Any],
    cost_engine: DynamicCostEngine,
    traffic_states: Dict[Any, TrafficState],
    weights: CostWeights,
) -> Tuple[float, float, float]:
    """Calculate total cost, distance, and travel time for a specific node and key sequence.

    Returns:
        Tuple of (total_cost, total_distance_meters, total_travel_time_sec).
    """
    total_cost = 0.0
    total_dist = 0.0
    total_time = 0.0

    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        key = route_keys[i] if i < len(route_keys) else 0

        if G.is_multigraph():
            edge_data = G[u][v].get(key, next(iter(G[u][v].values())))
        else:
            edge_data = G[u][v]

        state = traffic_states.get((u, v, key)) or traffic_states.get((u, v))
        res = cost_engine.compute_edge_cost(u, v, edge_data, traffic_state=state, weights=weights)

        total_cost += res.effective_cost
        total_dist += res.distance_meters
        total_time += res.travel_time_sec

    return round(total_cost, 4), round(total_dist, 2), round(total_time, 2)


def _dijkstra_constrained(
    G: Union[nx.DiGraph, nx.MultiDiGraph],
    origin: Any,
    destination: Any,
    cost_engine: DynamicCostEngine,
    traffic_states: Dict[Any, TrafficState],
    weights: CostWeights,
    excluded_nodes: Set[Any],
    excluded_edges: Set[Tuple[Any, Any]],
) -> Optional[RouteResult]:
    """Constrained Dijkstra search respecting excluded nodes and excluded directed edges."""
    if origin in excluded_nodes or destination in excluded_nodes:
        return None
    if origin not in G or destination not in G:
        return None
    if origin == destination:
        return RouteResult(
            origin=origin,
            destination=destination,
            route=[origin],
            edge_keys=[],
            total_cost=0.0,
            total_distance_meters=0.0,
            total_travel_time_sec=0.0,
            is_valid=True,
            algorithm_name="ConstrainedDijkstra",
        )

    pq: List[Tuple[float, Any]] = [(0.0, origin)]
    min_costs: Dict[Any, float] = {origin: 0.0}
    predecessors: Dict[Any, Tuple[Any, Any, float, float, float]] = {}
    visited = set()

    while pq:
        curr_cost, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == destination:
            break

        for v, edge_dict in G[u].items():
            if v in visited or v in excluded_nodes:
                continue

            best_edge_key = None
            best_edge_cost = INFINITY_COST
            best_edge_dist = 0.0
            best_edge_time = 0.0

            if G.is_multigraph():
                for key, data in edge_dict.items():
                    # Check if edge (u, v) or (u, v, key) is excluded
                    if (u, v) in excluded_edges or (u, v, key) in excluded_edges:
                        continue
                    state = traffic_states.get((u, v, key)) or traffic_states.get((u, v))
                    res = cost_engine.compute_edge_cost(u, v, data, traffic_state=state, weights=weights)
                    if not res.is_closed and res.effective_cost < best_edge_cost:
                        best_edge_cost = res.effective_cost
                        best_edge_key = key
                        best_edge_dist = res.distance_meters
                        best_edge_time = res.travel_time_sec
            else:
                if (u, v) in excluded_edges:
                    continue
                state = traffic_states.get((u, v))
                res = cost_engine.compute_edge_cost(u, v, edge_dict, traffic_state=state, weights=weights)
                if not res.is_closed:
                    best_edge_cost = res.effective_cost
                    best_edge_key = 0
                    best_edge_dist = res.distance_meters
                    best_edge_time = res.travel_time_sec

            if best_edge_cost < INFINITY_COST:
                new_cost = curr_cost + best_edge_cost
                if new_cost < min_costs.get(v, INFINITY_COST):
                    min_costs[v] = new_cost
                    predecessors[v] = (u, best_edge_key, best_edge_cost, best_edge_dist, best_edge_time)
                    heapq.heappush(pq, (new_cost, v))

    if destination not in predecessors:
        return None

    path_nodes = [destination]
    path_keys = []
    curr = destination
    while curr != origin:
        u, key, _, _, _ = predecessors[curr]
        path_keys.append(key)
        path_nodes.append(u)
        curr = u

    path_nodes.reverse()
    path_keys.reverse()

    total_cost, total_dist, total_time = _evaluate_path_metrics(
        G, path_nodes, path_keys, cost_engine, traffic_states, weights
    )

    return RouteResult(
        origin=origin,
        destination=destination,
        route=path_nodes,
        edge_keys=path_keys,
        total_cost=total_cost,
        total_distance_meters=total_dist,
        total_travel_time_sec=total_time,
        is_valid=True,
        algorithm_name="ConstrainedDijkstra",
    )


class GraphConstrainedPathGenerator:
    """Generates up to K distinct, loopless, graph-constrained candidate paths."""

    def __init__(self, cost_engine: Optional[DynamicCostEngine] = None):
        """Initialize candidate path generator.

        Args:
            cost_engine: DynamicCostEngine instance. If None, uses default engine.
        """
        self.cost_engine = cost_engine or DynamicCostEngine()

    def generate(
        self,
        G: Union[nx.DiGraph, nx.MultiDiGraph],
        origin: Any,
        destination: Any,
        k: int = 5,
        cost_engine: Optional[DynamicCostEngine] = None,
        traffic_states: Optional[Dict[Tuple[Any, Any], TrafficState]] = None,
        weights: Optional[CostWeights] = None,
    ) -> List[CandidateRoute]:
        """Generate up to K distinct, valid candidate routes between origin and destination.

        Uses Yen's K-Shortest Paths algorithm with dynamic cost evaluation.
        Every candidate path is guaranteed to be a continuous, directed, open path.

        Args:
            G: Directed NetworkX road graph (DiGraph or MultiDiGraph).
            origin: Origin node ID.
            destination: Destination node ID.
            k: Maximum number of distinct feasible routes to generate (k >= 1).
            cost_engine: Optional dynamic cost engine overriding default.
            traffic_states: Optional dictionary mapping (u, v) -> TrafficState.
            weights: Optional CostWeights profile.

        Returns:
            List of CandidateRoute objects (up to k items, sorted by ascending cost).
        """
        if k < 1:
            return []

        engine = cost_engine or self.cost_engine
        active_weights = weights or engine.weights
        traffic_states = traffic_states or {}

        if origin not in G or destination not in G:
            return []

        # Find the 1st shortest path using baseline Dijkstra
        initial_route = _dijkstra_constrained(
            G=G,
            origin=origin,
            destination=destination,
            cost_engine=engine,
            traffic_states=traffic_states,
            weights=active_weights,
            excluded_nodes=set(),
            excluded_edges=set(),
        )

        if initial_route is None or not initial_route.is_valid:
            return []

        # A holds the accepted K shortest paths
        A: List[CandidateRoute] = [
            CandidateRoute(
                candidate_id=1,
                origin=origin,
                destination=destination,
                route=initial_route.route,
                edge_keys=initial_route.edge_keys or [],
                total_distance_meters=initial_route.total_distance_meters,
                total_travel_time_sec=initial_route.total_travel_time_sec,
                total_cost=initial_route.total_cost,
                is_valid=True,
                metadata={"rank": 1, "generation_method": "shortest_path"},
            )
        ]

        # B holds potential candidate paths as a min-heap: (cost, counter, path_nodes, path_keys)
        candidate_heap: List[Tuple[float, int, List[Any], List[Any]]] = []
        heap_counter = 0
        seen_routes: Set[Tuple[Any, ...]] = {tuple(initial_route.route)}

        for i in range(1, k):
            prev_route = A[i - 1]
            prev_nodes = prev_route.route
            prev_keys = prev_route.edge_keys

            # The spur node ranges from the first node up to the second-to-last node
            for j in range(len(prev_nodes) - 1):
                spur_node = prev_nodes[j]
                root_path_nodes = prev_nodes[: j + 1]
                root_path_keys = prev_keys[:j]

                # Identify edges to exclude: edges that share the same root path in previously accepted routes
                excluded_edges: Set[Tuple[Any, Any]] = set()
                for p in A:
                    if len(p.route) > j + 1 and p.route[: j + 1] == root_path_nodes:
                        next_node = p.route[j + 1]
                        if G.is_multigraph() and len(p.edge_keys) > j:
                            key = p.edge_keys[j]
                            excluded_edges.add((spur_node, next_node, key))
                        excluded_edges.add((spur_node, next_node))

                # Identify nodes to exclude: all nodes in the root path except the spur node
                excluded_nodes: Set[Any] = set(root_path_nodes[:-1])

                # Calculate the spur path from spur_node to destination
                spur_res = _dijkstra_constrained(
                    G=G,
                    origin=spur_node,
                    destination=destination,
                    cost_engine=engine,
                    traffic_states=traffic_states,
                    weights=active_weights,
                    excluded_nodes=excluded_nodes,
                    excluded_edges=excluded_edges,
                )

                if spur_res is not None and spur_res.is_valid:
                    # Combine root path and spur path
                    total_nodes = root_path_nodes[:-1] + spur_res.route
                    total_keys = root_path_keys + (spur_res.edge_keys or [])

                    route_tuple = tuple(total_nodes)
                    if route_tuple not in seen_routes:
                        seen_routes.add(route_tuple)
                        cost, dist, t_sec = _evaluate_path_metrics(
                            G, total_nodes, total_keys, engine, traffic_states, active_weights
                        )
                        heap_counter += 1
                        heapq.heappush(candidate_heap, (cost, heap_counter, total_nodes, total_keys))

            if not candidate_heap:
                # No more alternative paths exist in the graph
                break

            # Pop the lowest-cost distinct path from the candidate heap
            best_cost, _, best_nodes, best_keys = heapq.heappop(candidate_heap)
            _, best_dist, best_time = _evaluate_path_metrics(
                G, best_nodes, best_keys, engine, traffic_states, active_weights
            )

            A.append(
                CandidateRoute(
                    candidate_id=len(A) + 1,
                    origin=origin,
                    destination=destination,
                    route=best_nodes,
                    edge_keys=best_keys,
                    total_distance_meters=best_dist,
                    total_travel_time_sec=best_time,
                    total_cost=best_cost,
                    is_valid=True,
                    metadata={"rank": len(A) + 1, "generation_method": "yen_ksp"},
                )
            )

        return A


def generate_candidate_paths(
    G: Union[nx.DiGraph, nx.MultiDiGraph],
    origin: Any,
    destination: Any,
    k: int = 5,
    cost_engine: Optional[DynamicCostEngine] = None,
    traffic_states: Optional[Dict[Tuple[Any, Any], TrafficState]] = None,
    weights: Optional[CostWeights] = None,
) -> List[CandidateRoute]:
    """Convenience function to generate up to K graph-constrained candidate paths.

    Args:
        G: Directed NetworkX graph.
        origin: Origin node ID.
        destination: Destination node ID.
        k: Maximum number of candidate paths (default 5).
        cost_engine: Dynamic cost engine.
        traffic_states: Optional edge traffic state dictionary.
        weights: Optional cost weights.

    Returns:
        List of CandidateRoute instances.
    """
    generator = GraphConstrainedPathGenerator(cost_engine=cost_engine)
    return generator.generate(
        G=G,
        origin=origin,
        destination=destination,
        k=k,
        traffic_states=traffic_states,
        weights=weights,
    )
