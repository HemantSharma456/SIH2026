"""Deterministic Dijkstra shortest/fastest path baseline for road graphs."""

import heapq
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx

from src.algorithms.base import BaseRouter, RouteResult
from src.traffic.cost_engine import CostWeights, DynamicCostEngine, TrafficState, INFINITY_COST


class DijkstraRouter(BaseRouter):
    """Deterministic Dijkstra shortest path algorithm with dynamic edge costs."""

    def __init__(self, cost_engine: Optional[DynamicCostEngine] = None):
        """Initialize Dijkstra router.

        Args:
            cost_engine: Default dynamic cost engine. If None, instantiates a default engine.
        """
        self.cost_engine = cost_engine or DynamicCostEngine()

    def solve(
        self,
        G: Union[nx.DiGraph, nx.MultiDiGraph],
        origin: Any,
        destination: Any,
        cost_engine: Optional[DynamicCostEngine] = None,
        traffic_states: Optional[Dict[Tuple[Any, Any], TrafficState]] = None,
        weights: Optional[CostWeights] = None,
        **kwargs: Any,
    ) -> RouteResult:
        """Find the cost-optimal directed path from origin to destination.

        Handles MultiDiGraph (multiple parallel edges between the same pair of nodes)
        and skips closed edges (cost = INFINITY_COST).

        Args:
            G: Directed NetworkX graph (DiGraph or MultiDiGraph).
            origin: Starting node ID.
            destination: Target node ID.
            cost_engine: Dynamic cost engine to use (defaults to instance engine).
            traffic_states: Optional dictionary mapping (u, v) or (u, v, key) to TrafficState.
            weights: Optional cost weights.

        Returns:
            RouteResult with optimal route, distance, travel time, and cost.
        """
        start_time = time.perf_counter()
        engine = cost_engine or self.cost_engine
        active_weights = weights or engine.weights
        traffic_states = traffic_states or {}

        # Validation of endpoints
        if origin not in G or destination not in G:
            exec_time = (time.perf_counter() - start_time) * 1000.0
            error_msg = f"Origin '{origin}' or Destination '{destination}' not present in graph."
            return RouteResult(
                origin=origin,
                destination=destination,
                route=[],
                is_valid=False,
                algorithm_name="Dijkstra",
                execution_time_ms=exec_time,
                metadata={"error": error_msg},
            )

        # Same origin and destination
        if origin == destination:
            exec_time = (time.perf_counter() - start_time) * 1000.0
            return RouteResult(
                origin=origin,
                destination=destination,
                route=[origin],
                edge_keys=[],
                total_cost=0.0,
                total_distance_meters=0.0,
                total_travel_time_sec=0.0,
                is_valid=True,
                algorithm_name="Dijkstra",
                execution_time_ms=exec_time,
                metadata={"status": "origin_equals_destination"},
            )

        # Priority queue storing (cumulative_cost, node_id)
        pq: List[Tuple[float, Any]] = [(0.0, origin)]
        min_costs: Dict[Any, float] = {origin: 0.0}
        predecessors: Dict[Any, Tuple[Any, Any, float, float, float]] = {}
        # maps node v -> (u, chosen_edge_key, edge_cost, edge_dist, edge_time)

        visited = set()

        while pq:
            curr_cost, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == destination:
                break

            # Explore directed outgoing edges
            # In NetworkX, G[u] returns successors mapping {v: edge_dict}
            for v, edge_dict in G[u].items():
                if v in visited:
                    continue

                best_edge_key = None
                best_edge_cost = INFINITY_COST
                best_edge_dist = 0.0
                best_edge_time = 0.0

                if G.is_multigraph():
                    # edge_dict is {key: data_dict}
                    for key, data in edge_dict.items():
                        state = traffic_states.get((u, v, key)) or traffic_states.get((u, v))
                        res = engine.compute_edge_cost(u, v, data, traffic_state=state, weights=active_weights)
                        if not res.is_closed and res.effective_cost < best_edge_cost:
                            best_edge_cost = res.effective_cost
                            best_edge_key = key
                            best_edge_dist = res.distance_meters
                            best_edge_time = res.travel_time_sec
                else:
                    # edge_dict is data_dict
                    state = traffic_states.get((u, v))
                    res = engine.compute_edge_cost(u, v, edge_dict, traffic_state=state, weights=active_weights)
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

        exec_time = (time.perf_counter() - start_time) * 1000.0

        # Destination unreachable
        if destination not in predecessors and origin != destination:
            return RouteResult(
                origin=origin,
                destination=destination,
                route=[],
                edge_keys=[],
                total_cost=INFINITY_COST,
                total_distance_meters=0.0,
                total_travel_time_sec=0.0,
                is_valid=False,
                algorithm_name="Dijkstra",
                execution_time_ms=exec_time,
                metadata={"error": "Destination is unreachable (due to one-way constraints or closed roads)."},
            )

        # Reconstruct route from predecessors
        path_nodes: List[Any] = [destination]
        path_keys: List[Any] = []
        total_dist = 0.0
        total_time = 0.0
        curr = destination

        while curr != origin:
            u, key, _, dist, t_sec = predecessors[curr]
            path_keys.append(key)
            total_dist += dist
            total_time += t_sec
            path_nodes.append(u)
            curr = u

        path_nodes.reverse()
        path_keys.reverse()

        return RouteResult(
            origin=origin,
            destination=destination,
            route=path_nodes,
            edge_keys=path_keys,
            total_cost=min_costs[destination],
            total_distance_meters=round(total_dist, 2),
            total_travel_time_sec=round(total_time, 2),
            is_valid=True,
            algorithm_name="Dijkstra",
            execution_time_ms=round(exec_time, 3),
            metadata={"status": "optimal_route_found"},
        )


def dijkstra_route(
    G: Union[nx.DiGraph, nx.MultiDiGraph],
    origin: Any,
    destination: Any,
    cost_engine: Optional[DynamicCostEngine] = None,
    traffic_states: Optional[Dict[Tuple[Any, Any], TrafficState]] = None,
    weights: Optional[CostWeights] = None,
) -> RouteResult:
    """Convenience function to run Dijkstra routing on a graph.

    Args:
        G: Directed graph.
        origin: Origin node ID.
        destination: Destination node ID.
        cost_engine: Dynamic cost engine.
        traffic_states: Optional edge traffic states.
        weights: Optional cost weights.

    Returns:
        RouteResult dataclass.
    """
    router = DijkstraRouter(cost_engine=cost_engine)
    return router.solve(
        G=G,
        origin=origin,
        destination=destination,
        traffic_states=traffic_states,
        weights=weights,
    )
