"""Unit tests for deterministic Dijkstra baseline routing."""

import pytest
import networkx as nx

from src.algorithms.baselines.dijkstra import DijkstraRouter, dijkstra_route
from src.graph.validator import validate_route
from src.traffic.cost_engine import CostWeights, DynamicCostEngine, TrafficState, INFINITY_COST


@pytest.fixture
def synthetic_multigraph() -> nx.MultiDiGraph:
    """Create a synthetic MultiDiGraph with two paths:
    Path 1 (via 2): 1 -> 2 -> 4 (Short distance: 600m, slow speed: 20 km/h = 5.56 m/s -> ~108s)
    Path 2 (via 3): 1 -> 3 -> 4 (Long distance: 1000m, fast speed: 80 km/h = 22.22 m/s -> ~45s)
    """
    G = nx.MultiDiGraph()
    G.add_node(1, x=77.0, y=28.0)
    G.add_node(2, x=77.1, y=28.1)
    G.add_node(3, x=77.1, y=27.9)
    G.add_node(4, x=77.2, y=28.0)

    # Path 1: Short but slow
    G.add_edge(1, 2, key=0, length=300.0, maxspeed="20", highway="residential")
    G.add_edge(2, 4, key=0, length=300.0, maxspeed="20", highway="residential")

    # Path 2: Long but fast (motorway)
    G.add_edge(1, 3, key=0, length=500.0, maxspeed="80", highway="motorway")
    G.add_edge(3, 4, key=0, length=500.0, maxspeed="80", highway="motorway")

    # Multi-edge between 1 and 3: key 1 is slower
    G.add_edge(1, 3, key=1, length=500.0, maxspeed="30", highway="secondary")

    return G


def test_dijkstra_fastest_vs_shortest_path(synthetic_multigraph: nx.MultiDiGraph):
    """Test that time-weighted routing picks the highway (fastest) while distance-weighted picks residential (shortest)."""
    router = DijkstraRouter()

    # 1. Fastest time objective (weight_time=1.0, weight_distance=0.0)
    time_weights = CostWeights(weight_time=1.0, weight_distance=0.0)
    res_fastest = router.solve(synthetic_multigraph, origin=1, destination=4, weights=time_weights)

    assert res_fastest.is_valid is True
    assert res_fastest.route == [1, 3, 4]  # via highway
    assert res_fastest.total_distance_meters == 1000.0
    assert res_fastest.total_travel_time_sec == pytest.approx(45.0, rel=1e-2)

    # 2. Shortest distance objective (weight_time=0.0, weight_distance=1.0)
    dist_weights = CostWeights(weight_time=0.0, weight_distance=1.0)
    res_shortest = router.solve(synthetic_multigraph, origin=1, destination=4, weights=dist_weights)

    assert res_shortest.is_valid is True
    assert res_shortest.route == [1, 2, 4]  # via residential
    assert res_shortest.total_distance_meters == 600.0
    assert res_shortest.total_travel_time_sec == pytest.approx(108.0, rel=1e-2)


def test_dijkstra_multigraph_parallel_edge_selection(synthetic_multigraph: nx.MultiDiGraph):
    """Test that Dijkstra picks key=0 (80 km/h) rather than key=1 (30 km/h) for edge 1 -> 3."""
    res = dijkstra_route(synthetic_multigraph, origin=1, destination=3)
    assert res.is_valid is True
    assert res.route == [1, 3]
    assert res.edge_keys == [0]  # Selected key 0 (faster)


def test_dijkstra_avoids_closed_edges(synthetic_multigraph: nx.MultiDiGraph):
    """Test that if the faster highway (1->3) is closed, Dijkstra reroutes via the residential path (1->2->4)."""
    # Close highway edge (1 -> 3)
    traffic_states = {
        (1, 3): TrafficState(is_closed=True),
        (1, 3, 0): TrafficState(is_closed=True),
        (1, 3, 1): TrafficState(is_closed=True),
    }

    res = dijkstra_route(synthetic_multigraph, origin=1, destination=4, traffic_states=traffic_states)

    assert res.is_valid is True
    assert res.route == [1, 2, 4]  # Rerouted through open residential road
    assert res.total_distance_meters == 600.0

    # Route validator confirms route validity
    val = validate_route(synthetic_multigraph, res.route, origin=1, destination=4, traffic_states=traffic_states)
    assert val["is_valid"] is True


def test_dijkstra_respects_one_way_restrictions(synthetic_multigraph: nx.MultiDiGraph):
    """Test that traveling backwards against directed edges is impossible."""
    # Try routing from destination 4 back to origin 1 (all edges point forward)
    res = dijkstra_route(synthetic_multigraph, origin=4, destination=1)

    assert res.is_valid is False
    assert res.route == []
    assert res.total_cost == INFINITY_COST
    assert "unreachable" in res.metadata.get("error", "").lower()


def test_dijkstra_all_paths_closed_unreachable(synthetic_multigraph: nx.MultiDiGraph):
    """Test behavior when all outgoing routes from origin are closed."""
    traffic_states = {
        (1, 2): TrafficState(is_closed=True),
        (1, 3): TrafficState(is_closed=True),
        (1, 3, 0): TrafficState(is_closed=True),
        (1, 3, 1): TrafficState(is_closed=True),
    }

    res = dijkstra_route(synthetic_multigraph, origin=1, destination=4, traffic_states=traffic_states)
    assert res.is_valid is False
    assert res.route == []
    assert res.total_cost == INFINITY_COST


def test_dijkstra_origin_equals_destination(synthetic_multigraph: nx.MultiDiGraph):
    """Test trivial route where origin is the destination."""
    res = dijkstra_route(synthetic_multigraph, origin=2, destination=2)
    assert res.is_valid is True
    assert res.route == [2]
    assert res.total_cost == 0.0
    assert res.total_distance_meters == 0.0
    assert res.total_travel_time_sec == 0.0


def test_dijkstra_non_existent_node(synthetic_multigraph: nx.MultiDiGraph):
    """Test handling of non-existent node IDs."""
    res = dijkstra_route(synthetic_multigraph, origin=1, destination=9999)
    assert res.is_valid is False
    assert res.route == []
    assert "not present" in res.metadata.get("error", "")
