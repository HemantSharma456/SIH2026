"""Unit tests for graph-constrained candidate path generator (Stage 3A)."""

import pytest
import networkx as nx

from src.algorithms.qiga.path_generator import (
    CandidateRoute,
    GraphConstrainedPathGenerator,
    generate_candidate_paths,
)
from src.graph.validator import validate_route
from src.traffic.cost_engine import CostWeights, DynamicCostEngine, TrafficState


@pytest.fixture
def multi_path_graph() -> nx.MultiDiGraph:
    """Create a synthetic directed graph with 4 distinct paths from 1 to 6:
    Path A: 1 -> 2 -> 6 (length = 200 + 200 = 400m, speed = 50 km/h)
    Path B: 1 -> 3 -> 6 (length = 150 + 350 = 500m, speed = 40 km/h)
    Path C: 1 -> 4 -> 5 -> 6 (length = 100 + 100 + 100 = 300m, speed = 20 km/h)
    Path D: 1 -> 2 -> 3 -> 6 (length = 200 + 100 + 350 = 650m, speed = 40 km/h)
    """
    G = nx.MultiDiGraph()
    for n in range(1, 7):
        G.add_node(n, x=77.0 + n * 0.01, y=28.0 + n * 0.01)

    # Path A edges
    G.add_edge(1, 2, key=0, length=200.0, maxspeed="50", highway="primary")
    G.add_edge(2, 6, key=0, length=200.0, maxspeed="50", highway="primary")

    # Path B edges
    G.add_edge(1, 3, key=0, length=150.0, maxspeed="40", highway="secondary")
    G.add_edge(3, 6, key=0, length=350.0, maxspeed="40", highway="secondary")

    # Path C edges
    G.add_edge(1, 4, key=0, length=100.0, maxspeed="20", highway="residential")
    G.add_edge(4, 5, key=0, length=100.0, maxspeed="20", highway="residential")
    G.add_edge(5, 6, key=0, length=100.0, maxspeed="20", highway="residential")

    # Cross edge for Path D
    G.add_edge(2, 3, key=0, length=100.0, maxspeed="30", highway="tertiary")

    return G


@pytest.fixture
def single_path_graph() -> nx.DiGraph:
    """Create a simple linear directed graph: 1 -> 2 -> 3."""
    G = nx.DiGraph()
    G.add_node(1, x=77.0, y=28.0)
    G.add_node(2, x=77.1, y=28.1)
    G.add_node(3, x=77.2, y=28.2)
    G.add_edge(1, 2, length=100.0, maxspeed="40", highway="secondary")
    G.add_edge(2, 3, length=100.0, maxspeed="40", highway="secondary")
    return G


def test_single_valid_path_exists(single_path_graph: nx.DiGraph):
    """Test generating candidate path when exactly one route exists."""
    candidates = generate_candidate_paths(single_path_graph, origin=1, destination=3, k=5)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.route == [1, 2, 3]
    assert c.is_valid is True
    assert c.total_distance_meters == 200.0


def test_multiple_candidate_paths_generated(multi_path_graph: nx.MultiDiGraph):
    """Test generating up to K=4 distinct candidate paths."""
    candidates = generate_candidate_paths(multi_path_graph, origin=1, destination=6, k=4)
    assert len(candidates) == 4
    # All candidates must have unique node sequences
    unique_routes = {tuple(c.route) for c in candidates}
    assert len(unique_routes) == 4


def test_every_candidate_starts_at_origin_and_ends_at_destination(multi_path_graph: nx.MultiDiGraph):
    """Test that all candidates strictly connect origin to destination."""
    candidates = generate_candidate_paths(multi_path_graph, origin=1, destination=6, k=4)
    for c in candidates:
        assert c.origin == 1
        assert c.destination == 6
        assert c.route[0] == 1
        assert c.route[-1] == 6


def test_every_candidate_follows_valid_directed_edges(multi_path_graph: nx.MultiDiGraph):
    """Test that every consecutive step corresponds to a real directed edge in G."""
    candidates = generate_candidate_paths(multi_path_graph, origin=1, destination=6, k=4)
    for c in candidates:
        val = validate_route(multi_path_graph, c.route, origin=1, destination=6, edge_keys=c.edge_keys)
        assert val["is_valid"] is True


def test_one_way_restrictions_respected(multi_path_graph: nx.MultiDiGraph):
    """Test that routing against one-way directed edges produces no candidates."""
    # Edges go 1 -> 6, not 6 -> 1
    candidates = generate_candidate_paths(multi_path_graph, origin=6, destination=1, k=5)
    assert len(candidates) == 0


def test_closed_edges_are_avoided(multi_path_graph: nx.MultiDiGraph):
    """Test that closed roads are never present in any generated candidate path."""
    # Close edge 1 -> 2 (blocks Path A and Path D)
    traffic_states = {
        (1, 2): TrafficState(is_closed=True),
        (1, 2, 0): TrafficState(is_closed=True),
    }

    candidates = generate_candidate_paths(
        multi_path_graph,
        origin=1,
        destination=6,
        k=4,
        traffic_states=traffic_states,
    )

    # Remaining paths: Path B (1->3->6) and Path C (1->4->5->6)
    assert len(candidates) == 2
    for c in candidates:
        assert 2 not in c.route  # Node 2 was only reachable via closed edge (1, 2)
        val = validate_route(multi_path_graph, c.route, traffic_states=traffic_states)
        assert val["is_valid"] is True


def test_unreachable_destination_returns_empty_list(single_path_graph: nx.DiGraph):
    """Test that disconnected nodes cleanly return an empty list."""
    # Add an isolated node 99
    single_path_graph.add_node(99, x=77.9, y=28.9)
    candidates = generate_candidate_paths(single_path_graph, origin=1, destination=99, k=5)
    assert candidates == []


def test_invalid_origin_or_destination_handled(multi_path_graph: nx.MultiDiGraph):
    """Test non-existent node IDs."""
    assert generate_candidate_paths(multi_path_graph, origin=999, destination=6, k=3) == []
    assert generate_candidate_paths(multi_path_graph, origin=1, destination=999, k=3) == []


def test_k_larger_than_available_paths(single_path_graph: nx.DiGraph):
    """Test when K=10 requested but only 1 feasible path exists."""
    candidates = generate_candidate_paths(single_path_graph, origin=1, destination=3, k=10)
    assert len(candidates) == 1
    assert candidates[0].route == [1, 2, 3]


def test_no_duplicate_candidate_routes(multi_path_graph: nx.MultiDiGraph):
    """Test that candidate generator never returns duplicate route sequences."""
    candidates = generate_candidate_paths(multi_path_graph, origin=1, destination=6, k=10)
    route_tuples = [tuple(c.route) for c in candidates]
    assert len(route_tuples) == len(set(route_tuples))


def test_multigraph_parallel_edges_handled():
    """Test that parallel edges with different attributes are handled correctly."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=77.0, y=28.0)
    G.add_node(2, x=77.1, y=28.1)

    # Parallel edges: key 0 (fast 80 km/h) and key 1 (slow 20 km/h)
    G.add_edge(1, 2, key=0, length=500.0, maxspeed="80", highway="motorway")
    G.add_edge(1, 2, key=1, length=500.0, maxspeed="20", highway="residential")

    candidates = generate_candidate_paths(G, origin=1, destination=2, k=2)
    assert len(candidates) == 1  # 1 distinct node sequence
    assert candidates[0].edge_keys == [0]  # Selected key 0 (faster)


def test_candidate_metrics_calculated_correctly(multi_path_graph: nx.MultiDiGraph):
    """Test that total cost, distance, and travel time are non-zero and positive."""
    candidates = generate_candidate_paths(multi_path_graph, origin=1, destination=6, k=3)
    for c in candidates:
        assert c.total_distance_meters > 0.0
        assert c.total_travel_time_sec > 0.0
        assert c.total_cost > 0.0
        assert c.num_nodes >= 3


def test_candidate_generation_with_custom_cost_weights(multi_path_graph: nx.MultiDiGraph):
    """Test that changing cost weights reorders candidates by objective."""
    # Distance-weighted objective: Path C (300m) should be ranked #1
    dist_weights = CostWeights(weight_time=0.0, weight_distance=1.0)
    candidates_dist = generate_candidate_paths(
        multi_path_graph, origin=1, destination=6, k=3, weights=dist_weights
    )

    # Path C is 1 -> 4 -> 5 -> 6 (300m)
    assert candidates_dist[0].route == [1, 4, 5, 6]
    assert candidates_dist[0].total_distance_meters == 300.0

    # Time-weighted objective: Path A (400m at 50 km/h = 28.8s) should be ranked #1
    time_weights = CostWeights(weight_time=1.0, weight_distance=0.0)
    candidates_time = generate_candidate_paths(
        multi_path_graph, origin=1, destination=6, k=3, weights=time_weights
    )
    # Path A is 1 -> 2 -> 6 (28.8s vs Path C 54s)
    assert candidates_time[0].route == [1, 2, 6]
