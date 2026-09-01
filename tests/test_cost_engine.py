"""Unit tests for dynamic edge cost engine and traffic state modeling."""

import math
import pytest

from src.traffic.cost_engine import (
    CostWeights,
    DynamicCostEngine,
    TrafficState,
    INFINITY_COST,
    DEFAULT_HIGHWAY_SPEEDS_KMH,
)


@pytest.fixture
def cost_engine() -> DynamicCostEngine:
    return DynamicCostEngine()


def test_free_flow_travel_time(cost_engine: DynamicCostEngine):
    """Test travel time calculation under free-flow conditions (length / speed)."""
    # 500 meters at 50 km/h (13.8889 m/s) -> 36.0 seconds
    edge_data = {"length": 500.0, "maxspeed": "50"}
    res = cost_engine.compute_edge_cost(1, 2, edge_data)

    expected_speed_mps = 50.0 / 3.6
    expected_time_sec = 500.0 / expected_speed_mps

    assert res.is_closed is False
    assert res.distance_meters == 500.0
    assert math.isclose(res.speed_mps, expected_speed_mps, rel_tol=1e-3)
    assert math.isclose(res.travel_time_sec, expected_time_sec, rel_tol=1e-3)
    assert math.isclose(res.effective_cost, expected_time_sec, rel_tol=1e-3)  # default weight_time=1.0


def test_distance_calculation(cost_engine: DynamicCostEngine):
    """Test distance preservation and fallback for missing lengths."""
    edge_data = {"length": 1234.5, "highway": "residential"}
    res = cost_engine.compute_edge_cost(1, 2, edge_data)
    assert res.distance_meters == 1234.5

    # Missing length fallback to 1.0 meter
    res_missing = cost_engine.compute_edge_cost(1, 2, {})
    assert res_missing.distance_meters == 1.0


def test_speed_fallback_by_highway_type(cost_engine: DynamicCostEngine):
    """Test safe speed limit fallback according to OSM road hierarchy."""
    # Motorway -> 80 km/h
    res_motorway = cost_engine.compute_edge_cost(1, 2, {"length": 1000.0, "highway": "motorway"})
    assert math.isclose(res_motorway.free_flow_speed_mps, 80.0 / 3.6, rel_tol=1e-3)

    # Primary -> 50 km/h
    res_primary = cost_engine.compute_edge_cost(1, 2, {"length": 1000.0, "highway": "primary"})
    assert math.isclose(res_primary.free_flow_speed_mps, 50.0 / 3.6, rel_tol=1e-3)

    # Residential -> 25 km/h
    res_residential = cost_engine.compute_edge_cost(1, 2, {"length": 1000.0, "highway": "residential"})
    assert math.isclose(res_residential.free_flow_speed_mps, 25.0 / 3.6, rel_tol=1e-3)

    # Unknown highway type -> default 30 km/h
    res_unknown = cost_engine.compute_edge_cost(1, 2, {"length": 1000.0, "highway": "futuristic_hyperloop"})
    assert math.isclose(res_unknown.free_flow_speed_mps, 30.0 / 3.6, rel_tol=1e-3)


def test_malformed_and_mph_speed_parsing(cost_engine: DynamicCostEngine):
    """Test robust parsing of messy OSM speed strings (mph, lists, numbers)."""
    # 30 mph -> ~48.28 km/h -> 13.41 m/s
    res_mph = cost_engine.compute_edge_cost(1, 2, {"length": 1000.0, "maxspeed": "30 mph"})
    assert math.isclose(res_mph.free_flow_speed_mps, (30.0 * 1.60934) / 3.6, rel_tol=1e-3)

    # List of speeds: ['40', '50'] -> uses first 40 km/h
    res_list = cost_engine.compute_edge_cost(1, 2, {"length": 1000.0, "maxspeed": ["40", "50"]})
    assert math.isclose(res_list.free_flow_speed_mps, 40.0 / 3.6, rel_tol=1e-3)

    # Malformed string: 'none' or 'walk' -> fallback by highway
    res_malformed = cost_engine.compute_edge_cost(1, 2, {"length": 1000.0, "maxspeed": "signals", "highway": "secondary"})
    assert math.isclose(res_malformed.free_flow_speed_mps, 40.0 / 3.6, rel_tol=1e-3)


def test_congestion_and_incident_scaling(cost_engine: DynamicCostEngine):
    """Test dynamic traffic slowdown when congestion factor > 1.0."""
    edge_data = {"length": 1000.0, "maxspeed": "50"}  # ~72s free-flow
    free_flow = cost_engine.compute_edge_cost(1, 2, edge_data)

    # 2.0x Congestion Factor -> speed halved, travel time doubled
    state_congested = TrafficState(congestion_factor=2.0)
    res_congested = cost_engine.compute_edge_cost(1, 2, edge_data, traffic_state=state_congested)

    assert math.isclose(res_congested.travel_time_sec, free_flow.travel_time_sec * 2.0, rel_tol=1e-3)
    assert math.isclose(res_congested.speed_mps, free_flow.speed_mps / 2.0, rel_tol=1e-3)

    # Incident multiplier also further scales slowdown
    state_incident = TrafficState(congestion_factor=2.0, incident_multiplier=1.5)
    res_incident = cost_engine.compute_edge_cost(1, 2, edge_data, traffic_state=state_incident)
    assert math.isclose(res_incident.travel_time_sec, free_flow.travel_time_sec * 3.0, rel_tol=1e-3)


def test_explicit_live_speed_override(cost_engine: DynamicCostEngine):
    """Test that live observed speed directly overrides static free-flow speed."""
    edge_data = {"length": 1000.0, "maxspeed": "80"}
    state_live = TrafficState(current_speed_mps=5.0)  # Heavy jam: 5 m/s (18 km/h)
    res = cost_engine.compute_edge_cost(1, 2, edge_data, traffic_state=state_live)

    assert res.speed_mps == 5.0
    assert res.travel_time_sec == 200.0  # 1000m / 5m/s


def test_closed_edge_behavior(cost_engine: DynamicCostEngine):
    """Test that closed roads return infinite cost and is_closed=True."""
    edge_data = {"length": 500.0, "maxspeed": "50"}

    # Closure via TrafficState
    state_closed = TrafficState(is_closed=True)
    res_state = cost_engine.compute_edge_cost(1, 2, edge_data, traffic_state=state_closed)
    assert res_state.is_closed is True
    assert res_state.effective_cost == INFINITY_COST

    # Closure via edge attribute
    edge_data_closed = {"length": 500.0, "closed": True}
    res_attr = cost_engine.compute_edge_cost(1, 2, edge_data_closed)
    assert res_attr.is_closed is True
    assert res_attr.effective_cost == INFINITY_COST


def test_custom_cost_weights(cost_engine: DynamicCostEngine):
    """Test weighted combination of time, distance, and congestion delay."""
    edge_data = {"length": 2000.0, "maxspeed": "72"}  # 72 km/h = 20 m/s -> free-flow = 100s
    state = TrafficState(congestion_factor=2.0)  # congested time = 200s, delay = 100s

    weights = CostWeights(
        weight_time=0.5,       # 0.5 * 200 = 100
        weight_distance=10.0,  # 10 * 2.0 km = 20
        weight_congestion=0.2, # 0.2 * 100 delay = 20
    )
    # Expected cost = 100 + 20 + 20 = 140

    res = cost_engine.compute_edge_cost(1, 2, edge_data, traffic_state=state, weights=weights)
    assert math.isclose(res.effective_cost, 140.0, rel_tol=1e-3)
    assert res.travel_time_sec == 200.0
    assert res.distance_meters == 2000.0
