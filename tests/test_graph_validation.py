"""Unit tests for graph validation, connectivity checks, and configuration."""

import pytest
import networkx as nx
from pathlib import Path

from src.config import load_config, AppConfig
from src.graph.validator import (
    validate_graph,
    check_od_connectivity,
    get_graph_summary,
    GraphValidationError,
)


@pytest.fixture
def valid_synthetic_graph() -> nx.DiGraph:
    """Create a minimal valid directed road network graph."""
    G = nx.DiGraph()
    # 4 nodes in a diamond shape with coordinates (x=lon, y=lat)
    G.add_node(1, x=77.2090, y=28.6328)
    G.add_node(2, x=77.2190, y=28.6428)
    G.add_node(3, x=77.1990, y=28.6228)
    G.add_node(4, x=77.2290, y=28.6528)

    # Directed edges with length and attributes
    G.add_edge(1, 2, length=120.5, highway="primary", maxspeed="50", oneway=True)
    G.add_edge(2, 4, length=150.0, highway="secondary", maxspeed="40", oneway=True)
    G.add_edge(1, 3, length=90.0, highway="residential", oneway=False)
    G.add_edge(3, 4, length=200.0, highway="primary", maxspeed="60", oneway=False)
    G.add_edge(4, 1, length=250.0, highway="primary", oneway=True)  # Cycle to make it strongly connected

    return G


def test_valid_graph_passes_validation(valid_synthetic_graph: nx.DiGraph):
    """Test that a well-formed directed graph passes validation."""
    result = validate_graph(valid_synthetic_graph)
    assert result["is_valid"] is True
    assert result["num_nodes"] == 4
    assert result["num_edges"] == 5
    assert result["is_directed"] is True
    assert result["num_strongly_connected_components"] == 1
    assert result["largest_scc_size"] == 4


def test_empty_graph_fails_validation():
    """Test that an empty graph raises GraphValidationError."""
    G = nx.DiGraph()
    with pytest.raises(GraphValidationError, match="Graph is empty: 0 nodes found"):
        validate_graph(G)


def test_undirected_graph_fails_validation():
    """Test that an undirected graph is rejected for traffic routing."""
    G = nx.Graph()
    G.add_node(1, x=77.0, y=28.0)
    G.add_node(2, x=77.1, y=28.1)
    G.add_edge(1, 2, length=100.0)

    with pytest.raises(GraphValidationError, match="Road network must be a directed graph"):
        validate_graph(G)


def test_missing_node_coordinates_fails():
    """Test that missing coordinates trigger validation error."""
    G = nx.DiGraph()
    G.add_node(1, x=77.0, y=28.0)
    G.add_node(2)  # missing coordinates
    G.add_edge(1, 2, length=50.0)

    with pytest.raises(GraphValidationError, match="missing spatial coordinates"):
        validate_graph(G, require_coordinates=True)


def test_missing_edge_length_fails():
    """Test that edges without 'length' attribute trigger validation error."""
    G = nx.DiGraph()
    G.add_node(1, x=77.0, y=28.0)
    G.add_node(2, x=77.1, y=28.1)
    G.add_edge(1, 2)  # missing length

    with pytest.raises(GraphValidationError, match="missing valid 'length' attributes"):
        validate_graph(G, require_lengths=True)


def test_non_positive_edge_length_fails():
    """Test that non-positive lengths (<= 0) are rejected."""
    G = nx.DiGraph()
    G.add_node(1, x=77.0, y=28.0)
    G.add_node(2, x=77.1, y=28.1)
    G.add_edge(1, 2, length=-10.0)

    with pytest.raises(GraphValidationError, match="non-positive length"):
        validate_graph(G, require_lengths=True)


def test_check_od_connectivity_success(valid_synthetic_graph: nx.DiGraph):
    """Test OD connectivity between reachable nodes."""
    connected, msg = check_od_connectivity(valid_synthetic_graph, origin=1, destination=4)
    assert connected is True
    assert "Directed path exists" in msg


def test_check_od_connectivity_unreachable():
    """Test OD connectivity when one-way restrictions prevent a path."""
    G = nx.DiGraph()
    G.add_node("A", x=77.0, y=28.0)
    G.add_node("B", x=77.1, y=28.1)
    G.add_edge("A", "B", length=100.0)  # Only A -> B, no B -> A

    connected, msg = check_od_connectivity(G, origin="B", destination="A")
    assert connected is False
    assert "No directed path exists" in msg


def test_check_od_connectivity_missing_node(valid_synthetic_graph: nx.DiGraph):
    """Test OD connectivity with non-existent node."""
    connected, msg = check_od_connectivity(valid_synthetic_graph, origin=1, destination=999)
    assert connected is False
    assert "Destination node '999' does not exist" in msg


def test_check_od_connectivity_same_node(valid_synthetic_graph: nx.DiGraph):
    """Test OD connectivity when origin and destination are identical."""
    connected, msg = check_od_connectivity(valid_synthetic_graph, origin=1, destination=1)
    assert connected is True
    assert "identical" in msg


def test_get_graph_summary(valid_synthetic_graph: nx.DiGraph):
    """Test summary extraction and metric computation."""
    summary = get_graph_summary(valid_synthetic_graph)
    assert summary["num_nodes"] == 4
    assert summary["num_edges"] == 5
    assert summary["is_directed"] is True
    assert summary["bounds"]["min_lat"] == 28.6228
    assert summary["bounds"]["max_lat"] == 28.6528
    assert summary["total_length_km"] == pytest.approx(0.8105, rel=1e-3)
    assert "primary" in summary["highway_type_distribution"]


def test_config_loader(tmp_path: Path):
    """Test YAML config loading and fallback to defaults."""
    # Test fallback
    cfg = load_config(tmp_path / "non_existent.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.study_area.place_name == "Connaught Place, New Delhi, India"
    assert cfg.study_area.network_type == "drive"
    assert cfg.cache.enabled is True

    # Test custom YAML
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
study_area:
  place_name: "Indira Gandhi International Airport, New Delhi"
  network_type: "drive"
cache:
  enabled: false
  cache_dir: "custom_cache"
"""
    )
    custom_cfg = load_config(custom_yaml)
    assert custom_cfg.study_area.place_name == "Indira Gandhi International Airport, New Delhi"
    assert custom_cfg.cache.enabled is False
    assert custom_cfg.cache.cache_dir == "custom_cache"
