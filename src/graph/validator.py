"""Validation and diagnostic routines for road network graphs."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx


class GraphValidationError(Exception):
    """Raised when a road network graph fails structural or attribute validation."""
    pass


def validate_graph(G: nx.Graph, require_coordinates: bool = True, require_lengths: bool = True) -> Dict[str, Any]:
    """Validate that a road graph meets the requirements for traffic routing.

    Checks:
    1. Graph is not empty (contains at least one node and one edge).
    2. Graph is directed (nx.DiGraph or nx.MultiDiGraph) to support one-way road modeling.
    3. Nodes contain geographic coordinates (x/y or lon/lat).
    4. Edges contain positive 'length' attribute where available.
    5. Connectivity metrics (strongly and weakly connected components).

    Args:
        G: NetworkX graph to validate.
        require_coordinates: If True, raises error if any node lacks coordinates.
        require_lengths: If True, raises error if any edge lacks length or has invalid length.

    Returns:
        Dictionary of validation results and metrics.

    Raises:
        GraphValidationError: If any critical validation check fails.
    """
    if G is None:
        raise GraphValidationError("Graph object is None.")

    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise GraphValidationError(f"Expected NetworkX graph instance, got {type(G).__name__}.")

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        raise GraphValidationError("Graph is empty: 0 nodes found.")
    if num_edges == 0:
        raise GraphValidationError("Graph is empty: 0 edges found.")

    if not G.is_directed():
        raise GraphValidationError(
            f"Road network must be a directed graph to respect one-way traffic restrictions, got {type(G).__name__}."
        )

    # Check node coordinates
    nodes_missing_coords: List[Any] = []
    for node, data in G.nodes(data=True):
        has_xy = ("x" in data and "y" in data)
        has_lonlat = ("lon" in data and "lat" in data)
        if not (has_xy or has_lonlat):
            nodes_missing_coords.append(node)

    if nodes_missing_coords and require_coordinates:
        sample_missing = nodes_missing_coords[:5]
        raise GraphValidationError(
            f"{len(nodes_missing_coords)} nodes are missing spatial coordinates (x, y / lon, lat). "
            f"Sample node IDs: {sample_missing}"
        )

    # Check edge length attributes
    edges_missing_length: List[Tuple[Any, Any]] = []
    edges_non_positive_length: List[Tuple[Any, Any, float]] = []

    for u, v, *rest in G.edges(data=True):
        data = rest[-1] if rest else {}
        if "length" not in data:
            edges_missing_length.append((u, v))
        else:
            try:
                length_val = float(data["length"])
                if length_val <= 0:
                    edges_non_positive_length.append((u, v, length_val))
            except (ValueError, TypeError):
                edges_missing_length.append((u, v))

    if edges_missing_length and require_lengths:
        sample_missing = edges_missing_length[:5]
        raise GraphValidationError(
            f"{len(edges_missing_length)} edges are missing valid 'length' attributes. "
            f"Sample edges: {sample_missing}"
        )

    if edges_non_positive_length and require_lengths:
        sample_invalid = edges_non_positive_length[:5]
        raise GraphValidationError(
            f"{len(edges_non_positive_length)} edges have non-positive length. "
            f"Sample edges: {sample_invalid}"
        )

    # Connectivity metrics
    num_scc = nx.number_strongly_connected_components(G)
    scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
    largest_scc_size = max(scc_sizes) if scc_sizes else 0

    num_wcc = nx.number_weakly_connected_components(G)
    wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
    largest_wcc_size = max(wcc_sizes) if wcc_sizes else 0

    return {
        "is_valid": True,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_directed": G.is_directed(),
        "is_multigraph": G.is_multigraph(),
        "nodes_missing_coords_count": len(nodes_missing_coords),
        "edges_missing_length_count": len(edges_missing_length),
        "num_strongly_connected_components": num_scc,
        "largest_scc_size": largest_scc_size,
        "num_weakly_connected_components": num_wcc,
        "largest_wcc_size": largest_wcc_size,
    }


def check_od_connectivity(
    G: nx.DiGraph | nx.MultiDiGraph,
    origin: Any,
    destination: Any,
) -> Tuple[bool, Optional[str]]:
    """Verify if a directed path exists between origin and destination nodes.

    Args:
        G: Directed NetworkX road graph.
        origin: Origin node identifier.
        destination: Destination node identifier.

    Returns:
        Tuple of (is_connected: bool, message: Optional[str]).
    """
    if origin not in G:
        return False, f"Origin node '{origin}' does not exist in the graph."
    if destination not in G:
        return False, f"Destination node '{destination}' does not exist in the graph."
    if origin == destination:
        return True, "Origin and destination are identical."

    try:
        has_route = nx.has_path(G, origin, destination)
        if has_route:
            return True, f"Directed path exists from {origin} to {destination}."
        else:
            return False, f"No directed path exists from {origin} to {destination} (one-way restriction or disconnected components)."
    except Exception as e:
        return False, f"Connectivity check failed with error: {str(e)}"


def get_graph_summary(G: nx.Graph) -> Dict[str, Any]:
    """Extract descriptive statistics and geographic boundaries from a road graph.

    Args:
        G: NetworkX graph.

    Returns:
        Dictionary containing spatial bounds, attribute coverage, and road type distributions.
    """
    validation_info = validate_graph(G, require_coordinates=False, require_lengths=False)

    # Compute bounding box
    lats: List[float] = []
    lons: List[float] = []
    for _, data in G.nodes(data=True):
        if "y" in data:
            lats.append(data["y"])
        elif "lat" in data:
            lats.append(data["lat"])

        if "x" in data:
            lons.append(data["x"])
        elif "lon" in data:
            lons.append(data["lon"])

    bounds = {
        "min_lat": min(lats) if lats else None,
        "max_lat": max(lats) if lats else None,
        "min_lon": min(lons) if lons else None,
        "max_lon": max(lons) if lons else None,
    }

    # Edge attribute analysis
    total_length_meters = 0.0
    highway_counts: Dict[str, int] = {}
    edges_with_maxspeed = 0
    edges_with_geometry = 0
    oneway_counts: Dict[str, int] = {}

    for u, v, *rest in G.edges(data=True):
        data = rest[-1] if rest else {}
        if "length" in data:
            try:
                total_length_meters += float(data["length"])
            except (ValueError, TypeError):
                pass

        if "highway" in data:
            hw = data["highway"]
            if isinstance(hw, list):
                hw_str = "/".join(str(h) for h in hw)
            else:
                hw_str = str(hw)
            highway_counts[hw_str] = highway_counts.get(hw_str, 0) + 1

        if "maxspeed" in data and data["maxspeed"] is not None:
            edges_with_maxspeed += 1

        if "geometry" in data and data["geometry"] is not None:
            edges_with_geometry += 1

        if "oneway" in data:
            ow_str = str(data["oneway"])
            oneway_counts[ow_str] = oneway_counts.get(ow_str, 0) + 1

    num_edges = G.number_of_edges()
    speed_coverage_pct = (edges_with_maxspeed / num_edges * 100.0) if num_edges > 0 else 0.0
    geometry_coverage_pct = (edges_with_geometry / num_edges * 100.0) if num_edges > 0 else 0.0

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": num_edges,
        "graph_type": type(G).__name__,
        "is_directed": G.is_directed(),
        "is_multigraph": G.is_multigraph(),
        "bounds": bounds,
        "total_length_km": round(total_length_meters / 1000.0, 3),
        "highway_type_distribution": highway_counts,
        "maxspeed_coverage_percent": round(speed_coverage_pct, 2),
        "geometry_coverage_percent": round(geometry_coverage_pct, 2),
        "oneway_distribution": oneway_counts,
        "strongly_connected_components": validation_info["num_strongly_connected_components"],
        "largest_scc_node_count": validation_info["largest_scc_size"],
        "weakly_connected_components": validation_info["num_weakly_connected_components"],
        "largest_wcc_node_count": validation_info["largest_wcc_size"],
    }


def validate_route(
    G: Union[nx.DiGraph, nx.MultiDiGraph],
    route: List[Any],
    origin: Optional[Any] = None,
    destination: Optional[Any] = None,
    edge_keys: Optional[List[Any]] = None,
    traffic_states: Optional[Dict[Tuple[Any, Any], Any]] = None,
) -> Dict[str, Any]:
    """Validate that a candidate route is continuous, unblocked, and respects one-way streets.

    Checks:
    1. Route is a non-empty sequence of nodes.
    2. Starts at expected origin and ends at expected destination.
    3. Every consecutive pair (u, v) is a valid directed edge in G.
    4. No edges along the route are closed (in attributes or traffic_states).
    5. Computes total physical distance along the route.

    Args:
        G: Directed road graph.
        route: Sequence of node IDs.
        origin: Expected start node (optional check).
        destination: Expected end node (optional check).
        edge_keys: Optional sequence of edge keys for MultiDiGraph edges.
        traffic_states: Optional dynamic traffic conditions.

    Returns:
        Dictionary containing is_valid, reason, total_distance_meters, and step_details.
    """
    if not route or len(route) == 0:
        return {"is_valid": False, "reason": "Route is empty.", "total_distance_meters": 0.0}

    if origin is not None and route[0] != origin:
        return {
            "is_valid": False,
            "reason": f"Route starts at {route[0]}, expected origin {origin}.",
            "total_distance_meters": 0.0,
        }

    if destination is not None and route[-1] != destination:
        return {
            "is_valid": False,
            "reason": f"Route ends at {route[-1]}, expected destination {destination}.",
            "total_distance_meters": 0.0,
        }

    # Single node path
    if len(route) == 1:
        if route[0] not in G:
            return {"is_valid": False, "reason": f"Node {route[0]} does not exist in graph.", "total_distance_meters": 0.0}
        return {"is_valid": True, "reason": "Single-node path.", "total_distance_meters": 0.0}

    traffic_states = traffic_states or {}
    total_dist = 0.0
    steps = []

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]

        if not G.has_edge(u, v):
            return {
                "is_valid": False,
                "reason": f"Disconnected step or one-way violation: no directed edge from {u} to {v}.",
                "total_distance_meters": total_dist,
            }

        # Handle edge attributes
        if G.is_multigraph():
            edge_dict = G[u][v]
            key = edge_keys[i] if (edge_keys and i < len(edge_keys)) else next(iter(edge_dict.keys()))
            data = edge_dict.get(key, next(iter(edge_dict.values())))
        else:
            key = 0
            data = G[u][v]

        # Check closure
        state = traffic_states.get((u, v, key)) or traffic_states.get((u, v))
        is_closed = (state and state.is_closed) or data.get("closed", False) is True
        if is_closed:
            return {
                "is_valid": False,
                "reason": f"Route uses closed edge ({u} -> {v}, key={key}).",
                "total_distance_meters": total_dist,
            }

        edge_len = float(data.get("length", 0.0))
        total_dist += edge_len
        steps.append({"u": u, "v": v, "key": key, "length": edge_len})

    return {
        "is_valid": True,
        "reason": "Route is a continuous, valid directed walk with all edges open.",
        "total_distance_meters": round(total_dist, 2),
        "steps_count": len(steps),
        "step_details": steps,
    }

