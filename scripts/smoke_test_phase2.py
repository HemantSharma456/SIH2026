"""Phase 2 Smoke Test: Dynamic Edge Cost Engine + Dijkstra Baseline.

Loads the cached Connaught Place road network, evaluates routes under
free-flow, distance-optimal, and congested/incident conditions, and validates
route continuity and one-way compliance.
"""

from pathlib import Path
import sys
import networkx as nx

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.graph.loader import load_road_network
from src.graph.validator import validate_route
from src.traffic.cost_engine import CostWeights, DynamicCostEngine, TrafficState
from src.algorithms.baselines.dijkstra import DijkstraRouter


def main() -> int:
    print("=" * 75)
    print("  SIH 2026 -- PHASE 2: DYNAMIC EDGE COST & DIJKSTRA ROUTING SMOKE TEST")
    print("=" * 75)

    # 1. Load cached road graph
    cfg = load_config()
    place_name = cfg.study_area.place_name
    network_type = cfg.study_area.network_type
    cache_dir = cfg.cache.cache_dir
    prefix = cfg.cache.filename_prefix
    dist_meters = cfg.study_area.buffer_dist_meters or 1000

    print(f"Loading cached network for: {place_name}")
    G = load_road_network(
        place_name=place_name,
        network_type=network_type,
        dist_meters=dist_meters,
        cache_dir=cache_dir,
        filename_prefix=prefix,
        force_refresh=False,
    )
    print(f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.\n")

    # 2. Select two reachable nodes from the largest Strongly Connected Component (SCC)
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    scc_list = sorted(list(largest_scc))
    # Pick two distant nodes in the component
    origin = scc_list[0]
    destination = scc_list[len(scc_list) // 2]

    print(f"Selected Origin Node      : {origin}")
    print(f"Selected Destination Node : {destination}")
    print("-" * 75)

    engine = DynamicCostEngine()
    router = DijkstraRouter(cost_engine=engine)

    # ---------------------------------------------------------
    # Scenario 1: Fastest Path (Travel Time Weighted)
    # ---------------------------------------------------------
    time_weights = CostWeights(weight_time=1.0, weight_distance=0.0, weight_congestion=0.0)
    res_time = router.solve(G, origin=origin, destination=destination, weights=time_weights)

    val_time = validate_route(G, res_time.route, origin=origin, destination=destination, edge_keys=res_time.edge_keys)

    print("\n[SCENARIO 1: Time-Optimal / Fastest Route (Free-Flow)]")
    print(f"  - Route Validity        : {'VALID' if val_time['is_valid'] else 'INVALID'} ({val_time['reason']})")
    print(f"  - Number of Route Nodes : {res_time.num_nodes}")
    print(f"  - Total Distance        : {res_time.total_distance_meters:.1f} m ({res_time.total_distance_km:.3f} km)")
    print(f"  - Est. Travel Time      : {res_time.total_travel_time_sec:.1f} s ({res_time.total_travel_time_sec/60:.2f} min)")
    print(f"  - Evaluated Cost        : {res_time.total_cost:.2f}")
    print(f"  - Execution Runtime     : {res_time.execution_time_ms:.3f} ms")
    print(f"  - Node Sequence         : {res_time.route}")

    # ---------------------------------------------------------
    # Scenario 2: Shortest Distance Path
    # ---------------------------------------------------------
    dist_weights = CostWeights(weight_time=0.0, weight_distance=1.0, weight_congestion=0.0)
    res_dist = router.solve(G, origin=origin, destination=destination, weights=dist_weights)

    val_dist = validate_route(G, res_dist.route, origin=origin, destination=destination, edge_keys=res_dist.edge_keys)

    print("\n[SCENARIO 2: Distance-Optimal / Shortest Path]")
    print(f"  - Route Validity        : {'VALID' if val_dist['is_valid'] else 'INVALID'} ({val_dist['reason']})")
    print(f"  - Number of Route Nodes : {res_dist.num_nodes}")
    print(f"  - Total Distance        : {res_dist.total_distance_meters:.1f} m ({res_dist.total_distance_km:.3f} km)")
    print(f"  - Est. Travel Time      : {res_dist.total_travel_time_sec:.1f} s ({res_dist.total_travel_time_sec/60:.2f} min)")
    print(f"  - Evaluated Cost        : {res_dist.total_cost:.3f}")
    print(f"  - Execution Runtime     : {res_dist.execution_time_ms:.3f} ms")
    print(f"  - Node Sequence         : {res_dist.route}")

    # ---------------------------------------------------------
    # Scenario 3: Dynamic Incident / Road Closure Rerouting
    # ---------------------------------------------------------
    # Simulate a road closure on the first major hop of the fastest route
    if len(res_time.route) >= 3:
        blocked_u = res_time.route[1]
        blocked_v = res_time.route[2]
        print(f"\n[SCENARIO 3: Dynamic Incident Simulation - Road Closure at Edge ({blocked_u} -> {blocked_v})]")

        incident_states = {
            (blocked_u, blocked_v): TrafficState(is_closed=True),
        }

        res_reroute = router.solve(
            G,
            origin=origin,
            destination=destination,
            traffic_states=incident_states,
            weights=time_weights,
        )

        val_reroute = validate_route(
            G,
            res_reroute.route,
            origin=origin,
            destination=destination,
            edge_keys=res_reroute.edge_keys,
            traffic_states=incident_states,
        )

        print(f"  - Route Validity        : {'VALID' if val_reroute['is_valid'] else 'INVALID'} ({val_reroute['reason']})")
        print(f"  - Number of Route Nodes : {res_reroute.num_nodes}")
        print(f"  - Total Distance        : {res_reroute.total_distance_meters:.1f} m ({res_reroute.total_distance_km:.3f} km)")
        print(f"  - Est. Travel Time      : {res_reroute.total_travel_time_sec:.1f} s ({res_reroute.total_travel_time_sec/60:.2f} min)")
        print(f"  - Evaluated Cost        : {res_reroute.total_cost:.2f}")
        print(f"  - Closed Edge Avoided   : {not any((u == blocked_u and v == blocked_v) for u, v in zip(res_reroute.route[:-1], res_reroute.route[1:]))}")
        print(f"  - Node Sequence         : {res_reroute.route}")

    print("\n" + "=" * 75)
    print("  PHASE 2 SMOKE TEST COMPLETED SUCCESSFULLY.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
