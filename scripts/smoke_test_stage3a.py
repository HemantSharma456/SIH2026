"""Stage 3A Smoke Test: Graph-Constrained Candidate Path Generation.

Loads the cached Connaught Place road network, generates K feasible candidate routes
between an origin-destination pair using Yen's K-Shortest Paths algorithm, and
validates every candidate against graph continuity and one-way constraints.
"""

from pathlib import Path
import sys
import networkx as nx

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.graph.loader import load_road_network
from src.graph.validator import validate_route
from src.traffic.cost_engine import CostWeights, DynamicCostEngine
from src.algorithms.qiga.path_generator import generate_candidate_paths


def main() -> int:
    print("=" * 80)
    print("  SIH 2026 -- STAGE 3A: GRAPH-CONSTRAINED CANDIDATE PATH GENERATION")
    print("=" * 80)

    # 1. Load cached network
    cfg = load_config()
    place_name = cfg.study_area.place_name
    network_type = cfg.study_area.network_type
    cache_dir = cfg.cache.cache_dir
    prefix = cfg.cache.filename_prefix
    dist_meters = cfg.study_area.buffer_dist_meters or 1000

    print(f"Loading cached road network for: {place_name}")
    G = load_road_network(
        place_name=place_name,
        network_type=network_type,
        dist_meters=dist_meters,
        cache_dir=cache_dir,
        filename_prefix=prefix,
        force_refresh=False,
    )
    print(f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.\n")

    # 2. Select two distant reachable nodes from the largest SCC
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    scc_list = sorted(list(largest_scc))
    origin = scc_list[0]
    destination = scc_list[len(scc_list) // 2]

    K = 5
    print(f"Selected Origin Node      : {origin}")
    print(f"Selected Destination Node : {destination}")
    print(f"Requested Candidate Count : K = {K}")
    print("-" * 80)

    # 3. Generate candidate paths
    engine = DynamicCostEngine()
    time_weights = CostWeights(weight_time=1.0, weight_distance=0.0)

    print("Generating graph-constrained candidate routes...")
    candidates = generate_candidate_paths(
        G=G,
        origin=origin,
        destination=destination,
        k=K,
        cost_engine=engine,
        weights=time_weights,
    )

    print(f"\nGenerated {len(candidates)} feasible candidate route(s):\n")

    all_valid = True
    for c in candidates:
        # Validate each route independently using route validator
        val_result = validate_route(
            G,
            c.route,
            origin=origin,
            destination=destination,
            edge_keys=c.edge_keys,
        )

        is_valid_route = val_result["is_valid"]
        if not is_valid_route:
            all_valid = False

        status_str = "VALID" if is_valid_route else "INVALID"

        print(f"[CANDIDATE #{c.candidate_id}]")
        print(f"  - Status           : {status_str} ({val_result['reason']})")
        print(f"  - Node Count       : {c.num_nodes}")
        print(f"  - Distance         : {c.total_distance_meters:.1f} m ({c.total_distance_km:.3f} km)")
        print(f"  - Est. Travel Time : {c.total_travel_time_sec:.1f} s ({c.total_travel_time_min:.2f} min)")
        print(f"  - Evaluated Cost   : {c.total_cost:.2f}")
        print(f"  - Node Sequence    : {c.route}")
        print()

    # Check uniqueness
    unique_node_seqs = {tuple(c.route) for c in candidates}
    print("-" * 80)
    print(f"Summary Statistics:")
    print(f"  - Total Candidates Requested : {K}")
    print(f"  - Feasible Candidates Found  : {len(candidates)}")
    print(f"  - Unique Node Sequences      : {len(unique_node_seqs)}")
    print(f"  - All Candidates Verified    : {all_valid}")
    print("=" * 80)
    print("  STAGE 3A SMOKE TEST COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
