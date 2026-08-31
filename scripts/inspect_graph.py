"""Inspection script for loading, validating, and reporting road network statistics."""

import argparse
import logging
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.graph.loader import load_road_network
from src.graph.validator import check_od_connectivity, get_graph_summary, validate_graph
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inspect_graph")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate an OSMnx road network.")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML.")
    parser.add_argument("--place", type=str, default=None, help="Override place name to download.")
    parser.add_argument("--force-refresh", action="store_true", help="Force redownload ignoring cache.")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating visualization image.")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    place_name = args.place or cfg.study_area.place_name
    network_type = cfg.study_area.network_type
    cache_dir = cfg.cache.cache_dir
    prefix = cfg.cache.filename_prefix

    print("=" * 70)
    print("  SIH 2026: ROAD NETWORK INSPECTION & VALIDATION")
    print("=" * 70)
    print(f"Target Study Area : {place_name}")
    print(f"Network Type       : {network_type}")
    print(f"Cache Directory    : {cache_dir}")
    print(f"Force Refresh      : {args.force_refresh}")
    print("-" * 70)

    try:
        # Load network
        logger.info("Loading road network...")
        dist_meters = cfg.study_area.buffer_dist_meters or 1000
        G = load_road_network(
            place_name=place_name,
            network_type=network_type,
            dist_meters=dist_meters,
            cache_dir=cache_dir,
            filename_prefix=prefix,
            force_refresh=args.force_refresh,
        )

        # Validate graph
        logger.info("Running structural and attribute validation...")
        val_result = validate_graph(G, require_coordinates=True, require_lengths=True)
        summary = get_graph_summary(G)

        # Print structured report
        print("\n" + "=" * 70)
        print("  GRAPH METRICS & STATISTICS")
        print("=" * 70)
        print(f"  - Total Nodes                     : {summary['num_nodes']}")
        print(f"  - Total Edges                     : {summary['num_edges']}")
        print(f"  - Graph Type                      : {summary['graph_type']}")
        print(f"  - Is Directed                     : {summary['is_directed']}")
        print(f"  - Is MultiGraph                   : {summary['is_multigraph']}")
        print(f"  - Total Road Length               : {summary['total_length_km']} km")
        print(f"  - Strongly Connected Components   : {summary['strongly_connected_components']} (Largest: {summary['largest_scc_node_count']} nodes)")
        print(f"  - Weakly Connected Components     : {summary['weakly_connected_components']} (Largest: {summary['largest_wcc_node_count']} nodes)")
        print(f"  - Speed Limit Coverage            : {summary['maxspeed_coverage_percent']}%")
        print(f"  - Geometry Coverage               : {summary['geometry_coverage_percent']}%")

        b = summary["bounds"]
        print("\n" + "-" * 70)
        print("  GEOGRAPHIC BOUNDING BOX")
        print("-" * 70)
        print(f"  - Latitude Range  : [{b['min_lat']:.6f}, {b['max_lat']:.6f}]")
        print(f"  - Longitude Range : [{b['min_lon']:.6f}, {b['max_lon']:.6f}]")

        print("\n" + "-" * 70)
        print("  ROAD HIERARCHY / HIGHWAY TYPES")
        print("-" * 70)
        for hw, count in sorted(summary["highway_type_distribution"].items(), key=lambda x: -x[1]):
            print(f"  - {hw:<28} : {count:>5} edges")

        print("\n" + "-" * 70)
        print("  ONE-WAY RESTRICTIONS")
        print("-" * 70)
        for ow, count in summary["oneway_distribution"].items():
            print(f"  - Oneway = {ow:<20} : {count:>5} edges")

        # Test sample OD connectivity in largest SCC
        scc_nodes = max(nx.strongly_connected_components(G), key=len)
        if len(scc_nodes) >= 2:
            scc_list = list(scc_nodes)
            origin_sample = scc_list[0]
            dest_sample = scc_list[-1]
            connected, msg = check_od_connectivity(G, origin_sample, dest_sample)
            print("\n" + "-" * 70)
            print("  SAMPLE OD CONNECTIVITY TEST")
            print("-" * 70)
            print(f"  - Origin Node      : {origin_sample}")
            print(f"  - Destination Node : {dest_sample}")
            print(f"  - Status           : {'CONNECTED' if connected else 'DISCONNECTED'}")
            print(f"  - Details          : {msg}")

        # Visualization export
        if cfg.visualization.export_plot and not args.no_plot:
            try:
                import osmnx as ox
                output_file = Path(cfg.visualization.output_image)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Generating graph visualization: {output_file}")
                fig, ax = ox.plot_graph(
                    G,
                    node_size=15,
                    node_color="#3388ff",
                    edge_color="#777777",
                    edge_linewidth=1.2,
                    bgcolor="#111111",
                    show=False,
                    close=True,
                    filepath=output_file,
                    save=True,
                )
                print(f"\n  - Saved plot to : {output_file}")
            except Exception as e:
                logger.warning(f"Could not generate plot: {e}")

        print("\n" + "=" * 70)
        print("  VALIDATION SUCCESSFUL: Graph is ready for Phase 2 routing.")
        print("=" * 70)
        return 0

    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
