"""Road network graph loader with local caching and attribute preservation."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import networkx as nx
import osmnx as ox

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Convert a place name or string to a filesystem-safe filename."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_").lower()


def load_road_network(
    place_name: str = "Connaught Place, New Delhi, India",
    network_type: str = "drive",
    dist_meters: int = 1000,
    cache_dir: str = "data/cache",
    filename_prefix: Optional[str] = None,
    force_refresh: bool = False,
    simplify: bool = True,
) -> nx.MultiDiGraph:
    """Download or load a cached drivable road network using OSMnx.

    Features:
    - Downloads drivable road network respecting one-way streets.
    - Preserves node spatial coordinates ('x', 'y' / 'lon', 'lat') and OSM node IDs.
    - Preserves edge length, speed limits ('maxspeed'), road classification ('highway'),
      lanes, and geometry.
    - Caches graph locally in GraphML format to avoid repeated network requests.
    - Automatically falls back to point/distance buffer query if place query is a point.

    Args:
        place_name: Natural language location query (e.g. "Connaught Place, New Delhi, India").
        network_type: Type of street network ('drive', 'walk', 'bike', 'all').
        dist_meters: Buffer distance in meters around the center point if point fallback is triggered.
        cache_dir: Directory path to store/load cached graphs.
        filename_prefix: Optional custom prefix for the cache file.
        force_refresh: If True, bypasses cache and re-downloads from OSM.
        simplify: If True, simplifies graph topology while retaining edge geometries.

    Returns:
        nx.MultiDiGraph: Directed multi-graph representing the road network.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    prefix = filename_prefix or _sanitize_filename(place_name)
    cache_file = cache_path / f"{prefix}_{network_type}.graphml"

    if cache_file.exists() and not force_refresh:
        logger.info(f"Loading cached road network from {cache_file}")
        G = ox.load_graphml(filepath=cache_file)
        # Ensure node coordinate attributes are numeric
        for _, data in G.nodes(data=True):
            if "x" in data:
                data["x"] = float(data["x"])
            if "y" in data:
                data["y"] = float(data["y"])
            if "lat" in data:
                data["lat"] = float(data["lat"])
            if "lon" in data:
                data["lon"] = float(data["lon"])

        # Ensure edge length attributes are numeric
        for _, _, data in G.edges(data=True):
            if "length" in data:
                try:
                    data["length"] = float(data["length"])
                except (ValueError, TypeError):
                    pass
        return G

    logger.info(f"Downloading road network for '{place_name}' (type: {network_type}) via OSMnx...")
    # Configure OSMnx settings
    ox.settings.use_cache = True
    ox.settings.log_console = False

    try:
        # First attempt: polygon boundary by place name
        G = ox.graph_from_place(
            place_name,
            network_type=network_type,
            simplify=simplify,
            retain_all=False,
            truncate_by_edge=True,
        )
    except (TypeError, ValueError) as err:
        logger.info(f"Place query '{place_name}' did not return a boundary polygon ({err}). Falling back to point buffer ({dist_meters}m)...")
        # Geocode to point and extract bounded street network around center
        point = ox.geocode(place_name)
        G = ox.graph_from_point(
            point,
            dist=dist_meters,
            network_type=network_type,
            simplify=simplify,
            retain_all=False,
            truncate_by_edge=True,
        )

    # Save to local cache for future fast loading
    try:
        logger.info(f"Saving downloaded road network to cache: {cache_file}")
        ox.save_graphml(G, filepath=cache_file)
    except Exception as e:
        logger.warning(f"Failed to save graph cache to {cache_file}: {e}")

    return G

