"""Dynamic edge cost engine for road networks.

Calculates travel times, distance costs, and traffic-aware composite impedance
without mutating the underlying static road graph.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union


# Default fallback speed limits in km/h for standard OSM highway classifications.
# Used when the OSM 'maxspeed' tag is missing, malformed, or ambiguous.
DEFAULT_HIGHWAY_SPEEDS_KMH: Dict[str, float] = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 60.0,
    "trunk_link": 40.0,
    "primary": 50.0,
    "primary_link": 35.0,
    "secondary": 40.0,
    "secondary_link": 30.0,
    "tertiary": 30.0,
    "tertiary_link": 25.0,
    "residential": 25.0,
    "living_street": 15.0,
    "unclassified": 30.0,
    "service": 20.0,
    "default": 30.0,
}

# Infinite cost constant for closed edges to avoid numerical overflow while blocking paths
INFINITY_COST: float = float("inf")


@dataclass
class CostWeights:
    """Configurable weights for composite routing cost calculation.

    cost = weight_time * travel_time_sec
         + weight_distance * (distance_meters / 1000.0)
         + weight_congestion * congestion_penalty
    """
    weight_time: float = 1.0          # Multiplier for travel time (seconds)
    weight_distance: float = 0.0      # Multiplier for distance (kilometers)
    weight_congestion: float = 0.0    # Multiplier for congestion delay penalty (seconds)


@dataclass
class TrafficState:
    """Dynamic traffic condition for a specific road edge.

    Maintains separation between static network topology and dynamic simulation data.
    """
    current_speed_mps: Optional[float] = None  # Live observed speed in meters per second
    flow: Optional[float] = None               # Vehicle flow (veh/hr)
    capacity: Optional[float] = None           # Road capacity (veh/hr)
    congestion_factor: float = 1.0             # Congestion multiplier (>= 1.0 means congested)
    incident_multiplier: float = 1.0           # Incident delay multiplier (>= 1.0)
    is_closed: bool = False                    # Whether road segment is closed (e.g. accident/construction)


@dataclass
class EdgeCostResult:
    """Detailed breakdown of computed edge routing metrics."""
    effective_cost: float        # Final impedance value used by routing algorithms
    travel_time_sec: float       # Estimated traversal time in seconds
    distance_meters: float       # Physical road length in meters
    speed_mps: float             # Effective traversal speed in meters/second
    free_flow_speed_mps: float   # Base free-flow speed limit in meters/second
    congestion_factor: float     # Applied congestion factor
    is_closed: bool              # Whether edge is closed


class DynamicCostEngine:
    """Engine for computing dynamic, traffic-aware edge routing costs."""

    def __init__(
        self,
        default_weights: Optional[CostWeights] = None,
        highway_speeds_kmh: Optional[Dict[str, float]] = None,
    ):
        """Initialize cost engine with configurable parameters.

        Args:
            default_weights: Default weighting profile for composite cost calculation.
            highway_speeds_kmh: Custom fallback speed dictionary mapping highway types to km/h.
        """
        self.weights = default_weights or CostWeights()
        self.highway_speeds_kmh = highway_speeds_kmh or dict(DEFAULT_HIGHWAY_SPEEDS_KMH)

    def parse_maxspeed_mps(
        self,
        maxspeed_val: Any,
        highway_type: Any = None,
    ) -> float:
        """Extract speed limit in meters per second from OSM attributes with safe fallback.

        Args:
            maxspeed_val: Value from OSM 'maxspeed' tag (str, int, list, or None).
            highway_type: Value from OSM 'highway' tag for fallback lookup.

        Returns:
            Speed limit in meters per second (m/s).
        """
        if maxspeed_val is not None:
            # Handle list of speeds (OSMnx sometimes returns lists for multi-lane/dual carriageways)
            if isinstance(maxspeed_val, list) and len(maxspeed_val) > 0:
                maxspeed_val = maxspeed_val[0]

            if isinstance(maxspeed_val, (int, float)):
                if not math.isnan(maxspeed_val) and maxspeed_val > 0:
                    return float(maxspeed_val) / 3.6  # Convert km/h to m/s

            if isinstance(maxspeed_val, str):
                val_str = maxspeed_val.strip().lower()
                # Check for mph tag
                if "mph" in val_str:
                    num_match = re.search(r"(\d+(\.\d+)?)", val_str)
                    if num_match:
                        mph = float(num_match.group(1))
                        return (mph * 1.60934) / 3.6  # Convert mph to m/s
                # Check for km/h or raw number
                num_match = re.search(r"(\d+(\.\d+)?)", val_str)
                if num_match:
                    kmh = float(num_match.group(1))
                    if kmh > 0:
                        return kmh / 3.6

        # Fallback to highway type
        hw_key = "default"
        if highway_type is not None:
            if isinstance(highway_type, list) and len(highway_type) > 0:
                hw_type_str = str(highway_type[0]).lower()
            else:
                hw_type_str = str(highway_type).lower()

            if hw_type_str in self.highway_speeds_kmh:
                hw_key = hw_type_str

        fallback_kmh = self.highway_speeds_kmh.get(hw_key, self.highway_speeds_kmh["default"])
        return fallback_kmh / 3.6

    def compute_edge_cost(
        self,
        u: Any,
        v: Any,
        edge_data: Dict[str, Any],
        traffic_state: Optional[TrafficState] = None,
        weights: Optional[CostWeights] = None,
    ) -> EdgeCostResult:
        """Compute the dynamic impedance and metric breakdown for a road edge.

        Args:
            u: Origin node ID.
            v: Destination node ID.
            edge_data: Dictionary of edge attributes (must contain 'length').
            traffic_state: Optional dynamic traffic conditions for this edge.
            weights: Optional custom cost weights overriding defaults.

        Returns:
            EdgeCostResult dataclass containing effective cost and metrics.
        """
        active_weights = weights or self.weights

        # Physical road length in meters
        try:
            length_meters = float(edge_data.get("length", 1.0))
            if length_meters <= 0:
                length_meters = 1.0
        except (ValueError, TypeError):
            length_meters = 1.0

        # Check road closure
        is_closed = False
        if traffic_state is not None and traffic_state.is_closed:
            is_closed = True
        elif edge_data.get("closed", False) is True:
            is_closed = True

        if is_closed:
            return EdgeCostResult(
                effective_cost=INFINITY_COST,
                travel_time_sec=INFINITY_COST,
                distance_meters=length_meters,
                speed_mps=0.0,
                free_flow_speed_mps=0.0,
                congestion_factor=float("inf"),
                is_closed=True,
            )

        # Determine free-flow speed (m/s)
        free_flow_speed_mps = self.parse_maxspeed_mps(
            maxspeed_val=edge_data.get("maxspeed"),
            highway_type=edge_data.get("highway"),
        )
        # Enforce minimum speed floor (1.0 m/s = 3.6 km/h) to prevent division by zero
        free_flow_speed_mps = max(free_flow_speed_mps, 1.0)

        # Incorporate dynamic traffic state
        congestion_factor = 1.0
        incident_multiplier = 1.0
        effective_speed_mps = free_flow_speed_mps

        if traffic_state is not None:
            congestion_factor = max(traffic_state.congestion_factor, 1.0)
            incident_multiplier = max(traffic_state.incident_multiplier, 1.0)

            if traffic_state.current_speed_mps is not None and traffic_state.current_speed_mps > 0:
                effective_speed_mps = max(traffic_state.current_speed_mps, 0.5)
            else:
                # Speed degrades inversely with congestion and incident factors
                effective_speed_mps = free_flow_speed_mps / (congestion_factor * incident_multiplier)
                effective_speed_mps = max(effective_speed_mps, 0.5)

        # Travel times in seconds
        free_flow_time_sec = length_meters / free_flow_speed_mps
        travel_time_sec = length_meters / effective_speed_mps
        congestion_delay_sec = max(0.0, travel_time_sec - free_flow_time_sec)

        # Composite weighted cost
        # cost = w_time * t + w_dist * (km) + w_cong * delay
        effective_cost = (
            active_weights.weight_time * travel_time_sec
            + active_weights.weight_distance * (length_meters / 1000.0)
            + active_weights.weight_congestion * congestion_delay_sec
        )

        return EdgeCostResult(
            effective_cost=effective_cost,
            travel_time_sec=travel_time_sec,
            distance_meters=length_meters,
            speed_mps=effective_speed_mps,
            free_flow_speed_mps=free_flow_speed_mps,
            congestion_factor=congestion_factor * incident_multiplier,
            is_closed=False,
        )
