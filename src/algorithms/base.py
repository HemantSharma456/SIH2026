"""Base interfaces and data structures for routing optimizers and baselines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx

from src.traffic.cost_engine import CostWeights, DynamicCostEngine, TrafficState


@dataclass
class RouteResult:
    """Standardized output structure for any routing algorithm."""
    origin: Any
    destination: Any
    route: List[Any]                          # Sequence of node IDs from origin to destination
    edge_keys: Optional[List[Any]] = None    # Edge keys for MultiDiGraph parallel edge disambiguation
    total_cost: float = float("inf")          # Total evaluated impedance
    total_distance_meters: float = 0.0        # Physical route length in meters
    total_travel_time_sec: float = 0.0        # Estimated traversal duration in seconds
    is_valid: bool = False                    # Whether route is complete, continuous, and unblocked
    algorithm_name: str = "Unknown"           # Identifier of the routing algorithm
    execution_time_ms: float = 0.0            # Algorithm execution runtime in milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the route path."""
        return len(self.route)

    @property
    def total_distance_km(self) -> float:
        """Total route distance in kilometers."""
        return self.total_distance_meters / 1000.0


class BaseRouter(ABC):
    """Abstract base class for all routing algorithms."""

    @abstractmethod
    def solve(
        self,
        G: Union[nx.DiGraph, nx.MultiDiGraph],
        origin: Any,
        destination: Any,
        cost_engine: Optional[DynamicCostEngine] = None,
        traffic_states: Optional[Dict[Tuple[Any, Any], TrafficState]] = None,
        weights: Optional[CostWeights] = None,
        **kwargs: Any,
    ) -> RouteResult:
        """Compute an optimal route from origin to destination.

        Args:
            G: Directed road graph (DiGraph or MultiDiGraph).
            origin: Starting node ID.
            destination: Target node ID.
            cost_engine: Dynamic edge cost engine for evaluating impedance.
            traffic_states: Optional edge-specific traffic conditions mapping (u, v) -> TrafficState.
            weights: Optional cost weighting profile.

        Returns:
            RouteResult dataclass.
        """
        pass
