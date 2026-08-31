"""Configuration manager for the traffic optimization system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class StudyAreaConfig:
    place_name: str = "Connaught Place, New Delhi, India"
    network_type: str = "drive"
    buffer_dist_meters: Optional[int] = None


@dataclass
class CacheConfig:
    enabled: bool = True
    cache_dir: str = "data/cache"
    filename_prefix: str = "connaught_place_drive"


@dataclass
class VisualizationConfig:
    export_plot: bool = True
    output_image: str = "data/inspection_graph.png"


@dataclass
class AppConfig:
    study_area: StudyAreaConfig = field(default_factory=StudyAreaConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults if not found.

    Args:
        config_path: Path to YAML config file. If None, looks for configs/default_config.yaml.

    Returns:
        AppConfig dataclass populated with values.
    """
    if config_path is None:
        # Default to repo root / configs / default_config.yaml
        default_file = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"
        if default_file.exists():
            config_path = default_file

    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        study_area_data = data.get("study_area", {})
        cache_data = data.get("cache", {})
        vis_data = data.get("visualization", {})

        return AppConfig(
            study_area=StudyAreaConfig(
                place_name=study_area_data.get("place_name", "Connaught Place, New Delhi, India"),
                network_type=study_area_data.get("network_type", "drive"),
                buffer_dist_meters=study_area_data.get("buffer_dist_meters"),
            ),
            cache=CacheConfig(
                enabled=cache_data.get("enabled", True),
                cache_dir=cache_data.get("cache_dir", "data/cache"),
                filename_prefix=cache_data.get("filename_prefix", "connaught_place_drive"),
            ),
            visualization=VisualizationConfig(
                export_plot=vis_data.get("export_plot", True),
                output_image=vis_data.get("output_image", "data/inspection_graph.png"),
            ),
        )

    return AppConfig()
