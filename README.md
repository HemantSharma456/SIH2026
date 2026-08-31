# Quantum-Inspired Intelligent Traffic Route Optimization (PS 26137)

Smart India Hackathon (SIH) 2026 Project: **Quantum-Inspired Intelligent Traffic Route Optimization in Transportation Systems Using Metaheuristic Optimization**.

---

## Project Overview

This project develops an intelligent transportation optimization framework that models complex urban road networks, evaluates dynamic traffic conditions, and computes optimal vehicle routes using a **Quantum-Inspired Genetic Algorithm (QIGA)** compared against classical baselines (Dijkstra, A*, Classical GA, PSO, ACO).

> **Note**: This is a quantum-inspired classical metaheuristic optimization algorithm and does **not** require physical quantum computing hardware.

---

## Architecture & Data Pipeline

```
OpenStreetMap (OSM)
        ↓
      OSMnx
        ↓
   NetworkX Directed Road Graph
        ↓
 Dynamic Edge-Cost Engine (Phase 2)
        ↓
 Classical Routing Baselines (Dijkstra / A*) (Phase 2)
        ↓
 Quantum-Inspired Genetic Algorithm (QIGA) (Phase 3)
        ↓
 Graph-Constrained Route Decoder
        ↓
 Valid Optimized Routes
```

---

## Why OSM, OSMnx, and NetworkX?

1. **OpenStreetMap (OSM)**: Open, crowdsourced, highly detailed global geographic dataset containing rich topological attributes including one-way constraints, lane counts, speed limits (`maxspeed`), highway classifications, and turn restrictions.
2. **OSMnx**: Python library built specifically for spatial street networks. It simplifies raw spatial data, preserves real-world geometric curves, and converts OSM spatial entities directly into mathematically rigorous NetworkX graphs.
3. **NetworkX**: Standard Python graph analysis framework offering fast in-memory graph operations, adjacency iteration, strongly connected component analysis, and native compatibility with custom metaheuristics and search algorithms.

---

## Phase 1 Scope: Core Graph Foundation

- **Modular `src`-based Architecture**: Clean separation between configuration, graph ingestion, validation, and testing.
- **Configurable Study Area**: Easy configuration via `configs/default_config.yaml` (defaulting to *"Connaught Place, New Delhi, India"*).
- **Drivable Directed Graph Extraction**: Extracts directed graphs respecting one-way streets, preserving spatial coordinates (`x`, `y` / `lon`, `lat`) and physical edge lengths.
- **Local Caching**: Saves parsed graphs in `data/cache/*.graphml` format to prevent unnecessary repeated network downloads.
- **Rigorous Graph Validation & Diagnostics**: Validates connectivity, coordinate integrity, edge lengths, and one-way reachability.
- **Inspection Tooling**: Runnable CLI tool (`scripts/inspect_graph.py`) reporting graph metrics, road hierarchy distribution, and exporting visual network plots.
- **Unit Test Suite**: 100% offline unit tests verifying structural validation rules, connectivity checks, and configuration fallbacks on synthetic directed graphs.

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12/3.13
- Git

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Inspection Script
To download/load the configured study area, validate graph attributes, and export a visual plot:
```bash
python scripts/inspect_graph.py
```

Optional CLI flags:
- `--place "Custom Locality Name"`: Override the default study area.
- `--force-refresh`: Ignore cached `.graphml` files and re-download from OSM.
- `--config path/to/custom_config.yaml`: Use a custom configuration file.
- `--no-plot`: Skip saving the output image.

### 3. Run the Unit Test Suite
```bash
pytest tests/ -v
```

---

## Configuration (`configs/default_config.yaml`)

```yaml
study_area:
  place_name: "Connaught Place, New Delhi, India"
  network_type: "drive"
  buffer_dist_meters: null

cache:
  enabled: true
  cache_dir: "data/cache"
  filename_prefix: "connaught_place_drive"

visualization:
  export_plot: true
  output_image: "data/inspection_graph.png"
```

---

## Next Steps (Subsequent Phases)
- **Phase 2**: Dynamic Edge Cost Engine & Classical Routing Baselines (Dijkstra, A*).
- **Phase 3**: Quantum-Inspired Genetic Algorithm (QIGA) Core & Graph-Aware Decoder.
- **Phase 4**: Extended Baselines (Classical GA, PSO, ACO) & Benchmark Suite.
- **Phase 5**: SUMO Simulation & TraCI Dynamic Incident Interface.
