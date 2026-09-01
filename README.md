# Quantum-Inspired Intelligent Traffic Route Optimization (PS 26137)

Smart India Hackathon (SIH) 2026 Project: **Quantum-Inspired Intelligent Traffic Route Optimization in Transportation Systems Using Metaheuristic Optimization**.

---

## Project Overview

This project develops an intelligent transportation optimization framework that models complex urban road networks, evaluates dynamic traffic conditions, and computes optimal vehicle routes using a **Quantum-Inspired Genetic Algorithm (QIGA)** compared against classical baselines (Dijkstra, A*, Classical GA, PSO, ACO).

> **Note**: This is a quantum-inspired classical metaheuristic optimization algorithm executed on conventional hardware. It does **not** require physical quantum computing hardware and makes **no unsupported claims of physical quantum advantage**.

---

## Architecture & Data Pipeline

```
OpenStreetMap (OSM)
        ↓
      OSMnx
        ↓
   NetworkX Directed Road Graph (MultiDiGraph)
        ↓
 Dynamic Edge-Cost Engine (Phase 2)
        ↓
 Classical Routing Baselines (Dijkstra / A*) (Phase 2)
        ↓
 Graph-Constrained Candidate Path Generation (Stage 3A)
        ↓
 Quantum-Inspired Candidate Representation (Stage 3B)
        ↓
 QIGA Rotation-Gate Optimizer (Stage 3C)
        ↓
 Valid Optimized Routes
```

---

## Phase 3: Quantum-Inspired Genetic Algorithm (QIGA)

### Stage 3B: Quantum-Inspired Candidate Representation (`src/algorithms/qiga/quantum_state.py`)

#### 1. What "Quantum-Inspired" Means
In standard Classical Genetic Algorithms, each individual is represented as a fixed binary string or permutation. In QIGA, each candidate state is represented by a **qubit-like probability amplitude pair** $(\alpha_i, \beta_i)$ in $\mathbb{R}^2$:

$$\begin{bmatrix} \alpha_i \\ \beta_i \end{bmatrix}, \quad \text{such that } \alpha_i^2 + \beta_i^2 = 1$$

This allows a single quantum chromosome to represent a continuous **probabilistic superposition** over multiple candidate routing solutions simultaneously.

#### 2. Relationship Between Amplitudes and Probabilities
For each candidate route index $i \in \{0, \dots, K-1\}$:
- **Selection Probability**: $P(\text{candidate } i \text{ is selected}) = \beta_i^2$
- **Non-Selection Probability**: $P(\text{candidate } i \text{ is NOT selected}) = \alpha_i^2$
- **Initial Unbiased State**: $\alpha_i = \beta_i = \frac{1}{\sqrt{2}} \approx 0.707107 \implies P_i = 0.5$ for all candidates.

#### 3. What "Measurement" Means
Measurement models the quantum observation process:
- Sampling from the probability distribution $p_i = \frac{\beta_i^2}{\sum_{j=0}^{K-1} \beta_j^2}$ collapses the continuous probability amplitudes into a concrete discrete candidate route index $c \in \{0, \dots, K-1\}$.
- Deterministic reproducibility is guaranteed via explicit NumPy `Generator` seeding.

#### 4. Decoupling from Road Graph Data
The quantum state representation is strictly abstract ($\mathbb{R}^{K \times 2}$ vector space). It stores no node IDs, coordinates, or spatial geometries, maintaining $\mathcal{O}(K)$ memory footprint and ensuring portability to any road network.

---

### Stage 3A: Graph-Constrained Candidate Path Generation (`src/algorithms/qiga/path_generator.py`)

- **Why Candidate Paths**: Naive binary edge encoding creates disconnected graphs in >99.9% of random bitstrings. Our generator constructs a bounded feasible pool $\mathcal{P} = \{P_1, \dots, P_K\}$ of 100% physically valid, continuous, loopless routes.
- **Yen's $K$-Shortest Paths**: Runs in polynomial $\mathcal{O}(K \cdot L \cdot (|E| + |V| \log |V|))$ time, avoiding factorial path explosions.

---

## Phase 2: Dynamic Edge Cost Engine & Dijkstra Baseline

### 1. Dynamic Edge Cost Engine (`src/traffic/cost_engine.py`)
Computes composite edge impedance:
$$\text{Effective Cost} = w_{\text{time}} \cdot t_{\text{travel}} + w_{\text{dist}} \cdot d_{\text{km}} + w_{\text{cong}} \cdot \text{Delay}_{\text{sec}}$$

- **Free-Flow Travel Time**: $t_0 = \frac{\text{length (meters)}}{\text{speed (m/s)}}$
- **Traffic-Aware Travel Time**: Effective speed degrades: $v_{\text{eff}} = \frac{v_0}{C \times I}$
- **Closed Road Handling**: Closed edges evaluate to $\infty$ impedance, preventing traversal.

### 2. Highway Speed Fallback Assumptions
When OSM `maxspeed` tags are missing, free-flow speeds are inferred from the road hierarchy:
- `motorway`: 80 km/h | `trunk`: 60 km/h | `primary`: 50 km/h | `secondary`: 40 km/h | `tertiary`: 30 km/h | `residential`: 25 km/h | `living_street`: 15 km/h | `default`: 30 km/h

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12/3.13
- Git

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Stage 3B Smoke Test (Quantum Representation & Measurement)
```bash
python scripts/smoke_test_stage3b.py
```

### 3. Run the Stage 3A Smoke Test (Candidate Path Generation)
```bash
python scripts/smoke_test_stage3a.py
```

### 4. Run the Phase 2 Smoke Test (Dijkstra Baseline)
```bash
python scripts/smoke_test_phase2.py
```

### 5. Run the Full Test Suite
```bash
pytest tests/ -v
```

---

## Next Steps (Subsequent Phases)
- **Stage 3C**: QIGA Quantum Rotation Gate Updates, Elitist Replacement, and Convergence Loop.
- **Phase 4**: Extended Baselines (Classical GA, PSO, ACO) & Comparative Benchmark Suite.
- **Phase 5**: SUMO Simulation & TraCI Dynamic Incident Interface.
