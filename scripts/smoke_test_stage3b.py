"""Stage 3B Integration Smoke Test: Quantum-Inspired Candidate Representation.

Demonstrates quantum-inspired probability amplitude representation over K synthetic
candidate solutions, showing amplitude normalization, probability calculation,
and reproducible observation/measurement without coupling to road network graphs.
"""

from pathlib import Path
import sys
import numpy as np

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algorithms.qiga.quantum_state import QuantumCandidateState


def main() -> int:
    print("=" * 80)
    print("  SIH 2026 -- STAGE 3B: QUANTUM-INSPIRED CANDIDATE REPRESENTATION")
    print("=" * 80)

    K = 5
    print(f"Creating Quantum Candidate State for K = {K} candidate solutions...\n")

    # 1. Initialize unbiased quantum state
    qs = QuantumCandidateState(k=K)
    qs.validate()

    print("[1. INITIAL UNBIASED QUANTUM STATE (Equal Superposition)]")
    print(f"{'Candidate Index':<18} | {'Alpha (alpha_i)':<18} | {'Beta (beta_i)':<18} | {'P(Selected) = beta^2':<22}")
    print("-" * 84)

    sel_probs = qs.get_selection_probabilities()
    for i in range(K):
        print(f"Candidate #{i:<11} | {qs.alpha[i]:<18.6f} | {qs.beta[i]:<18.6f} | {sel_probs[i]:<22.4f}")

    print("-" * 84)
    print("Normalization Check: alpha^2 + beta^2 =", np.round(np.square(qs.alpha) + np.square(qs.beta), 6))

    # 2. Measurement / Observation under fixed seed
    print("\n[2. MEASUREMENT OBSERVATION SEQUENCE (Fixed Seed = 42)]")
    rng = np.random.default_rng(42)
    measurements = []

    for trial in range(1, 11):
        sampled_idx = qs.measure_candidate_index(rng=rng)
        measurements.append(sampled_idx)
        print(f"  - Trial {trial:02d} Observation -> Selected Candidate Index: {sampled_idx} (Valid: {0 <= sampled_idx < K})")

    # Check bounds
    all_valid_indices = all(0 <= idx < K for idx in measurements)
    print(f"\nAll observed indices strictly in [0, {K-1}]: {all_valid_indices}")

    # 3. Demonstration of a biased quantum state (favoring Candidate #2)
    print("\n[3. BIASED QUANTUM STATE DEMONSTRATION (Elevated Amplitude for Candidate #2)]")
    # Set Candidate #2 with high beta amplitude (0.95), others with lower amplitudes (0.312)
    custom_beta = [0.2, 0.2, 0.95, 0.2, 0.2]
    custom_alpha = [np.sqrt(1.0 - b**2) for b in custom_beta]

    biased_qs = QuantumCandidateState(k=K, alpha=custom_alpha, beta=custom_beta, auto_normalize=True)

    print(f"{'Candidate Index':<18} | {'Alpha (alpha_i)':<18} | {'Beta (beta_i)':<18} | {'P(Selected) = beta^2':<22}")
    print("-" * 84)
    biased_probs = biased_qs.get_selection_probabilities()
    for i in range(K):
        print(f"Candidate #{i:<11} | {biased_qs.alpha[i]:<18.6f} | {biased_qs.beta[i]:<18.6f} | {biased_probs[i]:<22.4f}")
    print("-" * 84)

    # Sample 100 times to observe empirical frequency
    rng_biased = np.random.default_rng(12345)
    sample_counts = {i: 0 for i in range(K)}
    num_samples = 100

    for _ in range(num_samples):
        idx = biased_qs.measure_candidate_index(rng=rng_biased)
        sample_counts[idx] += 1

    print(f"\nEmpirical selection frequency over {num_samples} measurements:")
    for i in range(K):
        pct = (sample_counts[i] / num_samples) * 100.0
        bar = "#" * int(pct // 2)
        print(f"  - Candidate #{i}: {sample_counts[i]:>3} hits ({pct:>5.1f}%) | {bar}")

    print("\n" + "=" * 80)
    print("  STAGE 3B INTEGRATION TEST COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
