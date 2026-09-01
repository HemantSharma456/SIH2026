"""Unit tests for quantum-inspired candidate state representation (Stage 3B)."""

import math
import numpy as np
import pytest

from src.algorithms.qiga.quantum_state import (
    QuantumCandidateState,
    QuantumStateError,
    NORMALIZATION_TOLERANCE,
)


def test_initialization_k1():
    """Test initialization for K=1 candidate."""
    qs = QuantumCandidateState(k=1)
    assert qs.k == 1
    assert qs.alpha.shape == (1,)
    assert qs.beta.shape == (1,)
    assert math.isclose(qs.alpha[0], 1.0 / math.sqrt(2.0), rel_tol=1e-5)
    assert math.isclose(qs.beta[0], 1.0 / math.sqrt(2.0), rel_tol=1e-5)


def test_initialization_k5():
    """Test default initialization for K=5 candidates."""
    qs = QuantumCandidateState(k=5)
    assert qs.k == 5
    assert qs.amplitudes.shape == (5, 2)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(qs.alpha, inv_sqrt2, rtol=1e-5)
    np.testing.assert_allclose(qs.beta, inv_sqrt2, rtol=1e-5)


def test_initialization_arbitrary_k():
    """Test initialization for large arbitrary K=50."""
    qs = QuantumCandidateState(k=50)
    assert qs.k == 50
    assert qs.amplitudes.shape == (50, 2)
    assert qs.validate() is True


def test_normalization_constraint():
    """Test that alpha^2 + beta^2 == 1 for every candidate state."""
    qs = QuantumCandidateState(k=10)
    sums_of_squares = np.sum(np.square(qs.amplitudes), axis=1)
    np.testing.assert_allclose(sums_of_squares, 1.0, atol=NORMALIZATION_TOLERANCE)


def test_initial_probabilities_equal_half():
    """Test that default unbiased initialization produces P(selected) = 0.5."""
    qs = QuantumCandidateState(k=5)
    sel_probs = qs.get_selection_probabilities()
    non_sel_probs = qs.get_non_selection_probabilities()

    np.testing.assert_allclose(sel_probs, 0.5, atol=1e-5)
    np.testing.assert_allclose(non_sel_probs, 0.5, atol=1e-5)
    np.testing.assert_allclose(sel_probs + non_sel_probs, 1.0, atol=1e-5)


def test_probabilities_bounded_in_zero_one():
    """Test that probabilities strictly remain in [0.0, 1.0] across various amplitude configurations."""
    # Highly biased state: Candidate 0 -> high prob, Candidate 1 -> low prob
    custom_alpha = [0.1, 0.99]
    custom_beta = [0.9949874, 0.141067]
    qs = QuantumCandidateState(k=2, alpha=custom_alpha, beta=custom_beta, auto_normalize=True)

    probs = qs.get_selection_probabilities()
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert probs[0] > 0.95
    assert probs[1] < 0.05


def test_measurement_returns_valid_indices():
    """Test that candidate measurement always returns an integer index in [0, K-1]."""
    K = 7
    qs = QuantumCandidateState(k=K)
    rng = np.random.default_rng(42)

    for _ in range(50):
        idx = qs.measure_candidate_index(rng=rng)
        assert isinstance(idx, int)
        assert 0 <= idx < K


def test_measurement_reproducibility():
    """Test that passing the exact same random seed generates identical measurement sequences."""
    qs = QuantumCandidateState(k=5)

    seq1 = [qs.measure_candidate_index(seed=12345 + i) for i in range(10)]
    seq2 = [qs.measure_candidate_index(seed=12345 + i) for i in range(10)]

    assert seq1 == seq2

    # Bitstring measurement reproducibility
    bs1 = qs.measure_bitstring(seed=999)
    bs2 = qs.measure_bitstring(seed=999)
    np.testing.assert_array_equal(bs1, bs2)


def test_different_seeds_produce_variation():
    """Test that different random seeds explore multiple candidates over many samples."""
    qs = QuantumCandidateState(k=10)
    samples_seed_a = [qs.measure_candidate_index(seed=1000 + i) for i in range(20)]
    samples_seed_b = [qs.measure_candidate_index(seed=5000 + i) for i in range(20)]

    # Over 20 trials with 10 candidates, multiple distinct indices must be explored
    assert len(set(samples_seed_a)) > 1
    assert len(set(samples_seed_b)) > 1


def test_k_zero_handling():
    """Test that K=0 is handled clearly."""
    qs = QuantumCandidateState(k=0)
    assert qs.k == 0
    assert len(qs.get_selection_probabilities()) == 0
    assert len(qs.measure_bitstring()) == 0

    with pytest.raises(ValueError, match="Cannot measure from a quantum state with K=0"):
        qs.measure_candidate_index()

    # Negative K
    with pytest.raises(ValueError, match="must be non-negative"):
        QuantumCandidateState(k=-1)


def test_k_one_always_selects_zero():
    """Test that when K=1, measurement deterministically returns candidate 0."""
    qs = QuantumCandidateState(k=1)
    for s in range(20):
        assert qs.measure_candidate_index(seed=s) == 0


def test_exact_probability_calculation():
    """Test mathematical exactness of P = beta^2 and 1-P = alpha^2."""
    # alpha = 0.6, beta = 0.8 -> alpha^2 = 0.36, beta^2 = 0.64
    qs = QuantumCandidateState(k=1, alpha=[0.6], beta=[0.8], auto_normalize=False)

    p_sel = qs.get_selection_probabilities()[0]
    p_non = qs.get_non_selection_probabilities()[0]

    assert math.isclose(p_sel, 0.64, rel_tol=1e-6)
    assert math.isclose(p_non, 0.36, rel_tol=1e-6)


def test_numerical_normalization():
    """Test that auto_normalize safely re-scales unnormalized amplitudes."""
    # Unnormalized: alpha = 3.0, beta = 4.0 (norm = 5.0) -> scaled to 0.6, 0.8
    qs = QuantumCandidateState(k=1, alpha=[3.0], beta=[4.0], auto_normalize=True)

    assert math.isclose(qs.alpha[0], 0.6, rel_tol=1e-6)
    assert math.isclose(qs.beta[0], 0.8, rel_tol=1e-6)
    assert qs.validate() is True


def test_no_nan_or_infinite_values():
    """Test that NaN and infinity are rejected with QuantumStateError."""
    with pytest.raises(QuantumStateError, match="contain NaN or infinite values"):
        QuantumCandidateState(k=2, alpha=[np.nan, 0.5], beta=[0.5, 0.5])

    with pytest.raises(QuantumStateError, match="contain NaN or infinite values"):
        QuantumCandidateState(k=2, alpha=[np.inf, 0.5], beta=[0.5, 0.5])


def test_state_validation_detects_unnormalized_state():
    """Test that validate() raises QuantumStateError when amplitudes violate constraint without auto_normalize."""
    with pytest.raises(QuantumStateError, match="Normalization violation"):
        QuantumCandidateState(k=2, alpha=[0.9, 0.9], beta=[0.9, 0.9], auto_normalize=False)
