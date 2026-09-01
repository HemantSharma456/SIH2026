"""Quantum-inspired candidate state representation for QIGA optimization.

Maintains probability amplitudes (alpha, beta) over a set of K candidate solutions,
providing observation/measurement primitives with numerical stability and reproducibility.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Numerical tolerance for amplitude normalization constraint (alpha^2 + beta^2 == 1)
NORMALIZATION_TOLERANCE: float = 1e-5


class QuantumStateError(Exception):
    """Raised when quantum candidate amplitudes violate mathematical constraints."""
    pass


class QuantumCandidateState:
    """Maintains quantum-inspired probability amplitudes (alpha, beta) for K candidate solutions.

    For each candidate solution i in {0, ..., K-1}:
        - alpha_i: Probability amplitude for non-selection
        - beta_i: Probability amplitude for selection
        - Constraint: alpha_i^2 + beta_i^2 == 1.0

    Selection Probability:
        P(candidate_i selected) = beta_i^2

    Non-selection Probability:
        P(candidate_i not selected) = alpha_i^2
    """

    def __init__(
        self,
        k: int,
        alpha: Optional[Union[np.ndarray, List[float]]] = None,
        beta: Optional[Union[np.ndarray, List[float]]] = None,
        auto_normalize: bool = True,
    ):
        """Initialize quantum candidate state for K candidate routes.

        Args:
            k: Number of candidate solutions (K >= 0).
            alpha: Optional custom initial alpha amplitudes of length K.
            beta: Optional custom initial beta amplitudes of length K.
            auto_normalize: If True, automatically normalizes amplitudes to satisfy alpha^2 + beta^2 = 1.

        Raises:
            ValueError: If K < 0 or if custom amplitude dimensions do not match K.
            QuantumStateError: If custom amplitudes contain NaN/infinite values.
        """
        if k < 0:
            raise ValueError(f"Number of candidates K must be non-negative, got {k}.")

        self.k: int = k

        if k == 0:
            self._amplitudes = np.empty((0, 2), dtype=np.float64)
            return

        if alpha is None and beta is None:
            # Default unbiased initialization: alpha = beta = 1 / sqrt(2) -> P = 0.5
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            self._amplitudes = np.full((k, 2), inv_sqrt2, dtype=np.float64)
        else:
            if alpha is None or beta is None:
                raise ValueError("Both 'alpha' and 'beta' must be provided if one is specified.")

            alpha_arr = np.asarray(alpha, dtype=np.float64).flatten()
            beta_arr = np.asarray(beta, dtype=np.float64).flatten()

            if len(alpha_arr) != k or len(beta_arr) != k:
                raise ValueError(
                    f"Amplitudes length mismatch: expected K={k}, got len(alpha)={len(alpha_arr)}, len(beta)={len(beta_arr)}."
                )

            if not (np.all(np.isfinite(alpha_arr)) and np.all(np.isfinite(beta_arr))):
                raise QuantumStateError("Amplitudes contain NaN or infinite values.")

            self._amplitudes = np.column_stack((alpha_arr, beta_arr))

            if auto_normalize:
                self.normalize()
            else:
                self.validate()

    @property
    def alpha(self) -> np.ndarray:
        """1D array of alpha amplitudes (non-selection) of shape (K,)."""
        return self._amplitudes[:, 0]

    @alpha.setter
    def alpha(self, values: np.ndarray) -> None:
        val_arr = np.asarray(values, dtype=np.float64).flatten()
        if len(val_arr) != self.k:
            raise ValueError(f"Expected {self.k} alpha values, got {len(val_arr)}.")
        self._amplitudes[:, 0] = val_arr

    @property
    def beta(self) -> np.ndarray:
        """1D array of beta amplitudes (selection) of shape (K,)."""
        return self._amplitudes[:, 1]

    @beta.setter
    def beta(self, values: np.ndarray) -> None:
        val_arr = np.asarray(values, dtype=np.float64).flatten()
        if len(val_arr) != self.k:
            raise ValueError(f"Expected {self.k} beta values, got {len(val_arr)}.")
        self._amplitudes[:, 1] = val_arr

    @property
    def amplitudes(self) -> np.ndarray:
        """2D array of shape (K, 2) where column 0 is alpha and column 1 is beta."""
        return self._amplitudes

    def get_selection_probabilities(self) -> np.ndarray:
        """Compute the selection probability P_i = beta_i^2 for each candidate.

        Returns:
            1D numpy array of probabilities in [0.0, 1.0] of length K.
        """
        if self.k == 0:
            return np.empty(0, dtype=np.float64)
        # Clip to [0, 1] for guaranteed numerical bounds
        probs = np.clip(np.square(self._amplitudes[:, 1]), 0.0, 1.0)
        return probs

    def get_non_selection_probabilities(self) -> np.ndarray:
        """Compute non-selection probability 1 - P_i = alpha_i^2 for each candidate.

        Returns:
            1D numpy array of probabilities in [0.0, 1.0] of length K.
        """
        if self.k == 0:
            return np.empty(0, dtype=np.float64)
        probs = np.clip(np.square(self._amplitudes[:, 0]), 0.0, 1.0)
        return probs

    def normalize(self) -> None:
        """Normalize amplitude pairs so that alpha_i^2 + beta_i^2 == 1.0 exactly.

        Safely handles zero or near-zero norms by resetting to equal superposition.
        """
        if self.k == 0:
            return

        norms = np.hypot(self._amplitudes[:, 0], self._amplitudes[:, 1])
        # Find any degenerate near-zero amplitudes
        zero_mask = norms < 1e-12
        if np.any(zero_mask):
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            self._amplitudes[zero_mask] = [inv_sqrt2, inv_sqrt2]
            norms[zero_mask] = 1.0

        self._amplitudes /= norms[:, np.newaxis]

    def validate(self, tol: float = NORMALIZATION_TOLERANCE) -> bool:
        """Check that all amplitudes satisfy mathematical and normalization constraints.

        Args:
            tol: Numerical tolerance for alpha^2 + beta^2 == 1.

        Returns:
            True if valid.

        Raises:
            QuantumStateError: If any amplitude is NaN, inf, or unnormalized.
        """
        if self.k == 0:
            return True

        if not np.all(np.isfinite(self._amplitudes)):
            raise QuantumStateError("Amplitudes contain NaN or non-finite values.")

        sums_of_squares = np.sum(np.square(self._amplitudes), axis=1)
        deviations = np.abs(sums_of_squares - 1.0)
        max_dev = np.max(deviations)

        if max_dev > tol:
            bad_idx = int(np.argmax(deviations))
            raise QuantumStateError(
                f"Normalization violation at candidate index {bad_idx}: "
                f"alpha^2 + beta^2 = {sums_of_squares[bad_idx]:.8f} (deviation {max_dev:.2e} > tol {tol:.2e})."
            )

        return True

    def measure_candidate_index(
        self,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> int:
        """Perform a quantum observation and collapse to sample a single candidate index.

        The probability of selecting candidate i is proportional to its selection probability beta_i^2:
            p_i = beta_i^2 / sum(beta_j^2)

        Args:
            rng: Optional explicit NumPy Generator.
            seed: Optional integer seed (used if rng is None).

        Returns:
            Integer index in {0, ..., K-1} representing the measured candidate.

        Raises:
            ValueError: If K == 0 (no candidates to sample).
        """
        if self.k == 0:
            raise ValueError("Cannot measure from a quantum state with K=0 candidates.")

        if self.k == 1:
            return 0

        gen = rng if rng is not None else np.random.default_rng(seed)

        probs = self.get_selection_probabilities()
        total_p = np.sum(probs)

        if total_p < 1e-12:
            # If all selection probabilities are zero, sample uniformly
            normalized_p = np.full(self.k, 1.0 / self.k, dtype=np.float64)
        else:
            normalized_p = probs / total_p

        # Sample single candidate index
        chosen_index = int(gen.choice(self.k, p=normalized_p))
        return chosen_index

    def measure_bitstring(
        self,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Perform a Bernoulli bitwise observation for each candidate.

        Each bit x_i in {0, 1} is sampled independently with P(x_i = 1) = beta_i^2.

        Args:
            rng: Optional explicit NumPy Generator.
            seed: Optional integer seed.

        Returns:
            1D numpy integer array of length K with values in {0, 1}.
        """
        if self.k == 0:
            return np.empty(0, dtype=np.int32)

        gen = rng if rng is not None else np.random.default_rng(seed)

        probs = self.get_selection_probabilities()
        uniform_draws = gen.random(self.k)
        bitstring = (uniform_draws < probs).astype(np.int32)
        return bitstring

    def to_dict(self) -> Dict[str, Any]:
        """Export state representation to dictionary for serialization/inspection."""
        return {
            "k": self.k,
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "selection_probabilities": self.get_selection_probabilities().tolist(),
        }

    def __repr__(self) -> str:
        return f"QuantumCandidateState(k={self.k}, probs={np.round(self.get_selection_probabilities(), 3)})"
