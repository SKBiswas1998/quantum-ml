"""
Quantum Kernel for Machine Learning.

References:
    Havlicek et al. (2019). Supervised learning with quantum-enhanced feature spaces.
"""
import numpy as np
from typing import Optional

VALID_FEATURE_MAPS = ('z', 'zz')


class QuantumKernel:
    """
    Quantum kernel for computing similarity in quantum feature space.

    K(x, x') = |<phi(x)|phi(x')>|^2

    Example:
        >>> kernel = QuantumKernel(num_qubits=2)
        >>> K = kernel.compute_matrix(X_train)
    """

    def __init__(self, num_qubits: int = 2, feature_map: str = 'zz'):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer")
        if feature_map not in VALID_FEATURE_MAPS:
            raise ValueError(f"feature_map must be one of {VALID_FEATURE_MAPS}")
        self.num_qubits = num_qubits
        self.feature_map = feature_map

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode classical data into quantum state using the specified feature map."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("Input must be a 1D array")
        if len(x) < self.num_qubits:
            raise ValueError(
                f"Input has {len(x)} features but num_qubits={self.num_qubits} requires at least {self.num_qubits}"
            )

        n = 2 ** self.num_qubits
        state = np.ones(n, dtype=np.complex128) / np.sqrt(n)

        if self.feature_map == 'z':
            # Z feature map: single-qubit phase rotations only
            for i in range(self.num_qubits):
                for k in range(n):
                    bit = (k >> (self.num_qubits - 1 - i)) & 1
                    state[k] *= np.exp(1j * x[i] * (1 - 2 * bit))
        elif self.feature_map == 'zz':
            # ZZ feature map: single rotations + pairwise interactions
            for i in range(self.num_qubits):
                for k in range(n):
                    bit = (k >> (self.num_qubits - 1 - i)) & 1
                    state[k] *= np.exp(1j * x[i] * (1 - 2 * bit))
            # Pairwise ZZ interactions
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    for k in range(n):
                        bit_i = (k >> (self.num_qubits - 1 - i)) & 1
                        bit_j = (k >> (self.num_qubits - 1 - j)) & 1
                        parity = 1 - 2 * (bit_i ^ bit_j)
                        state[k] *= np.exp(1j * x[i] * x[j] * parity)

        return state / np.linalg.norm(state)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value K(x1, x2)."""
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("Inputs must be 1D arrays")
        s1 = self._encode(x1)
        s2 = self._encode(x2)
        return float(np.abs(np.vdot(s1, s2)) ** 2)

    def compute_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if Y is not None:
            Y = np.asarray(Y, dtype=float)
            if Y.ndim != 2:
                raise ValueError("Y must be a 2D array")
        else:
            Y = X
        n_x, n_y = len(X), len(Y)
        K = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):
                K[i, j] = self.compute(X[i], Y[j])
        return K


__all__ = ["QuantumKernel"]
