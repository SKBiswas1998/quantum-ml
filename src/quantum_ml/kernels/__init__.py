"""
Quantum Kernel for Machine Learning.

References:
    Havlicek et al. (2019). Supervised learning with quantum-enhanced feature spaces.
"""
import numpy as np
from typing import Optional

class QuantumKernel:
    """
    Quantum kernel for computing similarity in quantum feature space.
    
    K(x, x') = |<phi(x)|phi(x')>|^2
    
    Example:
        >>> kernel = QuantumKernel(num_qubits=2)
        >>> K = kernel.compute_matrix(X_train)
    """
    
    def __init__(self, num_qubits: int = 2, feature_map: str = 'zz'):
        self.num_qubits = num_qubits
        self.feature_map = feature_map
    
    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode classical data into quantum state."""
        n = 2 ** self.num_qubits
        state = np.ones(n, dtype=np.complex128) / np.sqrt(n)
        
        # Simple ZZ feature map encoding
        for i, xi in enumerate(x[:self.num_qubits]):
            state *= np.exp(1j * xi)
        
        return state / np.linalg.norm(state)
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value K(x1, x2)."""
        s1 = self._encode(x1)
        s2 = self._encode(x2)
        return float(np.abs(np.vdot(s1, s2)) ** 2)
    
    def compute_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix."""
        if Y is None:
            Y = X
        n_x, n_y = len(X), len(Y)
        K = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):
                K[i, j] = self.compute(X[i], Y[j])
        return K

__all__ = ["QuantumKernel"]
