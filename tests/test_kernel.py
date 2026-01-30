"""Tests for QuantumKernel."""
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from quantum_ml.kernels import QuantumKernel

class TestQuantumKernel:
    def test_kernel_self_is_one(self):
        """K(x, x) should be close to 1."""
        kernel = QuantumKernel(num_qubits=2)
        x = np.array([0.5, 0.3])
        k = kernel.compute(x, x)
        assert np.isclose(k, 1.0, atol=0.01)
    
    def test_kernel_symmetry(self):
        """K(x, y) should equal K(y, x)."""
        kernel = QuantumKernel(num_qubits=2)
        x = np.array([0.5, 0.3])
        y = np.array([0.2, 0.8])
        assert np.isclose(kernel.compute(x, y), kernel.compute(y, x))
    
    def test_kernel_matrix_shape(self):
        """Kernel matrix should have correct shape."""
        kernel = QuantumKernel(num_qubits=2)
        X = np.random.rand(10, 2)
        K = kernel.compute_matrix(X)
        assert K.shape == (10, 10)
    
    def test_kernel_matrix_symmetric(self):
        """Kernel matrix should be symmetric."""
        kernel = QuantumKernel(num_qubits=2)
        X = np.random.rand(5, 2)
        K = kernel.compute_matrix(X)
        assert np.allclose(K, K.T)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
