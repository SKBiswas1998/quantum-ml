"""Tests for QuantumKernel."""
import pytest
import numpy as np
from quantum_ml.kernels import QuantumKernel


class TestKernelCreation:
    def test_default_creation(self):
        k = QuantumKernel()
        assert k.num_qubits == 2
        assert k.feature_map == 'zz'

    def test_custom_num_qubits(self):
        k = QuantumKernel(num_qubits=3)
        assert k.num_qubits == 3

    def test_z_feature_map(self):
        k = QuantumKernel(num_qubits=2, feature_map='z')
        assert k.feature_map == 'z'

    def test_invalid_num_qubits_zero(self):
        with pytest.raises(ValueError):
            QuantumKernel(num_qubits=0)

    def test_invalid_num_qubits_negative(self):
        with pytest.raises(ValueError):
            QuantumKernel(num_qubits=-1)

    def test_invalid_feature_map(self):
        with pytest.raises(ValueError, match="feature_map"):
            QuantumKernel(feature_map='invalid')

    def test_num_qubits_one(self):
        k = QuantumKernel(num_qubits=1)
        x = np.array([0.5])
        val = k.compute(x, x)
        assert np.isclose(val, 1.0, atol=0.01)


class TestKernelCompute:
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

    def test_kernel_value_in_range(self):
        """Kernel value should be in [0, 1]."""
        kernel = QuantumKernel(num_qubits=2)
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.random(2)
            y = rng.random(2)
            k = kernel.compute(x, y)
            assert 0.0 <= k <= 1.0 + 1e-10

    def test_different_feature_maps_give_different_results(self):
        """'z' and 'zz' should produce different kernel values (in general)."""
        x = np.array([1.5, 2.0])
        y = np.array([0.2, 2.8])
        k_z = QuantumKernel(num_qubits=2, feature_map='z').compute(x, y)
        k_zz = QuantumKernel(num_qubits=2, feature_map='zz').compute(x, y)
        # They should differ because zz adds pairwise interaction terms
        assert not np.isclose(k_z, k_zz, atol=0.01)

    def test_kernel_orthogonal_inputs(self):
        """Very different inputs should have lower kernel value than identical."""
        kernel = QuantumKernel(num_qubits=2)
        x = np.array([0.0, 0.0])
        y = np.array([3.14, 3.14])
        k_same = kernel.compute(x, x)
        k_diff = kernel.compute(x, y)
        assert k_same > k_diff

    def test_input_too_short_raises(self):
        """Input shorter than num_qubits should raise ValueError."""
        kernel = QuantumKernel(num_qubits=3)
        with pytest.raises(ValueError, match="features"):
            kernel.compute(np.array([0.5, 0.3]), np.array([0.5, 0.3]))

    def test_2d_input_raises(self):
        """2D input array should raise ValueError."""
        kernel = QuantumKernel(num_qubits=2)
        with pytest.raises(ValueError):
            kernel.compute(np.array([[0.5, 0.3]]), np.array([0.5, 0.3]))


class TestKernelMatrix:
    def test_kernel_matrix_shape_square(self):
        kernel = QuantumKernel(num_qubits=2)
        X = np.random.default_rng(42).random((10, 2))
        K = kernel.compute_matrix(X)
        assert K.shape == (10, 10)

    def test_kernel_matrix_symmetric(self):
        kernel = QuantumKernel(num_qubits=2)
        X = np.random.default_rng(42).random((5, 2))
        K = kernel.compute_matrix(X)
        assert np.allclose(K, K.T)

    def test_kernel_matrix_diagonal_ones(self):
        """Diagonal of K(X, X) should be close to 1."""
        kernel = QuantumKernel(num_qubits=2)
        X = np.random.default_rng(42).random((5, 2))
        K = kernel.compute_matrix(X)
        assert np.allclose(np.diag(K), 1.0, atol=0.01)

    def test_kernel_matrix_rectangular(self):
        """compute_matrix(X, Y) should give rectangular matrix."""
        kernel = QuantumKernel(num_qubits=2)
        X = np.random.default_rng(42).random((5, 2))
        Y = np.random.default_rng(43).random((3, 2))
        K = kernel.compute_matrix(X, Y)
        assert K.shape == (5, 3)

    def test_kernel_matrix_values_in_range(self):
        """All kernel matrix values should be in [0, 1]."""
        kernel = QuantumKernel(num_qubits=2)
        X = np.random.default_rng(42).random((5, 2))
        K = kernel.compute_matrix(X)
        assert np.all(K >= -1e-10)
        assert np.all(K <= 1.0 + 1e-10)

    def test_kernel_matrix_invalid_1d(self):
        kernel = QuantumKernel(num_qubits=2)
        with pytest.raises(ValueError):
            kernel.compute_matrix(np.array([1.0, 2.0]))


class TestKernelEncoding:
    def test_encode_produces_normalized_state(self):
        """Encoded state should have unit norm."""
        kernel = QuantumKernel(num_qubits=2)
        x = np.array([0.5, 0.3])
        state = kernel._encode(x)
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_encode_state_size(self):
        """Encoded state should have 2^n elements."""
        kernel = QuantumKernel(num_qubits=3)
        x = np.array([0.5, 0.3, 0.1])
        state = kernel._encode(x)
        assert len(state) == 8

    def test_z_encode_deterministic(self):
        """Same input should always produce same state."""
        kernel = QuantumKernel(num_qubits=2, feature_map='z')
        x = np.array([0.5, 0.3])
        s1 = kernel._encode(x)
        s2 = kernel._encode(x)
        assert np.allclose(s1, s2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
