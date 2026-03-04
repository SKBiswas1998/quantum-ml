# Quantum Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen)](#testing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> When your feature space is a Hilbert space — quantum kernels that compute similarity where classical methods can't reach.

Quantum kernel methods for machine learning, built from scratch on NumPy. Encode classical data into quantum states, then measure overlap in exponentially large feature spaces.

## What You Can Do

```python
from quantum_ml.kernels import QuantumKernel
import numpy as np

# -- Compute similarity between two data points --
kernel = QuantumKernel(num_qubits=2, feature_map='zz')
k = kernel.compute(np.array([0.5, 0.3]), np.array([0.2, 0.8]))
print(k)  # 0.219 — similarity in quantum feature space

# -- Full kernel matrix for a dataset --
X_train = np.random.default_rng(42).random((50, 2))
K = kernel.compute_matrix(X_train)
# K is 50x50, symmetric, diagonal = 1.0
# Plug into any kernel SVM: sklearn.svm.SVC(kernel='precomputed').fit(K, y)

# -- Two feature maps with different expressivity --
k_z  = QuantumKernel(num_qubits=2, feature_map='z').compute(x, y)   # Single-qubit rotations
k_zz = QuantumKernel(num_qubits=2, feature_map='zz').compute(x, y)  # + Pairwise ZZ interactions
# zz captures correlations between features that z cannot
```

## Feature Maps

| Map | Encoding | Expressivity |
|-----|----------|-------------|
| `'z'` | Single-qubit phase rotations: `exp(i * x_i * Z_i)` | Linear-like separations |
| `'zz'` | Z rotations + pairwise ZZ interactions: `exp(i * x_i * x_j * Z_i Z_j)` | Captures feature correlations — higher expressivity |

**The kernel formula:**  K(x, x') = \|<phi(x)\|phi(x')>\|^2

Both maps start from uniform superposition, apply phase encoding, then compute the squared overlap of the resulting quantum states.

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `kernels/` | Quantum kernel with Z and ZZ feature maps | Complete |
| `variational/` | VQE, QAOA | Planned |
| `neural/` | Quantum neural networks | Planned |

## Quick Start

```bash
git clone https://github.com/SKBiswas1998/quantum-ml.git
cd quantum-ml
pip install -r requirements.txt
```

### Use with scikit-learn

```python
from sklearn.svm import SVC
from quantum_ml.kernels import QuantumKernel
import numpy as np

# Your data
X_train, y_train = np.random.rand(100, 2), np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 2)

# Quantum kernel matrix
qk = QuantumKernel(num_qubits=2, feature_map='zz')
K_train = qk.compute_matrix(X_train)
K_test  = qk.compute_matrix(X_test, X_train)

# Train SVM with precomputed quantum kernel
clf = SVC(kernel='precomputed').fit(K_train, y_train)
predictions = clf.predict(K_test)
```

## Project Structure

```
quantum-ml/
├── src/quantum_ml/
│   ├── kernels/
│   │   └── __init__.py       # QuantumKernel with Z and ZZ feature maps
│   ├── variational/          # VQE, QAOA (planned)
│   └── neural/               # Quantum neural networks (planned)
├── tests/
│   └── test_kernel.py        # 23 tests — creation, compute, matrix, encoding
└── conftest.py               # Pytest configuration
```

## Testing

```bash
python -m pytest tests/ -v
```

```
23 passed in 0.38s
```

**Test coverage highlights:**
- K(x, x) = 1.0 for all feature maps (self-similarity)
- K(x, y) = K(y, x) (symmetry)
- All kernel values in [0, 1] verified over 20 random pairs
- Z and ZZ feature maps produce measurably different results
- Distant points have lower similarity than identical points
- Kernel matrix is symmetric with diagonal = 1.0
- Rectangular kernel matrices (X vs Y) work correctly
- Encoded states are normalized and deterministic
- Input validation: wrong dimensions, wrong types, too few features

## Key Design Decisions

- **Real feature maps**: `'z'` and `'zz'` are genuinely different encodings — ZZ adds pairwise interaction terms that capture feature correlations
- **Strict validation**: Raises `ValueError` if input has fewer features than qubits (was silently truncating before)
- **Normalized states**: All encoded quantum states are normalized to unit length
- **scikit-learn compatible**: `compute_matrix()` output works directly with `SVC(kernel='precomputed')`

## References

1. Havlicek et al. (2019). *Supervised learning with quantum-enhanced feature spaces.* Nature.
2. Schuld & Killoran (2019). *Quantum machine learning in feature Hilbert spaces.* PRL.
3. Peruzzo et al. (2014). *A variational eigenvalue solver on a photonic quantum processor.*

## License

MIT License

---
*Part of the Quantum Computing Portfolio by SK Biswas*
