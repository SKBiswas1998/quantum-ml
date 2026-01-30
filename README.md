# Quantum Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Quantum ML implementations with classical benchmarks.

## Features

- **Quantum Kernels**: QSVM, quantum feature maps
- **Variational Algorithms**: VQE, QAOA
- **Quantum Neural Networks**: Parameterized circuits
- **Benchmarks**: Quantum vs Classical comparisons

## Quick Start
```bash
git clone https://github.com/SKBiswas1998/quantum-ml.git
cd quantum-ml
pip install -r requirements.txt
```
```python
from quantum_ml.kernels import QuantumKernel

kernel = QuantumKernel(num_qubits=2)
K = kernel.compute_matrix(X_train)
```

## References

1. Havlicek et al. (2019). Supervised learning with quantum-enhanced feature spaces.
2. Peruzzo et al. (2014). A variational eigenvalue solver on a photonic quantum processor.

## License

MIT License

---
*Part of the Quantum Computing Portfolio by SK Biswas*
