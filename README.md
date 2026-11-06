# mvhg

[![PyPI version](https://img.shields.io/pypi/v/mvhg.svg)](https://pypi.org/project/mvhg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python C++ extension for sampling hypergeometric and multivariate hypergeometric distributions using the Rejection-Inversion for Log Concave Distributions (RILC) Algorithm [1-2].

## Installation

### From PyPI
```bash
pip install mvhg
```

### From Source
To build and install `mvhg` locally, clone the repository and install via `pip`:

```bash
git clone https://github.com/spsvihla/mvhg.git
cd mvhg
pip install .
```

*Note: Requires a C++17-compatible compiler, CMake (>= 3.15), and `scikit-build-core`.*

## References

1. Hörmann, W. A universal generator for discrete log-concave distributions. Computing 52, 89–96 (1994).
2. Hörmann, W. & Derflinger, G. Rejection-inversion to generate variates from monotone discrete distributions. ACM Trans. Model. Comput. Simul. 6, 169–184 (1996).

## License

Distributed under the MIT License.
