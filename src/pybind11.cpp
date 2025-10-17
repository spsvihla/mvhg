// Project-specific inlcudes
#include "mvhg.hpp"

// Pybind11 includes
#include <pybind11/pybind11.h>

namespace py = pybind11;


PYBIND11_MODULE(_mvhg, m) {
    m.def
    (
        "hypergeometric", 
        &hypergeometric,
        py::arg("N"),
        py::arg("K"),
        py::arg("n"),
        py::arg("num_samples"),
        py::arg("seed")
    );
    m.def
    (
        "multivariate_hypergeometric",
        &multivariate_hypergeometric,
        py::arg("Ns"),
        py::arg("N"),
        py::arg("Na"),
        py::arg("num_samples"),
        py::arg("seed")
    );
}