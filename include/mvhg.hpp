/**
 * @file mvhg.hpp
 * @author Sean Svihla
 */
#ifndef MVHG_HPP
#define MVHG_HPP

// standard library includes
#include <vector>

// Pybind11 includes
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


/**
 * 
 */
py::array_t<int>hypergeometric(int N, int K, int n, std::size_t num_samples, 
                               std::optional<unsigned int> seed);

/**
 * 
 */
py::array_t<int> multivariate_hypergeometric(const std::vector<int>& Ns, int N, 
                                             int Na, std::size_t num_samples, 
                                             std::optional<unsigned int> seed);
#endif // MVHG_HPP