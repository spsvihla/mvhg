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
 * @brief Draws samples from a hypergeometric distribution.
 * 
 * This function generates random samples from the standard hypergeometric
 * distribution, defined by drawing `n` items without replacement from a 
 * population of size `N` containing `K` "success" states.
 * 
 * @param N Total number of items in the population.
 * @param K Number of "success" items in the population.
 * @param n Number of draws (without replacement).
 * @param num_samples Number of independent samples to generate.
 * @param num_max_iter Maximum number of iterations of the rejection sampler.
 * @param seed Optional random seed for reproducibility.
 * 
 * @return py::array_t<int> A 1D NumPy array of length `num_samples`, 
 *         containing the number of successes observed in each draw.
 */
py::array_t<int> hypergeometric(int N, int K, int n, std::size_t num_samples, 
                                std::size_t num_max_iter, 
                                std::optional<unsigned int> seed);

/**
 * @brief Draws samples from a multivariate hypergeometric distribution.
 * 
 * This function generalizes the hypergeometric distribution to multiple 
 * categories. Given a population divided into groups of sizes `Ns`, it 
 * simulates drawing `Na` items without replacement and counts how many 
 * items are drawn from each category.
 * 
 * @param Ns Vector of category counts in the population (e.g., [N1, N2, ..., Nk]).
 * @param N  Total population size, must equal sum(Ns).
 * @param Na Number of draws (without replacement).
 * @param num_samples Number of independent samples to generate.
 * @param num_max_iter Maximum number of iterations of the rejection sampler.
 * @param seed Optional random seed for reproducibility.
 * 
 * @return py::array_t<int> A 2D NumPy array of shape `(num_samples, len(Ns))`, 
 *         where each row represents one sample of category counts.
 */
py::array_t<int> multivariate_hypergeometric(py::array_t<int>& Ns, int N, 
                                             int Na, std::size_t num_samples, 
                                             std::size_t num_max_iter,
                                             std::optional<unsigned int> seed);

#endif // MVHG_HPP