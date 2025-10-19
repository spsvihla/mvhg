/**
 * @file mvhg.cpp
 * @author Sean Svihla
 */

// standard library includes
#include <iostream>
#include <random>
#include <vector>

// Pybind11 includes
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


// contact point from Hormann (1996)
struct ContactPoint {
    int k0;
    double pk0;
};

// Uniform(0, 1)
inline double
rand_uniform_double(std::mt19937& rng)
{
    return (rng() + 0.5) / (rng.max() + 1.0);
}

inline double
get_pk(int N, int K, int n, int k)
{
    // compute PMF using factorial ratios via lgamma once
    double log_pk = std::lgamma(K + 1) - std::lgamma(k + 1) - std::lgamma(K - k + 1)
                  + std::lgamma(N - K + 1)
                  - std::lgamma(n - k + 1) - std::lgamma(N - K - n + k + 1)
                  - std::lgamma(N + 1) + std::lgamma(n + 1) + std::lgamma(N - n + 1);
    return std::exp(log_pk);
}

inline double 
get_pk_prev(int N, int K, int n, int k, double pk)
{
    // recurrence formula: p_{k-1} = p_k / r_{k-1}
    double r = (K - (k-1)) * (n - (k-1)) / static_cast<double>(k * (N - K - n + k));
    return pk / r;
}

inline double
get_pk_next(int N, int K, int n, int k, double pk)
{
    // recurrence formula: p_{k+1} = p_k * r_k
    double r = (K - k) * (n - k) / static_cast<double>((k + 1) * (N - K - n + k + 1));
    return pk * r;
}

inline int
get_mode(int N, int K, int n)
{
    return static_cast<int>(std::floor((n + 1) * (K + 1) / static_cast<double>(N + 2)));
}

inline int
get_k_min(int N, int K, int n)
{
    return std::max(0, n + K - N);
}

inline int
get_k_max(int N, int K, int n)
{
    return std::min(K, n);
}

// find right contact point; see Hormann (1996)
ContactPoint
find_contact_point(int N, int K, int n, int km, double pm)
{
    int k_max = get_k_max(N, K, n);

    int k10 = km;                                           // left bound of search
    double pk10 = pm;                                       // probability of left bound
    double pk11 = get_pk_next(N, K, n, k10, pk10);          // probability of right bound
    double rk0 = pk11 / pk10;

    while(k10 < k_max - 1)
    {
        double pk12 = get_pk_next(N, K, n, k10 + 1, pk11);
        double rk1 = pk12 / pk11;
        double xk0 = std::log(rk0) - std::log1p(1.0 / (km - k10));
        double xk1 = std::log(rk1) - std::log1p(1.0 / (km - (k10 + 1)));
        if(xk1 * xk0 <= 0)
        {
            return ContactPoint{k10+1, pk11};
        }
        pk10 = pk11;                                        // move search interval right
        pk11 = pk12;
        rk0 = rk1;
        k10++;
    }

    // k10 = k_max - 1
    double xk0 = std::log(rk0) - std::log1p(1.0 / (km - k10));
    double xk1 = -std::log1p(1.0 / (km - (k10 + 1)));
    if(xk1 * xk0 <= 0)
    {
        return ContactPoint{k10+1, pk11};
    }

    throw std::runtime_error("mvhg.cpp:find_contact_point(): Failed to find contact point.");
}

// rejection-inversion for log-concave (RILC) algorithm; see Horman (1996)
int
sample_tail(int N, int K, int n, int km, std::size_t num_max_iter, 
            std::mt19937& rng)
{
    // find contact points
    double pm = get_pk(N, K, n, km);
    auto [k0, pk0] = find_contact_point(N, K, n, km, pm);

    // define hat function h(x) = a * exp(-b * x)
    double b = std::log(pk0) - std::log(get_pk_prev(N, K, n, k0, pk0));
    double a = pk0 * std::exp(-b * k0);

    // define H(x) = -Integral[x, infinity, h(t)dt] and H^{-1}
    auto H = [=](double x) {
        return a * std::exp(b * x) / b;
    };
    auto Hinv = [=](double y) {
        return std::log(b * y / a) / b;
    };

    // define setup variables
    double ym = H(km + 0.5) - pm;
    double xm = Hinv(ym);

    // rejection-inversion loop
    for(std::size_t i = 0; i < num_max_iter; ++i)
    {
        double u = rand_uniform_double(rng);
        u = u * ym;
        double x = Hinv(u);
        int k = static_cast<int>(std::floor(x + 0.5));
        if(k <= k0 and k - x <= km - xm)                    // squeeze region
        {
            return k;
        }
        else if(u >= H(k + 0.5) - get_pk(N, K, n, k))       // rejection-inversion region
        {
            return k;
        }
    }

    throw std::runtime_error("Rejection sampler failed to converge in 'num_max_iter' iterations.");
}

int
draw(int N, int K, int n, std::size_t num_max_iter, std::mt19937& rng)
{
    if(n == 0 || K == 0)
    {
        return 0;
    }
    if(n == N)
    {
        return K;
    }

    int k_min = get_k_min(N, K, n);                 // minimum of support
    int k_max = get_k_max(N, K, n);                 // maximum of support

    int km = get_mode(N, K, n);                     // mode
    double pm = get_pk(N, K, n, km);                // probability of mode
    int km_ = get_mode(N, N - K, n);                // mode used for sampling from left tail

    double pk = get_pk(N, K, n, km + 1);
    double volr = pk;                               // right tail volume
    for(int k = km + 1; k < k_max; ++k)
    {
        pk = get_pk_next(N, K, n, k, pk);
        volr += pk;
    }

    pk = get_pk(N, K, n, k_min);
    double voll = pk;                               // left tail volume
    for(int k = k_min; k < km - 1; ++k)
    {
        pk = get_pk_next(N, K, n, k, pk);
        voll += pk;
    }

    double volt = voll + volr + pm;                 // total volume

    double u = rand_uniform_double(rng) * volt;
    if(u < pm)                                      // draw from mode
    {
        return km;
    }
    else if(u < pm + volr)                          // draw from right tail
    {
        return sample_tail(N, K, n, km + 1, num_max_iter, rng);
    }
    else                                            // draw from left tail
    {
        return n - sample_tail(N, N - K, n, km_ + 1, num_max_iter, rng);
    }
}

py::array_t<int>
hypergeometric(int N, int K, int n, std::size_t num_samples, 
               std::size_t num_max_iter, std::optional<unsigned int> seed)
{
    py::array_t<int> output(num_samples);
    int* buf = output.mutable_data();

    unsigned int seed_ = seed.value_or(std::random_device{}());
    std::mt19937 rng(seed_);

    #pragma omp parallel for
    for(std::size_t i = 0; i < num_samples; ++i)
    {
        buf[i] = draw(N, K, n, num_max_iter, rng);
    }

    return output;
}

py::array_t<int>
multivariate_hypergeometric(const std::vector<int>& Ns, int N, int Na,
                            std::size_t num_samples, std::size_t num_max_iter,
                            std::optional<unsigned int> seed)
{
    // shape = (num_samples, Ns.size())
    const ssize_t n_rows = static_cast<ssize_t>(num_samples);
    const ssize_t n_cols = static_cast<ssize_t>(Ns.size());

    py::array_t<int> output({n_rows, n_cols});
    int* buf = output.mutable_data();

    unsigned int seed_ = seed.value_or(std::random_device{}());
    std::mt19937 rng(seed_);

    // Fill in parallel (each row = one sample)
    #pragma omp parallel for
    for (std::size_t i = 0; i < num_samples; ++i)
    {
        int Nsum = 0;
        int Xsum = 0;
        int* row = buf + i * n_cols;  // pointer to start of row i

        for (std::size_t j = 0; j < Ns.size(); ++j)
        {
            row[j] = draw(N - Nsum, Na - Xsum, Ns[j], num_max_iter, rng);
            Xsum += row[j];
            Nsum += Ns[j];
        }
    }

    return output;
}
