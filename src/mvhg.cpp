/**
 * @file mvhg.cpp
 * @author Sean Svihla
 */

// standard library includes
#include <random>
#include <vector>

// Pybind11 includes
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


// Uniform(0, 1)
inline double
rand_uniform_double(std::mt19937& rng)
{
    return (rng() + 0.5) / (rng.max() + 1.0);
}

double
get_pk(int N, int K, int n, int k)
{
    // compute PMF using factorial ratios via lgamma once
    double log_pk = std::lgamma(K + 1) - std::lgamma(k + 1) - std::lgamma(K - k + 1)
                  + std::lgamma(N - K + 1)
                  - std::lgamma(n - k + 1) - std::lgamma(N - K - n + k + 1)
                  - std::lgamma(N + 1) + std::lgamma(n + 1) + std::lgamma(N - n + 1);
    return std::exp(log_pk);
}

double 
get_pk_prev(int N, int K, int n, int k, double pk)
{
    // recurrence formula: p_{k-1} = p_k / r_{k-1}
    double r = (K - (k-1)) * (n - (k-1)) / static_cast<double>(k * (N - K - n + k));
    return pk / r;
}

double
get_pk_next(int N, int K, int n, int k, double pk)
{
    // recurrence formula: p_{k+1} = p_k * r_k
    double r = (K - k) * (n - k) / static_cast<double>((k + 1) * (N - K - n + k + 1));
    return pk * r;
}

// hypergeometric sampler (Horman)
void 
hypergeometric_(int* out, int N, int K, int n, std::mt19937& rng)
{
    if(n == 0 || K == 0)
    {
        *out = 0;
    }
    if(n == N)
    {
        *out = K;
    }

    int k_min = std::max(0, n + K - N);
    int k_max = std::min(K, n);

    // compute mode
    double m = (n + 1) * (K + 1) / static_cast<double>(N + 2);
    int km = static_cast<int>(std::floor(m));               // mode index

    // find contact point
    // see Horman (1996) Corollary 1 and Remarks 1-2
    int k10 = km;                                           // left bound of right search
    int k01 = km;                                           // right bound of left search
    double pm = get_pk(N, K, n, km);                        // mode probability
    double pk10 = pm;                                       // right-left probability
    double pk01 = pk10;                                     // left-right probability

    double x0;                                              // contact point
    int k0;                                                 // right bound of contact interval
    double pk0;                                             // probability of k0

    bool did_solve = false;
    while(k01 >= k_min || k10 <= k_max)
    {
        if(k01 >= k_min)
        {
            double pk00 = get_pk_prev(N, K, n, k01, pk01);  // left bound probability
            double rk0 = pk01 / pk00;                       // hypergeometric recursion ratio
            double x0_ = km - 1 / (rk0 - 1);                // candidate contact point
            if(x0 >= k01 - 1 && x0 <= k01 + 1)
            {
                x0 = x0_;
                k0 = static_cast<int>(std::floor(x0 + 1));
                pk0 = pk01;
                did_solve = true;
                break;
            }
            pk01 = pk00;                                    // move search interval left
            k01--;
        }
        if(k10 <= k_max)
        {
            double pk11 = get_pk_next(N, K, n, k10, pk10);  // right bound probability
            double rk1 = pk11 / pk10;                       // hypergeometric recursion ratio
            double x0_ = km - 1 / (rk1 - 1);                // candidate contact point
            if(x0 >= k10 && x0 <= k10 + 1)
            {
                x0 = x0_;
                k0 = static_cast<int>(std::floor(x0 + 1));
                pk0 = pk11;
                did_solve = true;
                break;
            }
            pk10 = pk11;                                    // move search interval right
            k10++;
        }
    }
    if(!did_solve)
    {
        throw std::runtime_error("Failed to find contact point.");
    }

    // define setup variables
    // see Horman (1996) Algorithm RILC
    double b = std::log(pk0) - std::log(get_pk_prev(N, K, n, k0, pk0));
    double a = pk0 * std::exp(-b * pk0);
    double ym = a * std::exp(b * (km + 1/2)) / b - pm;
    double xm = std::log(ym * b / a) / b;

    bool is_done = false;
    while(!is_done)
    {
        double U = rand_uniform_double(rng);
        U = U * ym;
        double x = std::log(U * b / a) / b;
        int k = static_cast<int>(std::floor(x + 1/2));
        double pk = get_pk(N, K, n, k);
        if(k <= k0 and k - x <= km - xm)
        {
            *out = k;
            is_done = true;
        }
        else if(U >= a * std::exp(b * (k + 1/2)) / b - pk)
        {
            *out = k;
            is_done = true;
        }
    }
}

py::array_t<int>
hypergeometric(int N, int K, int n, std::size_t num_samples, 
               std::optional<unsigned int> seed)
{
    std::size_t buf_size = num_samples;
    std::unique_ptr<int[]> buf;

    try 
    {
        buf = std::make_unique<int[]>(buf_size);
    } 
    catch (const std::bad_alloc&) 
    {
        throw std::runtime_error(
            "Failed to allocate memory for stationary buffer. "
            "Requested " + std::to_string(buf_size) + " ints (" +
            std::to_string(buf_size * sizeof(int) / (1024.0 * 1024.0)) +
            " MiB)."
        );
    }

    unsigned int seed_ = seed.value_or(std::random_device{}());
    std::mt19937 rng(seed_);

    for (std::size_t i = 0; i < num_samples; ++i)
    {
        hypergeometric_(&buf[i], N, K, n, rng);
    }

    py::array_t<int> output(
        {static_cast<py::ssize_t>(num_samples)},
        {static_cast<py::ssize_t>(sizeof(int))},
        buf.get(),
        py::capsule(buf.release(), [](void* p) { delete[] reinterpret_cast<int*>(p); })
    );

    return output;
}

py::array_t<int>
multivariate_hypergeometric(const std::vector<int>& Ns, int N, int Na,
                            std::size_t num_samples, 
                            std::optional<unsigned int> seed)
{
    std::size_t buf_size = Ns.size() * num_samples;
    std::unique_ptr<int[]> buf;

    try 
    {
        buf = std::make_unique<int[]>(buf_size);
    } 
    catch (const std::bad_alloc&) 
    {
        throw std::runtime_error(
            "Failed to allocate memory for stationary buffer. "
            "Requested " + std::to_string(buf_size) + " ints (" +
            std::to_string(buf_size * sizeof(int) / (1024.0 * 1024.0)) +
            " MiB)."
        );
    }

    unsigned int seed_ = seed.value_or(std::random_device{}());
    std::mt19937 rng(seed_);

    #pragma omp parallel for
    for (std::size_t i = 0; i < num_samples; ++i) 
    {
        int Nsum = 0;
        int Xsum = 0;
        for (std::size_t j = 0; j < Ns.size(); ++j) 
        {
            hypergeometric_(&buf[Ns.size() * i + j], N - Nsum, Na - Xsum, Ns[j], rng);
            Xsum += buf[Ns.size() * i + j];
            Nsum += Ns[j];
        }
    }

    py::array_t<int> output(
        {static_cast<py::ssize_t>(Ns.size()),
         static_cast<py::ssize_t>(num_samples)},
        {static_cast<py::ssize_t>(num_samples * sizeof(int)),
         static_cast<py::ssize_t>(sizeof(int))},
        buf.get(),
        py::capsule(buf.release(), [](void* p) { delete[] reinterpret_cast<int*>(p); })
    );

    return output;
}
