#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device) {

    sycl::queue queue(device);
    std::vector<float> c(size * size);
    int64_t n = static_cast<int64_t>(size);

    oneapi::mkl::blas::gemm(
        queue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        n, n, n,
        1.0f,
        b.data(), n,
        a.data(), n,
        0.0f,
        c.data(), n
    );

    queue.wait_and_throw();
    return c;
}