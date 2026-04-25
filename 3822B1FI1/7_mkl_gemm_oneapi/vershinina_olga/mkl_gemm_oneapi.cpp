#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device) {

    sycl::queue queue(device);
    std::vector<float> c(size * size, 0.0f);
    int64_t n = static_cast<int64_t>(size);

    sycl::buffer<float, 1> buf_a{ a.data(), sycl::range<1>(a.size()) };
    sycl::buffer<float, 1> buf_b{ b.data(), sycl::range<1>(b.size()) };
    sycl::buffer<float, 1> buf_c{ c.data(), sycl::range<1>(c.size()) };

    oneapi::mkl::blas::row_major::gemm(
        queue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        n, n, n,
        1.0f,
        buf_a, n,
        buf_b, n,
        0.0f,
        buf_c, n
    );

    queue.wait_and_throw();
    return c;
}