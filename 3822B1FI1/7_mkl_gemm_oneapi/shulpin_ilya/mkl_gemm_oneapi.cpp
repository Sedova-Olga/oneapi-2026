#include "mkl_gemm_oneapi.h"

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device)
{
    std::vector<float> c(size * size, 0.0f);

    try {
        sycl::queue q{device};

        sycl::buffer<float, 1> A_buf{a.data(), sycl::range<1>{size * size}};
        sycl::buffer<float, 1> B_buf{b.data(), sycl::range<1>{size * size}};
        sycl::buffer<float, 1> C_buf{c.data(), sycl::range<1>{size * size}};

        oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
        oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

        int64_t m = static_cast<int64_t>(size);
        int64_t n = static_cast<int64_t>(size);
        int64_t k = static_cast<int64_t>(size);

        oneapi::mkl::blas::row_major::gemm(
            q,
            transA, transB,
            m, n, k,
            1.0f,
            A_buf, size,
            B_buf, size,
            0.0f,
            C_buf, size
        );

        q.wait();

        return c;

    } catch (sycl::exception const& e) {
        return {};
    }
}