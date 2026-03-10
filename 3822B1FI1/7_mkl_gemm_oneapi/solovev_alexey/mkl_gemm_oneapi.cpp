#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device) {

    const size_t N = size;

    sycl::queue q(device);

    float* A = sycl::malloc_shared<float>(N * N, q);
    float* B = sycl::malloc_shared<float>(N * N, q);
    float* C = sycl::malloc_shared<float>(N * N, q);

    for (size_t i = 0; i < N * N; ++i) {
        A[i] = a[i];
        B[i] = b[i];
        C[i] = 0.0f;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    oneapi::mkl::blas::column_major::gemm(
        q,
        oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans,
        N,
        N,
        N,
        alpha,
        A,
        N,
        B,
        N,
        beta,
        C,
        N
    ).wait();

    std::vector<float> result(N * N);

    for (size_t i = 0; i < N * N; ++i)
        result[i] = C[i];

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return result;
}
