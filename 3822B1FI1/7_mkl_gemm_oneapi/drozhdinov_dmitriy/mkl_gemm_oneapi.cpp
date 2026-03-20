#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float> &a,
                                 const std::vector<float> &b, size_t size,
                                 sycl::device device) {
  std::vector<float> ans(size * size, 0.0f);

  sycl::queue queue(device);

  {
    sycl::buffer<float> buf_a(a.data(), a.size());
    sycl::buffer<float> buf_b(b.data(), b.size());
    sycl::buffer<float> buf_ans(ans.data(), ans.size());

    auto nontrans = oneapi::mkl::transpose::nontrans;

    oneapi::mkl::blas::row_major::gemm(queue, nontrans, nontrans, size, size,
                                       size, 1, buf_a, size, buf_b, size, 0,
                                       buf_ans, size);
  }

  return ans;
}