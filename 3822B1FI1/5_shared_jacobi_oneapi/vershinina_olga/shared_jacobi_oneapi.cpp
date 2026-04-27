#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <utility>
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    sycl::queue queue(device);
    size_t n = b.size();
    float* s_a = sycl::malloc_shared<float>(n * n, queue);
    float* s_b = sycl::malloc_shared<float>(n, queue);
    float* s_x_cur = sycl::malloc_shared<float>(n, queue);
    float* s_x_nxt = sycl::malloc_shared<float>(n, queue);
    float* s_diff = sycl::malloc_shared<float>(1, queue);

    std::copy(a.begin(), a.end(), s_a);
    std::copy(b.begin(), b.end(), s_b);
    std::fill(s_x_cur, s_x_cur + n, 0.0f);
    std::fill(s_x_nxt, s_x_nxt + n, 0.0f);

    float* cur_ptr = s_x_cur;
    float* nxt_ptr = s_x_nxt;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        *s_diff = 0.0f;
        queue.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(s_diff, sycl::maximum<float>());

            h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> idx, auto& max_val) {
                size_t i = idx[0];
                float sum = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += s_a[i * n + j] * cur_ptr[j];
                    }
                }
                nxt_ptr[i] = (s_b[i] - sum) / s_a[i * n + i];
                max_val.combine(sycl::fabs(nxt_ptr[i] - cur_ptr[i]));
                });
            }).wait();

            if (*s_diff < accuracy) {
                std::swap(cur_ptr, nxt_ptr);
                break;
            }
            std::swap(cur_ptr, nxt_ptr);
    }

    std::vector<float> result(n);
    std::copy(cur_ptr, cur_ptr + n, result.begin());

    sycl::free(s_a, queue);
    sycl::free(s_b, queue);
    sycl::free(s_x_cur, queue);
    sycl::free(s_x_nxt, queue);
    sycl::free(s_diff, queue);

    return result;
}