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

    *s_diff = 1.0f;

    float* cur_ptr = s_x_cur;
    float* nxt_ptr = s_x_nxt;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::item<1> item) {
            size_t i = item[0];
            float sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sum += s_a[i * n + j] * cur_ptr[j];
                }
            }
            nxt_ptr[i] = (s_b[i] - sum) / s_a[i * n + i];
            });

        queue.single_task([=]() {
            float max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float d = sycl::fabs(nxt_ptr[i] - cur_ptr[i]);
                if (d > max_diff) max_diff = d;
            }
            s_diff[0] = max_diff;
            });

        
        queue.wait();
        
        if (s_diff[0] < accuracy) {
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