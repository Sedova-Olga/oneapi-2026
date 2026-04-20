#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <utility>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    sycl::queue queue(device);
    size_t n = b.size();
    std::vector<float> result(n, 0.0f);

    float* d_a = sycl::malloc_device<float>(n * n, queue);
    float* d_b = sycl::malloc_device<float>(n, queue);
    float* d_x_cur = sycl::malloc_device<float>(n, queue);
    float* d_x_nxt = sycl::malloc_device<float>(n, queue);
    float* d_diff = sycl::malloc_device<float>(1, queue);

    queue.fill(d_x_cur, 0.0f, n);
    queue.fill(d_x_nxt, 0.0f, n);
    queue.memcpy(d_a, a.data(), sizeof(float) * n * n);
    queue.memcpy(d_b, b.data(), sizeof(float) * n);
    queue.wait();

    float* cur_ptr = d_x_cur;
    float* nxt_ptr = d_x_nxt;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::item<1> item) {
            size_t i = item[0];
            float sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (j != i) sum += d_a[i * n + j] * cur_ptr[j];
            }
            nxt_ptr[i] = (d_b[i] - sum) / d_a[i * n + i];
            });

        queue.single_task([=]() {
            float max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float d = sycl::fabs(nxt_ptr[i] - cur_ptr[i]);
                if (d > max_diff) max_diff = d;
            }
            d_diff[0] = max_diff;
            });

        queue.wait();

        float current_diff = 0.0f;
        queue.memcpy(&current_diff, d_diff, sizeof(float)).wait();

        if (current_diff < accuracy) break;

        std::swap(cur_ptr, nxt_ptr);
    }

    queue.memcpy(result.data(), cur_ptr, sizeof(float) * n).wait();

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_x_cur, queue);
    sycl::free(d_x_nxt, queue);
    sycl::free(d_diff, queue);

    return result;
}