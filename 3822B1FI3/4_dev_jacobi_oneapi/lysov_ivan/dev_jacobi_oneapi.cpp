#include "dev_jacobi_oneapi.h"

#include <vector>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const size_t n = b.size();

    if (n == 0) {
        return {};
    }

    if (a.size() != n * n) {
        return {};
    }

    if (accuracy < 0.0f) {
        accuracy = 0.0f;
    }

    sycl::queue q(device);

    float* d_a = sycl::malloc_device<float>(a.size(), q);
    float* d_b = sycl::malloc_device<float>(b.size(), q);
    float* d_x_old = sycl::malloc_device<float>(n, q);
    float* d_x_new = sycl::malloc_device<float>(n, q);
    float* d_diff = sycl::malloc_device<float>(n, q);

    if (d_a == nullptr || d_b == nullptr || d_x_old == nullptr ||
        d_x_new == nullptr || d_diff == nullptr) {
        if (d_a) sycl::free(d_a, q);
        if (d_b) sycl::free(d_b, q);
        if (d_x_old) sycl::free(d_x_old, q);
        if (d_x_new) sycl::free(d_x_new, q);
        if (d_diff) sycl::free(d_diff, q);
        return {};
    }

    std::vector<float> zeros(n, 0.0f);
    std::vector<float> result(n, 0.0f);

    q.memcpy(d_a, a.data(), a.size() * sizeof(float));
    q.memcpy(d_b, b.data(), b.size() * sizeof(float));
    q.memcpy(d_x_old, zeros.data(), n * sizeof(float));
    q.memcpy(d_x_new, zeros.data(), n * sizeof(float));
    q.wait();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const size_t row_offset = i * n;

            float row_sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    row_sum += d_a[row_offset + j] * d_x_old[j];
                }
            }

            const float diag = d_a[row_offset + i];
            const float new_value = (d_b[i] - row_sum) / diag;

            d_x_new[i] = new_value;
            d_diff[i] = sycl::fabs(new_value - d_x_old[i]);
        }).wait();

        float max_diff = 0.0f;

        {
            sycl::buffer<float, 1> max_buffer(&max_diff, sycl::range<1>(1));

            q.submit([&](sycl::handler& h) {
                auto max_reduction =
                    sycl::reduction(max_buffer, h, sycl::maximum<float>());

                h.parallel_for(
                    sycl::range<1>(n),
                    max_reduction,
                    [=](sycl::id<1> idx, auto& max_val) {
                        max_val.combine(d_diff[idx]);
                    });
            }).wait();
        }

        if (max_diff < accuracy) {
            break;
        }

        q.memcpy(d_x_old, d_x_new, n * sizeof(float)).wait();
    }

    q.memcpy(result.data(), d_x_new, n * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_x_old, q);
    sycl::free(d_x_new, q);
    sycl::free(d_diff, q);

    return result;
}