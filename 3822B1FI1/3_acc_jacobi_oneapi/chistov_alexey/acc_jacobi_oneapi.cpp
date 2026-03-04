#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(const std::vector<float>& a,const std::vector<float>& b,float accuracy,sycl::device device) {
    const size_t n = static_cast<size_t>(std::sqrt(a.size()));
    const float accuracy_sq = accuracy * accuracy;

    std::vector<float> inv_diag(n);
    for (size_t i = 0; i < n; ++i) {
        inv_diag[i] = 1.0f / a[i * n + i];
    }

    sycl::queue q(device, sycl::property::queue::in_order{});

    sycl::buffer<float, 1> A_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> B_buf(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> inv_diag_buf(inv_diag.data(), sycl::range<1>(n));

    sycl::buffer<float, 1> x_curr_buf{sycl::range<1>(n)};
    sycl::buffer<float, 1> x_next_buf{sycl::range<1>(n)};
    sycl::buffer<float, 1> norm_buf{sycl::range<1>(1)};

    q.submit([&](sycl::handler& cgh) {
        auto x_acc = x_curr_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.fill(x_acc, 0.0f);
    });

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        q.submit([&](sycl::handler& cgh) {
            auto A_acc = A_buf.get_access<sycl::access::mode::read>(cgh);
            auto B_acc = B_buf.get_access<sycl::access::mode::read>(cgh);
            auto invD_acc = inv_diag_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_curr_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_next_acc = x_next_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                size_t row_start = i * n;

                float sum = 0.0f;

                #pragma unroll 4
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += A_acc[row_start + j] * x_curr_acc[j];
                    }
                }

                x_next_acc[i] = invD_acc[i] * (B_acc[i] - sum);
            });
        });

        q.submit([&](sycl::handler& cgh) {
            auto norm_acc = norm_buf.get_access<sycl::access::mode::write>(cgh);
            cgh.fill(norm_acc, 0.0f);
        });

        q.submit([&](sycl::handler& cgh) {
            auto x_curr_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_next_acc = x_next_buf.get_access<sycl::access::mode::read>(cgh);

            auto reduction = sycl::reduction(
                norm_buf,
                cgh,
                sycl::plus<float>());

            cgh.parallel_for(
                sycl::range<1>(n),
                reduction,
                [=](sycl::id<1> idx, auto& norm_sum) {
                    size_t i = idx[0];
                    float diff = x_next_acc[i] - x_curr_acc[i];
                    norm_sum += diff * diff;
                });
        }).wait();

        float norm_sq = norm_buf.get_host_access()[0];
        if (norm_sq < accuracy_sq) {
            break;
        }

        std::swap(x_curr_buf, x_next_buf);
    }

    std::vector<float> result(n);
    q.submit([&](sycl::handler& cgh) {
        auto x_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(x_acc, result.data());
    }).wait();

    return result;
}