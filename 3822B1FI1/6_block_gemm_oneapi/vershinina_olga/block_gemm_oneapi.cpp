#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device) {

    const size_t BLOCK_SIZE = 16;
    sycl::queue queue(device);

    std::vector<float> result(size * size, 0.0f);

    {
        sycl::buffer<float, 2> buf_a(a.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_b(b.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_c(result.data(), sycl::range<2>(size, size));

        queue.submit([&](sycl::handler& cgh) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(cgh);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(cgh);
            auto acc_c = buf_c.get_access<sycl::access::mode::write>(cgh);

            sycl::local_accessor<float, 2> local_a(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::local_accessor<float, 2> local_b(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            sycl::range<2> global_size(size, size);
            sycl::range<2> local_size(BLOCK_SIZE, BLOCK_SIZE);

            cgh.parallel_for(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
                size_t row = item.get_global_id(0);
                size_t col = item.get_global_id(1);
                size_t local_row = item.get_local_id(0);
                size_t local_col = item.get_local_id(1);

                float tmp = 0.0f;

                for (size_t k = 0; k < size; k += BLOCK_SIZE) {
                    local_a[local_row][local_col] = acc_a[row][k + local_col];
                    local_b[local_row][local_col] = acc_b[k + local_row][col];

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t n = 0; n < BLOCK_SIZE; ++n) {
                        tmp += local_a[local_row][n] * local_b[n][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }
                acc_c[row][col] = tmp;
                });
            });
    }
    return result;
}