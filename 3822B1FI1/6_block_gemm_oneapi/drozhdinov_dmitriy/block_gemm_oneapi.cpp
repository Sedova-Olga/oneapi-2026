#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    constexpr int BLOCK_SZ = 16;
    assert(size % BLOCK_SZ == 0);

    sycl::queue q(device);
    std::vector<float> c(size * size);
    {
        sycl::buffer<float> a_buf(a.data(), a.size());
        sycl::buffer<float> b_buf(b.data(), b.size());
        sycl::buffer<float> c_buf(c.data(), c.size());
        sycl::event e = q.submit([&](sycl::handler& handler) {
            sycl::local_accessor<float, 2> a_block(sycl::range<2>(BLOCK_SZ, BLOCK_SZ), handler);
            sycl::local_accessor<float, 2> b_block(sycl::range<2>(BLOCK_SZ, BLOCK_SZ), handler);
            auto in_a = a_buf.get_access<sycl::access::mode::read>(handler);
            auto in_b = b_buf.get_access<sycl::access::mode::read>(handler);
            auto out_c = c_buf.get_access<sycl::access::mode::write>(handler);
            handler.parallel_for(sycl::nd_range<2>(sycl::range<2>(size, size), sycl::range<2>(BLOCK_SZ, BLOCK_SZ)),
                [=](sycl::nd_item<2> item) {
                    const int ii = item.get_local_id(0);
                    const int jj = item.get_local_id(1);
                    const int i = item.get_global_id(0);
                    const int j = item.get_global_id(1);

                    float res = 0.0f;

                    for (int k_block = 0; k_block < size / BLOCK_SZ; k_block++) {
                        a_block[ii][jj] = in_a[i * size + (BLOCK_SZ * k_block + jj)];
                        b_block[ii][jj] = in_b[(BLOCK_SZ * k_block + ii) * size + j];
                        item.barrier(sycl::access::fence_space::local_space);
                        for (int k = 0; k < BLOCK_SZ; k++) {
                            res += a_block[ii][k] * b_block[k][jj];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    out_c[i * size + j] = res;
                });
            });
        e.wait();
    }
    
    return c;
}