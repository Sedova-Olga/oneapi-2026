#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a,const std::vector<float>& b,size_t size,sycl::device device) {
    constexpr size_t block_size = 16;
    const size_t total_elements = size * size;

    std::vector<float> result(total_elements, 0.0f);

    sycl::queue queue(device);

    float* dev_a = sycl::malloc_device<float>(total_elements, queue);
    float* dev_b = sycl::malloc_device<float>(total_elements, queue);
    float* dev_c = sycl::malloc_device<float>(total_elements, queue);

    queue.memcpy(dev_a, a.data(), total_elements * sizeof(float));
    queue.memcpy(dev_b, b.data(), total_elements * sizeof(float));

    queue.memset(dev_c, 0, total_elements * sizeof(float)).wait();

    const size_t padded_size =
        ((size + block_size - 1) / block_size) * block_size;

    sycl::range<2> global_range(padded_size, padded_size);
    sycl::range<2> local_range(block_size, block_size);

    const size_t num_blocks = size / block_size;

    queue.submit([&](sycl::handler& cgh) {

        sycl::local_accessor<float, 2> block_a(local_range, cgh);
        sycl::local_accessor<float, 2> block_b(local_range, cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item) {

                const size_t global_row = item.get_global_id(0);
                const size_t global_col = item.get_global_id(1);

                const size_t local_row = item.get_local_id(0);
                const size_t local_col = item.get_local_id(1);

                float c_value = 0.0f;

                for (size_t block_k = 0; block_k < num_blocks; ++block_k) {

                    const size_t tiled_col = block_k * block_size + local_col;
                    const size_t tiled_row = block_k * block_size + local_row;

                    if (global_row < size && tiled_col < size)
                        block_a[local_row][local_col] =
                            dev_a[global_row * size + tiled_col];
                    else
                        block_a[local_row][local_col] = 0.0f;

                    if (tiled_row < size && global_col < size)
                        block_b[local_row][local_col] =
                            dev_b[tiled_row * size + global_col];
                    else
                        block_b[local_row][local_col] = 0.0f;

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < block_size; ++k) {
                        c_value +=
                            block_a[local_row][k] *
                            block_b[k][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (global_row < size && global_col < size) {
                    dev_c[global_row * size + global_col] = c_value;
                }
            });
    });

    queue.wait();

    queue.memcpy(result.data(), dev_c,
                 total_elements * sizeof(float)).wait();

    sycl::free(dev_a, queue);
    sycl::free(dev_b, queue);
    sycl::free(dev_c, queue);

    return result;
}