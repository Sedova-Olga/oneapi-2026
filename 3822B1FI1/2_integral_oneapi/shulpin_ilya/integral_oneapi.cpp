#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) {
        return 0.0f;
    }

    const float dx = (end - start) / static_cast<float>(count);
    const float dy = dx;

    float result = 0.0f;

    try {
        sycl::queue q{device};

        {
            sycl::buffer<float> total_buf{&result, sycl::range<1>{1}};

            q.submit([&](sycl::handler& h) {
                auto red = sycl::reduction(total_buf, h, sycl::plus<float>());

                h.parallel_for(
                    sycl::range<2>(static_cast<size_t>(count), static_cast<size_t>(count)),
                    red,
                    [=](sycl::id<2> idx, auto& partial_sum) {
                        const size_t ix = idx[1];
                        const size_t iy = idx[0];

                        const float x_mid = start + (static_cast<float>(ix) + 0.5f) * dx;
                        const float y_mid = start + (static_cast<float>(iy) + 0.5f) * dy;

                        const float f_mid = sycl::sin(x_mid) * sycl::cos(y_mid);

                        partial_sum += f_mid * dx * dy;
                    });
            }).wait();
        }

        return result;
    }
    catch (sycl::exception const& e) {
        return 0.0f;
    }
}