#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float result = 0.0f;
  float delta = (end - start) / count;

  {
    sycl::buffer<float> buf_result(&result, 1);
    sycl::queue queue(device);

    queue.submit([&](sycl::handler &cgh) {
      auto reduction = sycl::reduction(buf_result, cgh, sycl::plus<float>());

      cgh.parallel_for(sycl::range<2>(count, count), reduction,
                       [=](sycl::id<2> idx, auto &sum) {
                         int i = idx[0];
                         int j = idx[1];

                         float x = start + (i + 0.5f) * delta;
                         float y = start + (j + 0.5f) * delta;

                         sum += (sycl::sin(x) * sycl::cos(y));
                       });
    });

    queue.wait();
  }

  return result * delta * delta;
}