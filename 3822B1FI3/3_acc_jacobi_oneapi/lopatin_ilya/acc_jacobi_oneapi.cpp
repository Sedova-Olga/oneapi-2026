#include "acc_jacobi_oneapi.h"

#include <algorithm>
#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <vector>

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {
  std::vector<float> curr_ans(b.size(), 0.0f);
  std::vector<float> prev_ans(b.size(), 0.0f);
  int size = b.size();
  int step = 0;
  float error = 0.0f;

  {
    sycl::buffer<float, 1> buf_a(a.data(), a.size());
    sycl::buffer<float, 1> buf_b(b.data(), b.size());
    sycl::buffer<float, 1> buf_curr(curr_ans.data(), curr_ans.size());
    sycl::buffer<float, 1> buf_prev(prev_ans.data(), prev_ans.size());
    sycl::buffer<float, 1> buf_error(&error, 1);

    while (step++ < ITERATIONS) {

      sycl::queue queue(device);

      queue.submit([&](sycl::handler &cgh) {
        auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
        auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
        auto in_prev = buf_prev.get_access<sycl::access::mode::read_write>(cgh);
        auto in_curr = buf_curr.get_access<sycl::access::mode::read_write>(cgh);

        auto reduction = sycl::reduction(buf_error, cgh, sycl::maximum<>());

        cgh.parallel_for(sycl::range<1>(size), reduction,
                         [=](sycl::id<1> id, auto &error) {
                           int i = id.get(0);
                           float curr = in_b[i];
                           for (int j = 0; j < size; j++) {
                             if (i != j) {
                               curr -= in_a[i * size + j] * in_prev[j];
                             }
                           }
                           curr /= in_a[i * size + i];
                           in_curr[i] = curr;

                           float diff = sycl::fabs(curr - in_prev[i]);
                           error.combine(diff);
                         });
      });

      queue.wait();

      {
        auto error = buf_error.get_host_access();
        if (error[0] < accuracy)
          break;
        error[0] = 0.0f;
      }

      {
        auto host_curr = buf_curr.get_host_access();
        auto host_prev = buf_prev.get_host_access();
        for (int i = 0; i < size; i++)
          host_prev[i] = host_curr[i];
      }
    }
  }

  return curr_ans;
}