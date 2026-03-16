#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) {
        return 0.0f;
    }

    bool need_finalize = false;
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        need_finalize = true;
    }

    using ExecSpace = Kokkos::SYCL;

    const float step = (end - start) / static_cast<float>(count);
    const float first_midpoint = start + 0.5f * step;

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    Kokkos::parallel_reduce(
            "IntegralKokkosX",
            Kokkos::RangePolicy<ExecSpace>(0, count),
            KOKKOS_LAMBDA(const int i, float& local_sum) {
                local_sum += sinf(first_midpoint + static_cast<float>(i) * step);
            },
            sum_x);

    Kokkos::parallel_reduce(
            "IntegralKokkosY",
            Kokkos::RangePolicy<ExecSpace>(0, count),
            KOKKOS_LAMBDA(const int j, float& local_sum) {
                local_sum += cosf(first_midpoint + static_cast<float>(j) * step);
            },
            sum_y);

    if (need_finalize) {
        Kokkos::finalize();
    }

    return sum_x * sum_y * step * step;
}