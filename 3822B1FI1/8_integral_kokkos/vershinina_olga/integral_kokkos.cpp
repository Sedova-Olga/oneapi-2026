#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {

    float result = 0.0f;
    float h = (end - start) / count;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { count, count }),
        KOKKOS_LAMBDA(const int i, const int j, float& lsum) {
        float x = start + (static_cast<float>(i) + 0.5f) * h;
        float y = start + (static_cast<float>(j) + 0.5f) * h;
        lsum += std::sin(x) * std::cos(y);
    },
        result
        );

    return result * h * h;
}