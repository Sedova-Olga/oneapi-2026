#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    float step = (end - start) / count;
    float result = 0.0f;
    
    Kokkos::View<float> result_view("result_view");
    Kokkos::deep_copy(result_view, 0.0f);
    
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::SYCL>(0, count * count),
        KOKKOS_LAMBDA(int idx, float& sum) {
            int i = idx / count;
            int j = idx % count;
            
            float x = start + (i + 0.5f) * step;
            float y = start + (j + 0.5f) * step;
            
            sum += std::sin(x) * std::cos(y) * step * step;
        },
        result_view
    );
    
    Kokkos::deep_copy(result, result_view);
    
    return result;
}