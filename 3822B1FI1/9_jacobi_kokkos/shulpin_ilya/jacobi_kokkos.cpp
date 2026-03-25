#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy)
{
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const size_t n = b.size();
    if (n == 0 || a.size() != n * n) {
        return {};
    }

    using ExecutionSpace = Kokkos::SYCL;
    using MemorySpace = Kokkos::SYCLDeviceUSMSpace;

    using View1D = Kokkos::View<float*, MemorySpace>;
    using View2D = Kokkos::View<float**, Kokkos::LayoutRight, MemorySpace>;

    View2D A("A", n, n);
    View1D bb("b", n);
    View1D x_curr("x_curr", n);
    View1D x_next("x_next", n);

    auto hA = Kokkos::create_mirror_view(A);
    auto hb = Kokkos::create_mirror_view(bb);

    for (size_t i = 0; i < n; ++i) {
        hb(i) = b[i];
        for (size_t j = 0; j < n; ++j) {
            hA(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(A, hA);
    Kokkos::deep_copy(bb, hb);

    Kokkos::deep_copy(x_curr, 0.0f);

    float max_diff = 0.0f;

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        max_diff = 0.0f;

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<ExecutionSpace>(0, n),
            KOKKOS_LAMBDA(const size_t i, float& local_max) {
                float sigma = 0.0f;

                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sigma += A(i, j) * x_curr(j);
                    }
                }

                float diag = A(i, i);
                float new_val = (Kokkos::abs(diag) < 1e-12f)
                              ? x_curr(i)
                              : (bb(i) - sigma) / diag;

                x_next(i) = new_val;

                float diff = Kokkos::abs(new_val - x_curr(i));
                if (diff > local_max) {
                    local_max = diff;
                }
            },
            max_diff
        );

        if (max_diff < accuracy) {
            break;
        }

        std::swap(x_curr, x_next);
    }

    std::vector<float> solution(n);
    auto hsolution = Kokkos::create_mirror_view(x_curr);
    Kokkos::deep_copy(hsolution, x_curr);

    for (size_t i = 0; i < n; ++i) {
        solution[i] = hsolution(i);
    }

    return solution;
}