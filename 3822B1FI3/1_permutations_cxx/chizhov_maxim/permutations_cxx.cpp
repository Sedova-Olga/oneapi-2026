#include "permutations_cxx.h"
#include <algorithm>
#include <string>

void Permutations(dictionary_t& dictionary) {
    for (auto& outer : dictionary) {
        outer.second.clear();

        std::string sorted_outer = outer.first;
        std::sort(sorted_outer.begin(), sorted_outer.end());

        for (auto& inner : dictionary) {
            if (outer.first == inner.first) {
                continue;
            }

            std::string sorted_inner = inner.first;
            std::sort(sorted_inner.begin(), sorted_inner.end());

            if (sorted_outer == sorted_inner) {
                outer.second.push_back(inner.first);
            }
        }

        std::sort(outer.second.begin(), outer.second.end(),
            std::greater<std::string>());
    }
}