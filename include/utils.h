#pragma once
#include <cmath>
#include <tuple>
inline int padToPowerOf2(int x) {
    if (x <= 0) return 0;
    return 1 << static_cast<int>(std::ceil(std::log2(x)));
}

inline std::tuple<int, int, int> padToNextPowerOf2(int M, int N, int R) {
    return {
        padToPowerOf2(M),
        padToPowerOf2(N),
        padToPowerOf2(R)
    };
}

inline std::tuple<int, int, int> padToNextPowerOf2(const std::tuple<int, int, int>& dims) {
    return padToNextPowerOf2(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}