#include "helper.hpp"

namespace rtorch_demo {

void matmul(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            float aik = a[i * n + k];
            float* row = out + i * n;
            const float* bro = b + k * n;
            for (std::size_t j = 0; j < n; ++j) {
                row[j] += aik * bro[j];
            }
        }
    }
}

float dot(const float* a, const float* b, std::size_t n) {
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

} // namespace rtorch_demo
