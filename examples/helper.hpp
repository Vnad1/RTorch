#pragma once
// RTorch sample helper framework (referenced via `with`).
// Demonstrates the "副框架/other files" contract: a header + implementation
// compiled into the formula's DLL.

#include <cstddef>
#include <cstdint>

namespace rtorch_demo {

// naive square-matrix multiply C = A*B, returns through out (row-major).
void matmul(const float* a, const float* b, float* out, std::size_t n);

// dot product of two vectors of length n.
float dot(const float* a, const float* b, std::size_t n);

} // namespace rtorch_demo
