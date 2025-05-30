// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "avx2.h"
#include "avxprivate.hpp"

namespace vespalib::hwaccelerated {

size_t
Avx2Accelerator::populationCount(const uint64_t *a, size_t sz) const noexcept {
    return helper::populationCount(a, sz);
}

double
Avx2Accelerator::squaredEuclideanDistance(const int8_t * a, const int8_t * b, size_t sz) const noexcept {
    return helper::squaredEuclideanDistance(a, b, sz);
}

double
Avx2Accelerator::squaredEuclideanDistance(const float * a, const float * b, size_t sz) const noexcept {
    return avx::euclideanDistanceSelectAlignment<float, 32>(a, b, sz);
}

double
Avx2Accelerator::squaredEuclideanDistance(const double * a, const double * b, size_t sz) const noexcept {
    return avx::euclideanDistanceSelectAlignment<double, 32>(a, b, sz);
}

void
Avx2Accelerator::and128(size_t offset, const std::vector<std::pair<const void *, bool>> &src, void *dest) const noexcept {
    helper::andChunks<32u, 4u>(offset, src, dest);
}

void
Avx2Accelerator::or128(size_t offset, const std::vector<std::pair<const void *, bool>> &src, void *dest) const noexcept {
    helper::orChunks<32u, 4u>(offset, src, dest);
}

void
Avx2Accelerator::convert_bfloat16_to_float(const uint16_t * src, float * dest, size_t sz) const noexcept {
    helper::convert_bfloat16_to_float(src, dest, sz);
}

int64_t
Avx2Accelerator::dotProduct(const int8_t * a, const int8_t * b, size_t sz) const noexcept
{
    return helper::multiplyAdd(a, b, sz);
}

}
