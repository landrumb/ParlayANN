#pragma once
#include <immintrin.h>

/* These pointers are NOT assumed to be 64 bit aligned 

This function assumes that AVX512F is available */
template <size_t dim>
float sq_euclidean(const float* a, const float* b) {
    __m512 total = _mm512_setzero_ps();

    constexpr size_t n_512 = dim / 16;

    for (size_t i=0; i < n_512; i++){
        __m512 a_chunk = _mm512_loadu_ps(a + 16*i);
        __m512 b_chunk = _mm512_loadu_ps(b + 16*i);

        a_chunk = _mm512_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm512_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, a_chunk);
    }

    constexpr bool remainder_256 = (dim % 16) >= 8;
    
    if constexpr (remainder_256) {
        __m256 a_chunk = _mm256_loadu_ps(a + 16 * n_512);
        __m256 b_chunk = _mm256_loadu_ps(b + 16 * n_512);

        a_chunk = _mm256_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm256_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, _mm512_castps256_ps512(a_chunk));
    }

    constexpr bool remainder_128 = (dim % 16) % 8 >= 4;

    if constexpr (remainder_128) {
        __m128 a_chunk = _mm_load_ps(a + 16 * n_512 + (8 * remainder_256));
        __m128 b_chunk = _mm_load_ps(b + 16 * n_512 + (8 * remainder_256));

        a_chunk = _mm_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, _mm512_castps128_ps512(a_chunk));
    }

    // I mean realistically dimensionalities will be divisible by 4 so this is just for completeness
    constexpr size_t remainder = dim % 4;

    if constexpr (remainder) {
        // doing a masked load to get just what we care about
        // all values we want (first `remainder`) have 1 in the corresponding bit
        // incredibly, this is the inverse of a non-zeroing mask load
        __mmask8 mask = (1 << remainder) - 1;

        __m128 a_chunk = _mm_maskz_loadu_ps(mask, a + 16 * n_512 + (8 * remainder_256) + (4 * remainder_128));
        __m128 b_chunk = _mm_maskz_loadu_ps(mask, b + 16 * n_512 + (8 * remainder_256) + (4 * remainder_128));

        a_chunk = _mm_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, _mm512_castps128_ps512(a_chunk));
    }

    return _mm512_reduce_add_ps(total);
}

