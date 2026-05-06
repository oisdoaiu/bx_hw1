#ifndef SIMD_WRAPPER_H
#define SIMD_WRAPPER_H

#ifdef __ARM_NEON
#include <arm_neon.h>

struct simd8float32{
    float32x4x2_t data;

    simd8float32() = default;

    explicit simd8float32(const float x){
        float32x4_t v = vdupq_n_f32(x);
        data.val[0] = v;
        data.val[1] = v;
    }

    explicit simd8float32(const float* ptr){
        data.val[0] = vld1q_f32(ptr);
        data.val[1] = vld1q_f32(ptr+4);
    }

    simd8float32 operator*(const simd8float32& other)const{
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    simd8float32 operator+(const simd8float32& other)const{
        simd8float32 result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    simd8float32& operator+=(const simd8float32& other){
        data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return *this;
    }

    simd8float32 operator-(const simd8float32& other)const{
        simd8float32 result;
        result.data.val[0] = vsubq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vsubq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    void storeu(float* ptr)const{
        vst1q_f32(ptr, data.val[0]);
        vst1q_f32(ptr+4, data.val[1]);
    }
};

#elif defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>

struct simd8float32{
    __m256 data;

    simd8float32() = default;

    explicit simd8float32(const float x)
        : data(_mm256_set1_ps(x)){}

    explicit simd8float32(const float* ptr)
        : data(_mm256_loadu_ps(ptr)){}

    simd8float32 operator*(const simd8float32& other)const{
        simd8float32 result;
        result.data = _mm256_mul_ps(data, other.data);
        return result;
    }

    simd8float32 operator+(const simd8float32& other)const{
        simd8float32 result;
        result.data = _mm256_add_ps(data, other.data);
        return result;
    }

    simd8float32& operator+=(const simd8float32& other){
        data = _mm256_add_ps(data, other.data);
        return *this;
    }

    simd8float32 operator-(const simd8float32& other)const{
        simd8float32 result;
        result.data = _mm256_sub_ps(data, other.data);
        return result;
    }

    void storeu(float* ptr)const{
        _mm256_storeu_ps(ptr, data);
    }
};

#else
#include <xmmintrin.h>
#include <emmintrin.h>

struct simd8float32{
    __m128 lo;
    __m128 hi;

    simd8float32() = default;

    explicit simd8float32(const float x)
        : lo(_mm_set1_ps(x)), hi(_mm_set1_ps(x)){}

    explicit simd8float32(const float* ptr)
        : lo(_mm_loadu_ps(ptr)), hi(_mm_loadu_ps(ptr+4)){}

    simd8float32 operator*(const simd8float32& other)const{
        simd8float32 result;
        result.lo = _mm_mul_ps(lo, other.lo);
        result.hi = _mm_mul_ps(hi, other.hi);
        return result;
    }

    simd8float32 operator+(const simd8float32& other)const{
        simd8float32 result;
        result.lo = _mm_add_ps(lo, other.lo);
        result.hi = _mm_add_ps(hi, other.hi);
        return result;
    }

    simd8float32& operator+=(const simd8float32& other){
        lo = _mm_add_ps(lo, other.lo);
        hi = _mm_add_ps(hi, other.hi);
        return *this;
    }

    simd8float32 operator-(const simd8float32& other)const{
        simd8float32 result;
        result.lo = _mm_sub_ps(lo, other.lo);
        result.hi = _mm_sub_ps(hi, other.hi);
        return result;
    }

    void storeu(float* ptr)const{
        _mm_storeu_ps(ptr, lo);
        _mm_storeu_ps(ptr+4, hi);
    }
};

#endif

#endif
