#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#define NOINLINE __attribute__((noinline))

typedef struct timespec Timespec;
static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}
static uint64_t elapsed_ns(Timespec start, Timespec stop) {
  return (uint64_t)(stop.tv_sec - start.tv_sec) * 1000000000LL + (uint64_t)(stop.tv_nsec - start.tv_nsec);
}

void NOINLINE wht_n_ref(float* x, size_t n) {
    assert(__builtin_popcountll(n) == 1);
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += h << 1) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

void NOINLINE wht_16_simd(float* x) {
    const size_t k = 2;
    const __m256 sign[3] = {
        _mm256_set_ps(-1., +1., -1., +1., -1., +1., -1, +1.),
        _mm256_set_ps(-1., -1., +1., +1., -1., -1., +1, +1.),
        _mm256_set_ps(-1., -1., -1., -1., +1., +1., +1, +1.),
    };
    const __m256i shuf[3] = {
        _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1), // shufps with 0xb1 == 0b10_11_00_01
        _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2), // shufps 0x4e == 0b01_00_11_10 (or shufpd with 0x5 == 0b01_01)
        _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4), // permpd with 0x4e == 0b01_01
    };

    __m256 u[k], v[k];
    for (int i = 0; i < k; i++) {
        u[i] = _mm256_loadu_ps(&x[i*8]);
    }

    // round 0,1,2
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < k; i++) {
            // clang optimizes the shuffle into vshufps, vshufpd, vpermpd with immediates
            u[i] = _mm256_fmadd_ps(u[i], sign[r], _mm256_permutevar8x32_ps(u[i], shuf[r]));
        }
    }

    // round 3
    v[0] = _mm256_add_ps(u[0], u[1]);
    v[1] = _mm256_sub_ps(u[0], u[1]);

    for (int i = 0; i < k; i++) {
        _mm256_storeu_ps(&x[i*8], v[i]);
    }
}


void NOINLINE wht_32_simd(float* x) {
    const size_t k = 4; // number of vectors
    const __m256 sign[3] = {
        _mm256_set_ps(-1., +1., -1., +1., -1., +1., -1, +1.),
        _mm256_set_ps(-1., -1., +1., +1., -1., -1., +1, +1.),
        _mm256_set_ps(-1., -1., -1., -1., +1., +1., +1, +1.),
    };
    const __m256i shuf[3] = {
        _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1), // shufps with 0xb1 == 0b10_11_00_01
        _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2), // shufps 0x4e == 0b01_00_11_10 (or shufpd with 0x5 == 0b01_01)
        _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4), // permpd with 0x4e == 0b01_01
    };

    __m256 u[k], v[k], w[k];
    for (int i = 0; i < k; i++) {
        u[i] = _mm256_loadu_ps(&x[i*8]);
    }

    // round 0,1,2
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < k; i++) {
            // clang optimizes the shuffle into vshufps, vshufpd, vpermpd with immediates
            u[i] = _mm256_fmadd_ps(u[i], sign[r], _mm256_permutevar8x32_ps(u[i], shuf[r]));
        }
    }

    // round 3
    v[0] = _mm256_add_ps(u[0], u[1]);
    v[1] = _mm256_sub_ps(u[0], u[1]);
    v[2] = _mm256_add_ps(u[2], u[3]);
    v[3] = _mm256_sub_ps(u[2], u[3]);

    // round 4
    w[0] = _mm256_add_ps(v[0], v[2]);
    w[1] = _mm256_add_ps(v[1], v[3]);
    w[2] = _mm256_sub_ps(v[0], v[2]);
    w[3] = _mm256_sub_ps(v[1], v[3]);

    for (int i = 0; i < k; i++) {
        _mm256_storeu_ps(&x[i*8], w[i]);
    }
}

void wht_64_simd_(float* x) {
    const size_t k = 8; // number of vectors
    const __m256 sign[3] = {
        _mm256_set_ps(-1., +1., -1., +1., -1., +1., -1, +1.),
        _mm256_set_ps(-1., -1., +1., +1., -1., -1., +1, +1.),
        _mm256_set_ps(-1., -1., -1., -1., +1., +1., +1, +1.),
    };
    const __m256i shuf[3] = {
        _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1), // shufps with 0xb1 == 0b10_11_00_01
        _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2), // shufps 0x4e == 0b01_00_11_10 (or shufpd with 0x5 == 0b01_01)
        _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4), // permpd with 0x4e == 0b01_01
    };

    __m256 u[k];
    for (int i = 0; i < k; i++) {
        u[i] = _mm256_loadu_ps(&x[i*8]);
    }

    // round 0,1,2
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < k; i++) {
            // clang optimizes the shuffle into vshufps, vshufpd, vpermpd with immediates
            u[i] = _mm256_fmadd_ps(u[i], sign[r], _mm256_permutevar8x32_ps(u[i], shuf[r]));
        }
    }

    const size_t rounds = 3 + 1;
    __m256 w[rounds][k];
    for (size_t i = 0; i < k; i++) {
        w[0][i] = u[i];
    }
    size_t r = 1;
    for (size_t h = 1; h < k; h *= 2, r += 1) {
        for (size_t i = 0; i < k; i += h * 2) {
            for (size_t j = i; j < i + h; j++) {
                w[r][j]     = _mm256_add_ps(w[r-1][j], w[r-1][j + h]);
                w[r][j + h] = _mm256_sub_ps(w[r-1][j], w[r-1][j + h]);
            }
        }
    }
    for (int i = 0; i < k; i++) {
        _mm256_storeu_ps(&x[i*8], w[rounds-1][i]);
    }
}

void NOINLINE wht_64_simd(float* x) {
    wht_64_simd_(x);
}

// this spills to stack, not sure there is any way around that
void NOINLINE wht_128_simd(float* x) {
    const size_t k = 16; // number of vectors
    const __m256 sign[3] = {
        _mm256_set_ps(-1., +1., -1., +1., -1., +1., -1, +1.),
        _mm256_set_ps(-1., -1., +1., +1., -1., -1., +1, +1.),
        _mm256_set_ps(-1., -1., -1., -1., +1., +1., +1, +1.),
    };
    const __m256i shuf[3] = {
        _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1), // shufps with 0xb1 == 0b10_11_00_01
        _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2), // shufps 0x4e == 0b01_00_11_10 (or shufpd with 0x5 == 0b01_01)
        _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4), // permpd with 0x4e == 0b01_01
    };

    __m256 u[k], v[k];
    for (int i = 0; i < k; i++) {
        u[i] = _mm256_loadu_ps(&x[i*8]);
    }

    // round 0,1,2
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < k; i++) {
            // clang optimizes the shuffle into vshufps, vshufpd, vpermpd with immediates
            u[i] = _mm256_fmadd_ps(u[i], sign[r], _mm256_permutevar8x32_ps(u[i], shuf[r]));
        }
    }

    const size_t rounds = 4 + 1;
    __m256 w[rounds][k];
    for (size_t i = 0; i < k; i++) {
        w[0][i] = u[i];
    }
    size_t r = 1;
    for (size_t h = 1; h < k; h *= 2, r += 1) {
        for (size_t i = 0; i < k; i += h * 2) {
            for (size_t j = i; j < i + h; j++) {
                w[r][j]     = _mm256_add_ps(w[r-1][j], w[r-1][j + h]);
                w[r][j + h] = _mm256_sub_ps(w[r-1][j], w[r-1][j + h]);
            }
        }
    }
    for (int i = 0; i < k; i++) {
        _mm256_storeu_ps(&x[i*8], w[rounds-1][i]);
    }
}

// this is another ordering where we do all on one half, then the other, then the last round
// compiler leaves the second half in registers so maybe less stack spilling
// 47 load/stores vs 67 for the above
// is maybe 10% faster but don't really trust the numbers
void NOINLINE wht_128_simd_alt(float* x) {
    const size_t k = 16; // number of vectors

    wht_64_simd_(x);
    wht_64_simd_(x + 64);

    const size_t h = 8;
    // this is round 7
    __m256 a, aa, b, bb;
    for (size_t i = 0; i < k; i += h * 2) {
        for (size_t j = i; j < i + h; j++) {
            a = _mm256_loadu_ps(&x[j * 8]);
            b = _mm256_loadu_ps(&x[(j + h) * 8]);
            aa = _mm256_add_ps(a, b);
            bb = _mm256_sub_ps(a, b);
            _mm256_storeu_ps(&x[j * 8], aa);
            _mm256_storeu_ps(&x[(j + h) * 8], bb);
        }
    }
}

void print(float* x, size_t n) {
    /*for (int i = 0; i < n; i++) {*/
    /*    printf("%g ", x[i]);*/
    /*}*/
    /*printf("\n");*/
}

// just messed around until got a non-trivial output
void NOINLINE test_case_1(float* x, size_t n) {
    for (int i = 0; i < n; i++) {
        x[i] = expf(1. / (float)n * i);
    }
}

void check_eq(float* x, float* y, size_t n) {
    for (int i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            printf("ERR n=%ld : x[%d] = %f y[%d] = %f\n", n, i, x[i], i, y[i]);
        }
        assert(x[i] == y[i]);
    }
}

int main() {
    // use sized arrays to let asan detect bounds
    float x16[16], x32[32], x64[64], x128[128];
    float y16[16], y32[32], y64[64], y128[128];

#ifdef TEST

#define CASE(f, N) \
    printf("n=%d\n", N); \
    test_case_1(x##N, N); \
    print(x##N, N); \
    wht_n_ref(x##N, N); \
    print(x##N, N); \
    test_case_1(y##N, N); \
    f(y##N); \
    print(y##N, N); \
    check_eq(x##N, y##N, N);

    CASE(wht_16_simd, 16)
    CASE(wht_32_simd, 32)
    CASE(wht_64_simd, 64)
    CASE(wht_128_simd, 128)
    CASE(wht_128_simd_alt, 128)

#else

    size_t rounds = 30000000;
    float acc;
    Timespec start, stop;

    test_case_1(x32, 32);
    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        memcpy(y32, x32, sizeof(float) * 32);
        acc += x32[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "overhead_32", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / rounds);
    uint64_t remove = elapsed_ns(start, stop);

    test_case_1(x32, 32);
    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        memcpy(y32, x32, sizeof(float) * 32);
        wht_n_ref(y32, 32);
        acc += y32[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_32_ref", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);

    test_case_1(x16, 16);
    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        memcpy(y16, x16, sizeof(float) * 16);
        wht_16_simd(y16);
        acc += y16[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_16_simd", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);

    test_case_1(x32, 32);
    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        memcpy(y32, x32, sizeof(float) * 32);
        wht_32_simd(y32);
        acc += y32[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_32_simd", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);

    test_case_1(x64, 64);
    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        memcpy(y64, x64, sizeof(float) * 64);
        wht_64_simd(y64);
        acc += y64[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_64_simd", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);

    test_case_1(x128, 128);
    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        memcpy(y128, x128, sizeof(float) * 128);
        wht_128_simd(y128);
        acc += y128[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_128_simd", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);

    test_case_1(x128, 128);
    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        memcpy(y128, x128, sizeof(float) * 128);
        wht_128_simd_alt(y128);
        acc += y128[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_128_simd_alt", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);

#endif
}

/*
wht_32_simd
Iterations:        100
Instructions:      4300
Total Cycles:      1110
Total uOps:        4700

Dispatch Width:    6
uOps Per Cycle:    4.23
IPC:               3.87
Block RThroughput: 7.8


Instruction Info:
[1]: #uOps
[2]: Latency
[3]: RThroughput
[4]: MayLoad
[5]: MayStore
[6]: HasSideEffects (U)

[1]    [2]    [3]    [4]    [5]    [6]    Instructions:
 1      8     0.50    *                   vmovups	ymm0, ymmword ptr [rdi]
 1      8     0.50    *                   vmovups	ymm1, ymmword ptr [rdi + 32]
 1      8     0.50    *                   vbroadcastsd	ymm5, qword ptr [rip + 2958]
 1      8     0.50    *                   vmovups	ymm2, ymmword ptr [rdi + 64]
 1      8     0.50    *                   vmovups	ymm3, ymmword ptr [rdi + 96]
 1      1     0.50                        vshufps	ymm4, ymm0, ymm0, 177
 1      4     0.50                        vfmadd231ps	ymm4, ymm5, ymm0
 1      1     0.50                        vshufps	ymm0, ymm1, ymm1, 177
 1      4     0.50                        vfmadd231ps	ymm0, ymm5, ymm1
 1      1     0.50                        vshufps	ymm1, ymm2, ymm2, 177
 1      4     0.50                        vfmadd231ps	ymm1, ymm5, ymm2
 1      1     0.50                        vshufps	ymm2, ymm3, ymm3, 177
 1      4     0.50                        vfmadd231ps	ymm2, ymm3, ymm5
 1      8     0.50    *                   vbroadcastf128	ymm5, xmmword ptr [rip + 2931]
 1      1     0.50                        vshufpd	ymm3, ymm4, ymm4, 5
 1      4     0.50                        vfmadd231ps	ymm3, ymm5, ymm4
 1      1     0.50                        vshufpd	ymm4, ymm0, ymm0, 5
 1      4     0.50                        vfmadd231ps	ymm4, ymm5, ymm0
 1      1     0.50                        vshufpd	ymm0, ymm1, ymm1, 5
 1      4     0.50                        vfmadd231ps	ymm0, ymm5, ymm1
 1      1     0.50                        vshufpd	ymm1, ymm2, ymm2, 5
 1      4     0.50                        vfmadd231ps	ymm1, ymm5, ymm2
 1      8     0.50    *                   vmovaps	ymm2, ymmword ptr [rip + 2931]
 2      6     1.00                        vpermpd	ymm6, ymm3, 78
 1      4     0.50                        vfmadd231ps	ymm6, ymm2, ymm3
 2      6     1.00                        vpermpd	ymm3, ymm4, 78
 1      4     0.50                        vfmadd231ps	ymm3, ymm2, ymm4
 2      6     1.00                        vpermpd	ymm4, ymm0, 78
 1      4     0.50                        vfmadd231ps	ymm4, ymm2, ymm0
 2      6     1.00                        vpermpd	ymm0, ymm1, 78
 1      4     0.50                        vfmadd231ps	ymm0, ymm2, ymm1
 1      3     0.50                        vaddps	ymm1, ymm6, ymm3
 1      3     0.50                        vsubps	ymm2, ymm6, ymm3
 1      3     0.50                        vaddps	ymm3, ymm4, ymm0
 1      3     0.50                        vsubps	ymm0, ymm4, ymm0
 1      3     0.50                        vaddps	ymm4, ymm1, ymm3
 1      3     0.50                        vaddps	ymm5, ymm2, ymm0
 1      3     0.50                        vsubps	ymm1, ymm1, ymm3
 1      3     0.50                        vsubps	ymm0, ymm2, ymm0
 1      1     1.00           *            vmovups	ymmword ptr [rdi], ymm4
 1      1     1.00           *            vmovups	ymmword ptr [rdi + 32], ymm5
 1      1     1.00           *            vmovups	ymmword ptr [rdi + 64], ymm1
 1      1     1.00           *            vmovups	ymmword ptr [rdi + 96], ymm0
 */
