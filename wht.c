#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#define NOINLINE __attribute__((noinline))

typedef struct timespec Timespec;
static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}
static uint64_t elapsed_ns(Timespec start, Timespec stop) {
  return (uint64_t)(stop.tv_sec - start.tv_sec) * 1000000000LL + (uint64_t)(stop.tv_nsec - start.tv_nsec);
}

void NOINLINE wht_32_ref(float* x) {
    for (int h = 1; h < 32; h <<= 1) {
        for (int i = 0; i < 32; i += h << 1) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

void NOINLINE wht_32_simd(float* x) {
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

    __m256 u[4], v[4], w[4];
    for (int i = 0; i < 4; i++) {
        u[i] = _mm256_loadu_ps(&x[i*8]);
    }

    // round 0,1,2
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 4; i++) {
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

    for (int i = 0; i < 4; i++) {
        _mm256_storeu_ps(&x[i*8], w[i]);
    }
}

void print(float x[32]) {
    for (int i = 0; i < 32; i++) {
        printf("%g ", x[i]);
    }
    printf("\n");
}

// just messed around until got a non-trivial output
void NOINLINE test_case_1(float x[32]) {
    for (int i = 0; i < 32; i++) {
        x[i] = expf(1. / 32. * i);
    }
}

int main() {
    float x[32];

#ifdef TEST
    float y[32];

    test_case_1(x);
    print(x);
    wht_32_ref(x);
    print(x);

    test_case_1(y);
    wht_32_simd(y);
    print(y);

    for (int i = 0; i < 32; i++) {
        if (x[i] != y[i]) {
            printf("ERR x[%d] = %f y[%d] = %f\n", i, x[i], i, y[i]);
        }
        assert(x[i] == y[i]);
    }
#else

    size_t rounds = 10000000;
    float acc;
    Timespec start, stop;

    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        test_case_1(x);
        acc += x[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "overhead", acc, elapsed_ns(start, stop), (double)elapsed_ns(start, stop) / rounds);
    uint64_t remove = elapsed_ns(start, stop);

    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        test_case_1(x);
        wht_32_ref(x);
        acc += x[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_32_ref", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);

    acc = 0;
    clock_ns(&start);
    for (size_t i = 0; i < rounds; i++) {
        test_case_1(x);
        wht_32_simd(x);
        acc += x[3];
    }
    clock_ns(&stop);
    printf("%20s acc=%f elapsed=%ldns per_iter=%.2fns\n", "wht_32_simd", acc, elapsed_ns(start, stop), (double)(elapsed_ns(start, stop) - remove) / rounds);
#endif
}

/*
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
