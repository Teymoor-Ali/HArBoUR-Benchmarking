#include <iostream>
#include <cstdlib>
#include <chrono>
#include <mkl.h>

#ifndef USE_DOUBLE
#define FLOAT float
#define GEMM cblas_sgemm
#else
#define FLOAT double
#define GEMM cblas_dgemm
#endif

#define LOOP 1

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Please specify problem size." << std::endl;
        return 1;
    }

    int m = std::atoi(argv[1]);

    double gflops = 2.0 * static_cast<double>(m) * static_cast<double>(m) * static_cast<double>(m);

    FLOAT *A = (FLOAT *)mkl_malloc(m * m * sizeof(FLOAT), 64);
    FLOAT *B = (FLOAT *)mkl_malloc(m * m * sizeof(FLOAT), 64);
    FLOAT *C = (FLOAT *)mkl_malloc(m * m * sizeof(FLOAT), 64);

    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Memory allocation failed." << std::endl;
        return 1;
    }

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            A[i + j * m] = static_cast<FLOAT>(rand()) / (static_cast<FLOAT>(RAND_MAX) - 0.5);
            B[i + j * m] = static_cast<FLOAT>(rand()) / (static_cast<FLOAT>(RAND_MAX) - 0.5);
            C[i + j * m] = 0.0;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int l = 0; l < LOOP; ++l) {
        GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, A, m, B, m, 0.0, C, m);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    double time = elapsed.count();

#ifndef USE_DOUBLE
    std::cout << "SGEMM Performance N = " << m << " : " << gflops / time * static_cast<double>(LOOP) * 1.e-9 << " GF" << std::endl;
    std::cout << "Time = " << time << " seconds" << std::endl;
#else
    std::cout << "DGEMM Performance N = " << m << " : " << gflops / time * static_cast<double>(LOOP) * 1.e-9 << " GF" << std::endl;
    std::cout << "Time = " << time << " seconds" << std::endl;
#endif

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
