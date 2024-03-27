/**
 * @file sgemm_v1.cu
 * @author your name (you@domain.com)
 * @brief
 * 这个示例是矩阵乘法的最基础的两个版本的实现，只是为了对比通过优化之后的性能
 * @version 0.1
 * @date 2024-03-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>

#include <cstddef>
#include <cstdlib>

#include "assert.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define FETECH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define checkCudaErrors(func)                                                \
  {                                                                          \
    cudaError_t e = (func);                                                  \
    if (e != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }

// 每个线程负责一个C矩阵中的一个元素的计算

__global__ void sgemm_v1(float* __restrict__ A, float* __restrict__ B,
                         float* __restrict__ C,  //
                         const int M,            //
                         const int K,            //
                         const int N) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  float resVal = 0.0;
  for (int i = 0; i < K; i++) {
    resVal += A[OFFSET(row, i, K)] * B[OFFSET(i, col, K)];
  }
  C[OFFSET(row, col, N)] = resVal;
}

// 通过shared_memory对访存过程进行优化
// tiling 优化：将C按照block进行划分块，通过shared
// memory将该block内的数据中global memory中保存下来
template <const int BLOCK_SIZE_M,                // A_tile bm * bk
          const int BLOCK_SIZE_K,                //
          const int BLOCK_SIZE_N>                // B_tile bk * bn
__global__ void sgemm_v2(float* __restrict__ A,  //
                         float* __restrict__ B,  //
                         float* __restrict__ C,
                         const int M,    // Matrix A : M * K
                         const int K,    //
                         const int N) {  // Matrix B : K * N
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  __shared__ float A_s[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float B_s[BLOCK_SIZE_K][BLOCK_SIZE_N];
  float resVal = 0.f;
  // iterate K / bk steps
  for (int i = 0; i < K / BLOCK_SIZE_K; i++) {
    // load data from global memory to shared memory
    A_s[ty][tx] = A[OFFSET(row, BLOCK_SIZE_K * i + tx, K)];
    B_s[ty][tx] = B[OFFSET(BLOCK_SIZE_K * i + ty, col, N)];
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE_K; j++) {
      resVal += A_s[ty][j] * B_s[j][tx];
    }
    __syncthreads();
  }
  C[OFFSET(row, col, N)] = resVal;
}

// 通过shared_memory对访存过程进行优化
// tiling 优化：将C按照block进行划分块，通过shared
// memory将该block内的数据中global memory中保存下来
// 使用float4进行优化，同时将B_s的数据进行转置
template <const int BLOCK_SIZE_M,                // A_tile bm * bk
          const int BLOCK_SIZE_K,                //
          const int BLOCK_SIZE_N>                // B_tile bk * bn
__global__ void sgemm_v3(float* __restrict__ A,  //
                         float* __restrict__ B,  //
                         float* __restrict__ C,
                         const int M,    // Matrix A : M * K
                         const int K,    //
                         const int N) {  // Matrix B : K * N
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  __shared__ float A_s[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float B_s[BLOCK_SIZE_K][BLOCK_SIZE_N];
  float resVal = 0.f;
  // iterate K / bk steps
  for (int i = 0; i < K / BLOCK_SIZE_K; i++) {
    // load data from global memory to shared memory
    A_s[ty][tx] = A[OFFSET(row, BLOCK_SIZE_K * i + tx, K)];
    B_s[tx][ty] = B[OFFSET(BLOCK_SIZE_K * i + ty, col, N)];
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE_K; j++) {
      resVal += A_s[ty][j] * B_s[tx][j];
    }
    __syncthreads();
  }
  C[OFFSET(row, col, N)] = resVal;
}

/**
 * @brief 2d tiling
 * 通过让每条线程负责更多的数据的计算，来让计算掩盖访存时候的延迟
 */
template <const int BLOCK_SIZE_M,   //
          const int BLOCK_SIZE_K,   //
          const int BLOCK_SIZE_N,   //
          const int THREAD_SIZE_M,  // 对 BM * BK 中的数据进行划分
          const int THREAD_SIZE_N>  // 对 BK * BN 中的数据进行划分
// 此时每个线程负责 TILE_SIZE_M个元素的计算，那么block中的线程就可以减少
__global__ void sgemm_v4(float* __restrict__ A,  //
                         float* __restrict__ B,  //
                         float* __restrict__ C,
                         const int M,    // Matrix A : M * K
                         const int K,    //
                         const int N) {  // Matrix B : K * N
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * BLOCK_SIZE_M;
  int col = BLOCK_SIZE_N * bx;

  float res[THREAD_SIZE_M][THREAD_SIZE_N] = {0};
  __shared__ float A_s[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float B_s[BLOCK_SIZE_K][BLOCK_SIZE_N];
  int num_blocks = CEIL_DIV(K, BLOCK_SIZE_K);
  int THREAD_NUMS = THREAD_SIZE_M * THREAD_SIZE_N;

  int tid = ty * blockDim.x + tx;

  // 除4是因为计算出每个线程至少一次性load4个元素
  const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
  const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

  int a_row_start = tid / A_TILE_THREAD_PER_ROW;
  int b_row_start = tid / B_TILE_THREAD_PER_ROW;
  int a_col = (tid % A_TILE_THREAD_PER_ROW) * 4;
  int b_col = (tid % B_TILE_THREAD_PER_ROW) * 4;

  //   int a_stride = 0;
  //   int b_stride = 0;

  A = &A[OFFSET(by * BLOCK_SIZE_M, 0, K)];
  B = &B[OFFSET(0, bx * BLOCK_SIZE_N, N)];
  // 大循环
  for (int i = 0; i < num_blocks; i++) {
    // load data from global memory to A_s and B_s
    // 每个线程负责将一部分数据从global memory中加载到shared memory中

    FETECH_FLOAT4(A_s[a_row_start][a_col]) =
        FETECH_FLOAT4(A[OFFSET(a_row_start, i * BLOCK_SIZE_K + a_col, K)]);
    FETECH_FLOAT4(B_s[b_row_start][b_col]) =
        FETECH_FLOAT4(B[OFFSET(i * BLOCK_SIZE_K + b_row_start, b_col, N)]);

    __syncthreads();

    // calculate rm * rn size res
    // 小循环
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
      for (int m = 0; m < THREAD_SIZE_M; m++) {
        for (int n = 0; n < THREAD_SIZE_N; n++) {
          res[m][n] +=
              A_s[ty * THREAD_SIZE_M + m][k] * B_s[k][tx * THREAD_SIZE_N + n];
        }
      }
    }
    __syncthreads();
  }
  // store to C
  for (int m = 0; m < THREAD_SIZE_M; m++) {
    for (int n = 0; n < THREAD_SIZE_N; n++) {
      if (row + ty * THREAD_SIZE_M + m < M &&
          col + tx * THREAD_SIZE_N + n < N) {
        C[OFFSET(row + ty * THREAD_SIZE_M + m, col + tx * THREAD_SIZE_N + n,
                 N)] = res[m][n];
      }
    }
  }
}
int main(int argc, char** argv) {
  if (argc != 4) {
    printf("usage: ./main [M] [K] [N]\n");
    exit(0);
  }
  int M = atoi(argv[1]);
  int K = atoi(argv[2]);
  int N = atoi(argv[3]);

  const int BLOCK_SIZE_K = 8;
  const int BLOCK_SIZE_M = 128;
  const int BLOCK_SIZE_N = 128;
  const int THREAD_SIZE_M = 8;
  const int THREAD_SIZE_N = 8;

  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);

  size_t bytes_A = sizeof(float) * M * K;
  size_t bytes_B = sizeof(float) * K * N;
  size_t bytes_C = sizeof(float) * M * N;
  float* h_A = (float*)malloc(bytes_A);
  float* h_B = (float*)malloc(bytes_B);
  float* h_C = (float*)malloc(bytes_C);
  float* h_C1 = (float*)malloc(bytes_C);

  float* d_A;
  float* d_B;
  float* d_C;

  checkCudaErrors(cudaMalloc(&d_A, bytes_A));
  checkCudaErrors(cudaMalloc(&d_B, bytes_B));
  checkCudaErrors(cudaMalloc(&d_C, bytes_C));
  double msecPerMatrixMul[2] = {0, 0};
  double gigaFlops[2] = {0, 0};
  double flopsPerMatrixMul = 2.0 * M * N * K;
  // generate A
  for (int i = 0; i < M * K; i++) {
    h_A[i] = i / 13;
  }

  // generate B
  for (int i = 0; i < K * N; i++) {
    h_B[i] = i % 13;
  }

  checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float msecTotal = 0;
  int nIter = 1;

  checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0; run < nIter; run++) {
    dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);
    dim3 dimGrid(CEIL_DIV(M, BLOCK_SIZE_M), CEIL_DIV(N, BLOCK_SIZE_N));
    // sgemm_v1<<<dimBlock, dimGrid>>>(d_A, d_B, d_C, M, K, N);
    // sgemm_v2<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N>
    //     <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    sgemm_v4<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M,
             THREAD_SIZE_N><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
  }

  //   printf("flag = %d\n", flag);
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

  msecPerMatrixMul[0] = msecTotal / nIter;
  gigaFlops[0] =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
  printf(
      "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
      gigaFlops[0], msecPerMatrixMul[0], flopsPerMatrixMul);
  cublasHandle_t blash_handle;
  cublasCreate(&blash_handle);
  float alpha = 1.0;
  float beta = 0.0;
  checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0; run < nIter; run++) {
    cublasSgemm(blash_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K,
                d_B, N, &beta, d_C, N);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));
  msecPerMatrixMul[1] = msecTotal / nIter;
  gigaFlops[1] =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
  printf(
      "cuBLAS gemm Performance= %.2f GFlop/s, Time = %.3f msec, Size = %.0f "
      "Ops\n",
      gigaFlops[1], msecPerMatrixMul[1], flopsPerMatrixMul);

  cublasDestroy(blash_handle);

  double eps = 1.e-6;  // machine zero
  bool correct = true;
  for (int i = 0; i < M * N; i++) {
    int row = i / N;
    int col = i % N;
    double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
    double dot_length = M;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i],
             h_C1[col * M + row], eps);
      correct = false;
      break;
    }
  }

  printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
  printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C1);
}