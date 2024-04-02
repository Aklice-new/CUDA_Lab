/**
 * @file gemv_n16.cu n = 16
 * @author aklice
 * @brief gemv矩阵向量乘法的实现，针对不同size的n做出不同的优化
 * @version 0.1
 * @date 2024-03-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <linux/limits.h>
#include <stdio.h>
#include <stdlib.h>

#define FETCH_FLOAT4(poniter) (reinterpret_cast<float4*>(&(poniter))[0])
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define checkCudaErrors(func)                                                \
  {                                                                          \
    cudaError_t e = (func);                                                  \
    if (e != cudaSuccess) {                                                  \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                        \
  }

template <const int WARP_SIZE>
__device__ float warpReduceSum(float sum, unsigned int mask) {
  // 0~15号线程 对应拿到  16~31号线程的sum
  if (WARP_SIZE >= 32) sum += __shfl_down_sync(mask, sum, 16);
  // 0~7号线程 对应拿到  8~15 号线程的sum
  if (WARP_SIZE >= 16) sum += __shfl_down_sync(mask, sum, 8);
  // 0~3号线程 对应拿到  4~7 号线程的sum
  if (WARP_SIZE >= 8) sum += __shfl_down_sync(mask, sum, 4);
  // 0~1号线程 对应拿到  2~3 号线程的sum
  if (WARP_SIZE >= 4) sum += __shfl_down_sync(mask, sum, 2);
  // 0号线程 对应拿到  1 号线程的sum
  if (WARP_SIZE >= 2) sum += __shfl_down_sync(mask, sum, 1);
  return sum;
}
// A : M * N, N 是128的倍数，一个warp负责计算一行的结果
// V : N * 1
// 一行N个元素，则每一个warp一次计算128个元素的结果，这里采用float4进行加速访问
// 32*4 = 128
template <const int MATRIX_M,  //矩阵的高
          const int MATRIX_N>  //矩阵的宽
__global__ void gemv_kernel(float* __restrict__ A, float* __restrict__ V,
                            float* __restrict__ out) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  const int WARP_SIZE = 32;
  const int ELEMENT_PER_WARP = 128;
  const int ITER_PER_ROW = MATRIX_N / ELEMENT_PER_WARP;
  int row = bx * blockDim.y + ty;  // 计算当前所在行
  int col = tx;
  int tid = ty * blockDim.x + tx;  // 计算当前线程的块内id
  int landId = tid % WARP_SIZE;
  float res = 0.f;
  // 将一行分成几个warp，然后32个线程，先计算对应负责的元素
  for (int i = 0; i < ITER_PER_ROW; i++) {
    int col = i * WARP_SIZE + landId;
    float4 a = FETCH_FLOAT4(A[OFFSET(row, col * 4, MATRIX_N)]);
    float4 v = FETCH_FLOAT4(V[col * 4]);
    res += a.x * v.x;
    res += a.y * v.y;
    res += a.z * v.z;
    res += a.w * v.w;
  }
  // 然后通过warp reduce 进行求和
  unsigned int mask = 0xffffffff;
  res = warpReduceSum<WARP_SIZE>(res, mask);
  if (landId == 0) {
    out[row] = res;
  }
}
int main() {
  // A : M * N
  // V : N * 1
  // out : M * 1
  const int MATRIX_M = 256 * 1024;
  const int MATRIX_N = 512;
  const int bytes_A = MATRIX_M * MATRIX_N * sizeof(float);
  const int bytes_V = MATRIX_N * sizeof(float);
  const int bytes_out = MATRIX_M * sizeof(float);

  float *h_A, *h_V, *h_out, *h_out1;
  h_A = (float*)malloc(bytes_A);
  h_V = (float*)malloc(bytes_V);
  h_out = (float*)malloc(bytes_out);
  h_out1 = (float*)malloc(bytes_out);

  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    h_A[i] = i * 1.0 / MATRIX_M;
  }

  for (int i = 0; i < MATRIX_N; i++) {
    h_V[i] = i * 1.0 / MATRIX_N;
  }

  float *d_A, *d_V, *d_out;
  checkCudaErrors(cudaMalloc((void**)&d_A, bytes_A));
  checkCudaErrors(cudaMalloc((void**)&d_V, bytes_V));
  checkCudaErrors(cudaMalloc((void**)&d_out, bytes_out));

  checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_V, h_V, bytes_V, cudaMemcpyHostToDevice));

  int nIter = 1000;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, 0));
  // excute kernel
  for (int i = 0; i < nIter; i++) {
    dim3 threads_per_block = dim3(32, 4);
    dim3 block_per_grid = MATRIX_M / 4;
    gemv_kernel<MATRIX_M, MATRIX_N>
        <<<block_per_grid, threads_per_block>>>(d_A, d_V, d_out);
  }
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msec;
  checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));

  checkCudaErrors(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

  // cubals gemv
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.f;
  checkCudaErrors(cudaMemcpy(d_out, h_out1, bytes_out, cudaMemcpyHostToDevice));
  cublasSgemv(handle, CUBLAS_OP_T, MATRIX_N, MATRIX_M, &alpha, d_A, MATRIX_N,
              d_V, 1, &beta, d_out, 1);
  checkCudaErrors(cudaMemcpy(h_out1, d_out, bytes_out, cudaMemcpyDeviceToHost));
  cublasDestroy(handle);
  // check error
  double eps = 1.e-6;  // machine zero
  bool correct = true;
  for (int i = 0; i < MATRIX_M; i++) {
    double abs_err = fabs(h_out[i] - h_out1[i]);
    double dot_length = MATRIX_M;
    double abs_val = fabs(h_out[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_out[i], h_out1[i], eps);
      correct = false;
      break;
    }
  }

  printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

  free(h_A);
  free(h_V);
  free(h_out);
  cudaFree(d_A);
  cudaFree(d_out);
  cudaFree(d_V);
}