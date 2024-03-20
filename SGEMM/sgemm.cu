#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "assert.h"

#define FETCH_FLOAT4(pointer) (reinterpreter_cast<float4*>(&pointer)[0])
#define OFFSET(row, col, ld) ((row * ld) + col)
#define CHECK_CUDA_ERROR(func)                                                 \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess) {                                                    \
      printf("%s %d CUDA : %s \n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                          \
  }
template <const int BLOCK_SIZE_M,           // block size M bm = 128
          const int BLOCK_SIZE_N,           // block size N bn = 128
          const int BLOCK_SIZE_K,           // block size K bk = 8
          const int THREAD_SIZE_Y,          // thread size rm = 8
          const int THREAD_SIZE_X,          // thread size rn = 8
          const bool ENABLE_DOUBLE_BUFFER>  // enable double buffer ?
__global__ void sgemm(float* __restrict__ A, float* __restrict__ B,
                      float* __restrict__ C, const int M, const int N,
                      const int K) {
  // block块的坐标
  int bx = blockIdx.x;
  int by = blockIdx.y;
  // thread的坐标
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 每个block中线程横向和纵向的线程个数
  const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
  const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
  const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

  int tid = ty * THREAD_X_PER_BLOCK + tx;

  // 为计算改block所申请的共享内存的空间，2倍大小是为了做double buffer
  __shared__ float A_s[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
  __shared__ float B_s[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
  // 该线程负责计算的一小块 rm * rn 寄存器类型
  float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

  // frag_a rm * 1  frag_b  1 * rn 寄存器类型
  float frag_a[2][THREAD_SIZE_Y];
  float frag_b[2][THREAD_SIZE_X];

  // 首先要从global memory中把数据加载到 shared memory中
  // shared memory size = (bm * bk) = 128 * 8
  // 一个block中256个线程，则每个线程负责 128 * 8 / 256= 4 个元素的加载
  // A_s,B_s来说每一行需要多少个线程来加载元素，这里采用的是float4类型对数据进行读取
  // 即一次性读入4个元素
  const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
  const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
  // 然后计算该线程读取的tile的列的位置, tid表示的是block的线程索引，分别得到A，
  // B 中的列的索引
  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
  // block中线程的总数除以读取一行所需要的线程数得到步长，因为可能一次性读不完所有数据
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;
}