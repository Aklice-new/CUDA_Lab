#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "assert.h"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer)[0])
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

  // 计算该线程负责读取的tile的row的位置
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
  // 然后计算该线程读取的tile的列的位置, tid表示的是block的线程索引，分别得到A，
  // B 中的列的索引
  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
  // block中线程的总数除以读取一行所需要的线程数得到步长，因为可能一次性读不完所有数据
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

  // 通过block的位置来定位到A B矩阵的对应行和对应列
  A = &A[(BLOCK_SIZE_M * by) * K];
  B = &B[BLOCK_SIZE_N * bx];
  // 首先将数据加载到A_s和B_s中来， 这时第一次加载，全部都加载到0中
  // double buffer
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
    // int ldg_index = i / A_TILE_ROW_STRIDE * 4;
    // 下面这部分是将本来 bm * bk 的区域进行转置，读入到 bk * bm 的区域中来
    float4 data = FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL, K)]);
    A_s[0][A_TILE_COL][A_TILE_ROW_START + i] = data.x;
    A_s[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = data.y;
    A_s[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = data.z;
    A_s[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = data.w;
  }
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_N; i += B_TILE_ROW_STRIDE) {
    // 读入B矩阵的时候不需要进行转置
    FETCH_FLOAT4(B_s[0][B_TILE_ROW_START + i][B_TILE_COL]) =
        FETCH_FLOAT4(B[OFFSET(B_TILE_ROW_START + i, B_TILE_COL, N)]);
  }
  __syncthreads();  // 保证数据加载完成

// 将该线程负责计算的 rm * rn区域内所需要的数据从A_s中加载到寄存器frag_a
#pragma unroll
  // 通过float4类型进行读取的，所以每次加4， THREAD_SIZE_Y 就是 rm
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
    // 这里因为A_s是转置存储的，所以本来需要读取A_s的一列，现在变为读取一行了
    // 这里是从第一列（行）开始读取的所以A_s第一维是0
    // THREAD_SIZE_Y * ty 表示的是当前这个线程负责的那一片
    // rm * rn 的区域的起始位置
    FETCH_FLOAT4(frag_a[0][thread_y]) =
        FETCH_FLOAT4(A_s[0][0][THREAD_SIZE_Y * ty + thread_y]);
  }
#pragma unroll
  for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
    FETCH_FLOAT4(frag_b[0][thread_x]) =
        FETCH_FLOAT4(B_s[0][0][THREAD_SIZE_X * tx + thread_x]);
  }

  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += BLOCK_SIZE_K;
    // double buffer 从全局内存中加载下一个tile  大循环
    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        float4 data = FETCH_FLOAT4(
            A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL + tile_idx, K)]);
        A_s[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i] = data.x;
        A_s[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] = data.y;
        A_s[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] = data.z;
        A_s[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] = data.w;
      }
    }
    // load next B_s

    // compute

    // update
  } while (1);
}