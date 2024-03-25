#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "assert.h"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define checkCudaErrors(func)                                                \
  {                                                                          \
    cudaError_t e = (func);                                                  \
    if (e != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }

template <const int BLOCK_SIZE_M,           // block size M bm = 128
          const int BLOCK_SIZE_K,           // block size K bk = 8
          const int BLOCK_SIZE_N,           // block size N bn = 128
          const int THREAD_SIZE_Y,          // thread size rm = 8
          const int THREAD_SIZE_X,          // thread size rn = 8
          const bool ENABLE_DOUBLE_BUFFER>  // enable double buffer ?
__global__ void sgemm_my(float* __restrict__ A, float* __restrict__ B,
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
  // A_s
  // B_s来说每一行需要多少个线程来加载元素，这里采用的是float4类型对数据进行读取
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
  for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
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
    // 大循环是指，为了计算一个block bm * bn
    // 将A矩阵的 bm * K 和B矩阵的 K * bn分别划分成 bm * bk 和 bk * bn 个小块
    // 大循环里就是负责计算这些小块里的数据
    // double buffer
    // load next A_s
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
      // load next B_s
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(B_s[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) =
            FETCH_FLOAT4(
                B[OFFSET(B_TILE_ROW_START + tile_idx + i, B_TILE_COL, N)]);
      }
    }

    int load_stage_idx = write_stage_idx ^ 1;

    // 下面的内容就是小循环内完成的工作，
    // 具体工作是完成从A_s 和 B_s中依次读入rm, rn个数据进入寄存器
    // 然后把A_s 和 B_s中的数据加载到寄存器中
    // 依次计算每一次 rm + rn 个数据对该线程负责的 rm * rn 个数据的贡献
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE_K - 1; j++) {
      // load next tile from shared mem to register
      // from A_s to frag_a
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = FETCH_FLOAT4(
            A_s[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
      }
      // from B_s to frag_b
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = FETCH_FLOAT4(
            B_s[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
      }
// compute THREAD_SIZE_X * THREAD_SIZE_Y
#pragma unroll
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
          // accum 在每一次的小循环中保存的只是完整解的一部分
          accum[thread_y][thread_x] +=
              frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
        }
      }
    }

    // 为了保证所有线程都完成对它负责的区域的数据的计算 需进行同步
    __syncthreads();
    // double buffer
    write_stage_idx ^= 1;
    // 最后在完成最后一次小的迭代，至于为什么要将8次小迭代分为7次加1次，原文作者说这样是为了隐藏延迟
    // 这里完成的是对下一个tile内 rm + rn
    // 个数据的预读取，因为这里特殊一些，所以分开做，然后同时完成当前tile内最后一组
    // rm + rn个数据的计算
    // from A_s to frag_a
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
      FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(
          A_s[load_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // from B_s to frag_b
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(
          B_s[load_stage_idx ^ 1][0][THREAD_SIZE_X * tx + thread_x]);
    }
// compute THREAD_SIZE_X * THREAD_SIZE_Y
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
#pragma unroll
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++) {
        // accum 在每一次的小循环中保存的只是完整解的一部分
        accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
      }
    }

  } while (tile_idx < K);
// 完成大循环后，该block中该线程负责的rm * rn的数据已经计算完成
// store back to C
#pragma unroll
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++) {
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(
          C[OFFSET(BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                   BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x, N)]) =
          FETCH_FLOAT4(accum[thread_y][thread_x]);
    }
  }
}

// K: ldA
// N: ldB
template <
    const int BLOCK_SIZE_M,   // height of block of C that each thread block
                              // calculate
    const int BLOCK_SIZE_K,   // width of block of A that each thread block load
                              // into shared memory
    const int BLOCK_SIZE_N,   // width of block of C that each thread block
                              // calculate
    const int THREAD_SIZE_Y,  // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER  // whether enable double buffering or not
    >
__global__ void Sgemm(float* __restrict__ A, float* __restrict__ B,
                      float* __restrict__ C, const int M, const int N,
                      const int K) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // the threads number in Block of X,Y
  const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
  const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
  const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

  // thread id in cur Block
  const int tid = ty * THREAD_X_PER_BLOCK + tx;

  // shared memory
  __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
  __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
  // registers for C
  float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
  // registers for A and B
  float frag_a[2][THREAD_SIZE_Y];
  float frag_b[2][THREAD_SIZE_X];
  // registers load global memory
  const int ldg_num_a =
      BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
  const int ldg_num_b =
      BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
  float ldg_a_reg[4 * ldg_num_a];
  float ldg_b_reg[4 * ldg_num_b];

  // threads number in one row
  const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
  const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

  // row number and col number that needs to be loaded by this thread
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

  // row stride that thread uses to load multiple rows of a tile
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

  A = &A[(BLOCK_SIZE_M * by) * K];
  B = &B[BLOCK_SIZE_N * bx];

// transfer first tile from global mem to shared mem
// load A from global memory to shared memory
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
    int ldg_index = i / A_TILE_ROW_STRIDE * 4;
    FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
        FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START + i,  // row
                              A_TILE_COL,            // col
                              K)]);
    As[0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
    As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
    As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
    As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
  }
// load B from global memory to shared memory
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
    FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) =
        FETCH_FLOAT4(B[OFFSET(B_TILE_ROW_START + i,  // row
                              B_TILE_COL,            // col
                              N)]);
  }
  __syncthreads();
// load A from shared memory to register
#pragma unroll
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
    FETCH_FLOAT4(frag_a[0][thread_y]) =
        FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
  }
// load B from shared memory to register
#pragma unroll
  for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
    FETCH_FLOAT4(frag_b[0][thread_x]) =
        FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
  }

  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += BLOCK_SIZE_K;
    // load next tile from global mem
    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
            FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START + i,   // row
                                  A_TILE_COL + tile_idx,  // col
                                  K)]);
      }
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        int ldg_index = i / B_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
            FETCH_FLOAT4(B[OFFSET(tile_idx + B_TILE_ROW_START + i,  // row
                                  B_TILE_COL,                       // col
                                  N)]);
      }
    }

    int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
    for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
// load next tile from shared mem to register
// load A from shared memory to register
#pragma unroll
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = FETCH_FLOAT4(
            As[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
      }
// load B from shared memory to register
#pragma unroll
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = FETCH_FLOAT4(
            Bs[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
      }
// compute C THREAD_SIZE_X x THREAD_SIZE_Y
#pragma unroll
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
          accum[thread_y][thread_x] +=
              frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
        }
      }
    }

    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index];
        As[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index + 1];
        As[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index + 2];
        As[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_index + 3];
      }
// load B from global memory to shared memory
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        int ldg_index = i / B_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) =
            FETCH_FLOAT4(ldg_b_reg[ldg_index]);
      }
      // use double buffer, only need one sync
      __syncthreads();
      // switch
      write_stage_idx ^= 1;
    }

// load first tile from shared mem to register of next iter
// load A from shared memory to register
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
      FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(
          As[load_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + thread_y]);
    }
// load B from shared memory to register
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(
          Bs[load_stage_idx ^ 1][0][THREAD_SIZE_X * tx + thread_x]);
    }
// compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
        accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
      }
    }
  } while (tile_idx < K);

// store back to C
#pragma unroll
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(
          C[OFFSET(BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                   BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x, N)]) =
          FETCH_FLOAT4(accum[thread_y][thread_x]);
    }
  }
}
int main(int argc, char** argv) {
  if (argc != 4) {
    printf("usage: ./main [M] [K] [N]\n");
    exit(0);
  }
  size_t M = atoi(argv[1]);
  size_t K = atoi(argv[2]);
  size_t N = atoi(argv[3]);

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

  const int BLOCK_SIZE_M = 128;
  const int BLOCK_SIZE_K = 8;
  const int BLOCK_SIZE_N = 128;
  const int THREAD_SIZE_X = 8;
  const int THREAD_SIZE_Y = 8;
  const bool ENABLE_DOUBLE_BUFFER = false;

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
  int nIter = 1000;

  checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0; run < nIter; run++) {
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    sgemm_my<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y,
             THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
  }
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

  // cublas
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  float alpha = 1.0;
  float beta = 0;
  checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0; run < nIter; run++) {
    cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K,
                d_B, N, &beta, d_C, N);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

  msecPerMatrixMul[1] = msecTotal / nIter;
  gigaFlops[1] =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
  printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
         gigaFlops[1], msecPerMatrixMul[1], flopsPerMatrixMul);

  cublasDestroy(blas_handle);

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
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_C[i], h_C1[col * M + row], eps);
      correct = false;
      break;
    }
  }

  printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
  printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

  // Free Memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C1);
}
