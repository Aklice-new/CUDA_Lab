# 对CUDA中内置类型Vector Type进行了实际的测试比较

测试了float, float2, float4这几种类型的加载的耗时，从stackoverflow中的[关于vector type的讨论](https://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4)， 按照博客中和官方中的说法([CUDA Pro Tip: Increase Performance with Vectorized Memory Access | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)) 使用vector type可以提高访存的效率， 因为在load float, float2, float4这三种不同的类型时，会使用不同的指令。

```cpp
float -> LD.E
float2 -> LD.E.64
float4 -> LD.E.128
```

### 环境

GPU： 4060 laptop

OS：Ubuntu 20.04



### 问题

经过我本机比较结果是三者的差距并不是很大（测试了多次，每次的结果都不太相同，但是明显看不出差距）

NUMS_SIZE 100000000

blockPerGrid  NUMS_SIZE + 4 * BLOCK_SIZE + 1 / BLOCK+_SIZE * 4

threadPerBlock 256

kernel执行100轮

| 类型  | Naive       | reinterpret float2 | thrust float2 | thrust float4 |
| --- | ----------- | ------------------ | ------------- | ------------- |
| 耗时  | 546.2752 ms | 552.6108 ms        | 549.5071 ms   | 551.8030 ms   |

```cpp
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>

#define NUM_SIZE 100000000
#define steps 100
#define BLOCK_SIZE 256

__global__ void kernel_naive(float* a, float* b, float* c) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int idx_beg = 4 * tid;
  if (idx_beg < NUM_SIZE) {
    float a1 = a[idx_beg];
    float b1 = b[idx_beg];
    float a2 = a[idx_beg + 1];
    float b2 = b[idx_beg + 1];
    float a3 = a[idx_beg + 2];
    float b3 = b[idx_beg + 2];
    float a4 = a[idx_beg + 3];
    float b4 = b[idx_beg + 3];
    float c1 = a1 + b1;
    float c2 = a2 + b2;
    float c3 = a3 + b3;
    float c4 = a4 + b4;
    c[idx_beg] = c1;
    c[idx_beg + 1] = c2;
    c[idx_beg + 2] = c3;
    c[idx_beg + 3] = c4;
  }
}
__global__ void kernel_float2(float* a, float* b, float* c) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int idx_beg = 4 * tid;
  if (idx_beg < NUM_SIZE) {
    float2 a1 = reinterpret_cast<float2*>(&(a[idx_beg]))[0];
    float2 b1 = reinterpret_cast<float2*>(&(b[idx_beg]))[0];
    float2 a2 = reinterpret_cast<float2*>(&(a[idx_beg + 2]))[0];
    float2 b2 = reinterpret_cast<float2*>(&(b[idx_beg + 2]))[0];
    float2* c1 = reinterpret_cast<float2*>(&(c[idx_beg]));
    float2* c2 = reinterpret_cast<float2*>(&(c[idx_beg + 2]));
    c1->x = a1.x + b1.x;
    c1->y = a1.y + b1.y;
    c2->x = a2.x + b2.x;
    c2->y = a2.y + b2.y;
  }
}
__global__ void kernel_float2_thrust(float2* a, float2* b, float2* c) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int idx_beg = 2 * tid;
  if (idx_beg < NUM_SIZE) {
    float2 a1 = a[idx_beg];
    float2 b1 = b[idx_beg];
    float2 a2 = a[idx_beg + 1];
    float2 b2 = b[idx_beg + 1];
    float2 c1, c2;
    c1.x = a1.x + b1.x;
    c1.y = a1.y + b1.y;
    c2.x = a2.x + b2.x;
    c2.y = a2.y + b2.y;
    c[idx_beg] = c1;
    c[idx_beg + 1] = c2;
  }
}
__global__ void kernel_float4_thrust(float4* a, float4* b, float4* c) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int idx_beg = tid;
  if (idx_beg < NUM_SIZE / 4) {
    float4 a1 = a[idx_beg];
    float4 b1 = b[idx_beg];
    float4 c1;
    c1.x = a1.x + b1.x;
    c1.y = a1.y + b1.y;
    c1.z = a1.z + b1.z;
    c1.w = a1.w + b1.w;
    c[idx_beg] = c1;
  }
}

int main() {
  const float a = 3.0f;
  const float b = 4.0f;
  thrust::device_vector<float> d_a(NUM_SIZE, a);
  thrust::device_vector<float> d_b(NUM_SIZE, b);
  thrust::device_vector<float> d_c(NUM_SIZE, 0);
  float time1;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  dim3 blockPerDim = (NUM_SIZE + 4 * BLOCK_SIZE - 1) / (BLOCK_SIZE * 4);
  for (int i = 0; i < steps; i++) {
    kernel_naive<<<blockPerDim, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()));
  }
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&time1, start, stop);
  thrust::host_vector<float> h_a = d_c;
  for (int i = 0; i < NUM_SIZE; i++) {
    if (h_a[i] != (a + b)) {
      printf("Error!!!!! No. %d   result is %.4f \n", i, h_a[i]);
      break;
    }
  }
  printf("Naive version cost time : %.4f ms \n", time1);

  thrust::device_vector<float> d_a1(NUM_SIZE, a);
  thrust::device_vector<float> d_b1(NUM_SIZE, b);
  thrust::device_vector<float> d_c1(NUM_SIZE, 0);
  cudaEventRecord(start, 0);
  blockPerDim = (NUM_SIZE + 4 * BLOCK_SIZE - 1) / (BLOCK_SIZE * 4);
  for (int i = 0; i < steps; i++) {
    kernel_float2<<<blockPerDim, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_a1.data()),
        thrust::raw_pointer_cast(d_b1.data()),
        thrust::raw_pointer_cast(d_c1.data()));
  }
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&time1, start, stop);
  h_a = d_c1;
  for (int i = 0; i < NUM_SIZE; i++) {
    if (h_a[i] != (a + b)) {
      printf("Error!!!!! No. %d   result is %.4f \n", i, h_a[i]);
      break;
    }
  }
  printf("Naive float2 version cost time : %.4f ms \n", time1);

  thrust::device_vector<float> d_a2(NUM_SIZE, a);
  thrust::device_vector<float> d_b2(NUM_SIZE, b);
  thrust::device_vector<float> d_c2(NUM_SIZE, 0);
  cudaEventRecord(start, 0);
  blockPerDim = (NUM_SIZE + 4 * BLOCK_SIZE - 1) / (BLOCK_SIZE * 4);
  for (int i = 0; i < steps; i++) {
    kernel_float2_thrust<<<blockPerDim, BLOCK_SIZE>>>(
        (float2*)thrust::raw_pointer_cast(d_a2.data()),
        (float2*)thrust::raw_pointer_cast(d_b2.data()),
        (float2*)thrust::raw_pointer_cast(d_c2.data()));
  }
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&time1, start, stop);
  h_a = d_c2;
  for (int i = 0; i < NUM_SIZE; i++) {
    if (h_a[i] != (a + b)) {
      printf("Error!!!!! No. %d   result is %.4f \n", i, h_a[i]);
      break;
    }
  }
  printf("float2 Thrust version cost time : %.4f ms \n", time1);

  thrust::device_vector<float> d_a3(NUM_SIZE, a);
  thrust::device_vector<float> d_b3(NUM_SIZE, b);
  thrust::device_vector<float> d_c3(NUM_SIZE, 0);
  cudaEventRecord(start, 0);
  blockPerDim = (NUM_SIZE + 4 * BLOCK_SIZE - 1) / (BLOCK_SIZE * 4);
  for (int i = 0; i < steps; i++) {
    kernel_float4_thrust<<<blockPerDim, BLOCK_SIZE>>>(
        (float4*)thrust::raw_pointer_cast(d_a3.data()),
        (float4*)thrust::raw_pointer_cast(d_b3.data()),
        (float4*)thrust::raw_pointer_cast(d_c3.data()));
  }
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&time1, start, stop);
  h_a = d_c3;
  for (int i = 0; i < NUM_SIZE; i++) {
    if (h_a[i] != (a + b)) {
      printf("Error!!!!! No. %d   result is %.4f \n", i, h_a[i]);
      break;
    }
  }
  printf("float4 Thrust version1 cost time : %.4f ms \n", time1);
}
```



### 查看汇编指令

使用cuobjdump对程序的sass码进行查看

```cpp
cuobjdump ${executable_file} -sass
```

查看后确实是使用了LDG.E , LDG.E.64, LDG.E.128这些个指令，但是从执行的时间来看，并没有很大的差距。
