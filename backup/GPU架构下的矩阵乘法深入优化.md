### 矩阵setup

1. 使用NVIDIA V100S-PCIE-32GB测试
2. 行优先存储
3. shape为4096 x 4096
4. 大小为 128 MB，远大于shared memory
5. A矩阵为m行k列，B矩阵为k行n列，矩阵乘的结果C矩阵为m行n列
6. 峰值算力公式为 FP32-15.7 TFLOPS，FP16-125 TFLOPS

---
### 0. 朴素矩阵乘kernel实现

```cu
__global__ void NaiveGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (uint32_t i = 0; i < K; i++) {
            tmp += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = tmp;
    }
}

void LaunchNaiveGemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
  const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  NaiveGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaPeekAtLastError());
}

```

注意，在这里使用了行m为BlockX，列n为BlockY，因此是对m和n进行了并行，对每个thread只需要考虑计算k即可，并且BlockX是threadIdx中优先变化的，当x变化到blockDim-1时，y才会加1，也就是对二维并行来说变化效果是(0,0),(1,0),...,(m,0),(0,1),(1,1),...,(m,1),...。

对每个(x,y)坐标对来说，都需要在内部循环计算k，比如需要计算从(x, 0)到(x, k)和(0, y)到(k, y)的乘加，当计算完内部的k循环后，x加1，变成计算从(x+1, 0)到(x+1, k)和(0, y)到(k, y)的乘加，这个过程中x从0到m的计算，y都是不变的，索引变化如图。

<img width="812" height="708" alt="Image" src="https://github.com/user-attachments/assets/92d04125-9fa4-471f-b02e-01607b2def59" />

一定要记住 **GPU是并行** 的，可以这么理解，GPU的 $m \times n$ 个线程在同一时刻完成了K循环的同一步，K循环是一个时间的步数，每一步里都有$m \times k$ 个线程在访存和运算。

因此这个时刻对A矩阵来说就是获取了从(0,k)到(m,k)的一列数据，一列数据需要访存m次，一共k列，就是 $m \times k$ 次访存，这个时刻对B矩阵来说就是获取了(k,y)这一个数据，一共需要访问k行，就是 $k$ 次访存。A矩阵和B矩阵加起来进行了 $k + m \times k$ 次访存。

**最终结果**  245 GFLOPS，约为cuBlas的2%。

<img width="589" height="212" alt="Image" src="https://github.com/user-attachments/assets/7561f6f6-6e18-49cf-9f2a-3f5a3316e908" />

---
### 1. y优先kernel实现

```cu
__global__ void YfirstGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (uint32_t i = 0; i < K; i++) {
            tmp += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = tmp;
    }
}

void LaunchYfirstGemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
  const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  YfirstGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaPeekAtLastError());
}


```

使用列n为BlockX，行m为BlockY，这时列变成了threadIdx中优先变化的量，也就是先计算从(y, 0)到(y, k)和(0, x)到(k, x)的乘加，当计算完内部的k循环后，x加1，变成计算从(y, 0)到(y, k)和(0, x+1)到(k, x+1)的乘加，索引变化如图。

<img width="851" height="710" alt="Image" src="https://github.com/user-attachments/assets/00dcfb25-d387-4528-bd67-b4af06d85fb8" />

在这种情况下，GPU在同一时刻完成K循环的同一步，这个时刻对A矩阵来说就是获取了(y,k)这一个数据，一共需要访问m行，就是 $m$ 次访存，这个时刻对B矩阵来说就是获取了从(k,0)到(k,n)的一行数据，yfirst的 **优化点** 就在这，由于同一个GPU时间刻的连续SM单元在访问连续数据，因此会按warp合并访问，一次访存就可以访问32B数据即8个FP32数据，一行数据需要访存 $n \div 8$ 次，一共k行，就是 $\frac{n\times k}{8}$ 次访存。A矩阵和B矩阵加起来进行了 $m + \frac{n\times k}{8}$ 次访存。

由于本示例中 $m = n = k$，因此理论上性能相比naive实现会提升8倍，同时我们还可以看到，在x增加的过程中y是不变的，因此会重复获取y索引的数据，这是下一个优化点的问题来源。

**最终结果**  1894 GFLOPS，约为cuBlas的15%，可以看到，实际上相比naive实现提升了7.73倍，也就是约8倍。

<img width="583" height="211" alt="Image" src="https://github.com/user-attachments/assets/bee89804-f51c-4e19-8395-f7881ce6ecae" />

---
### 2. shared memory kernel实现

```cu
__global__ void SharedGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float blockA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE * BLOCK_SIZE];

    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0f;

        for (uint32_t i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            int a_col = i * BLOCK_SIZE + threadIdx.x;
            int b_row = i * BLOCK_SIZE + threadIdx.y;
            
            blockA[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[row * K + a_col];
            blockB[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[b_row * N + col];

            __syncthreads();
            for (int32_t j = 0; j < BLOCK_SIZE; j++) {
                tmp += blockA[threadIdx.y * BLOCK_SIZE + j] * blockB[j * BLOCK_SIZE + threadIdx.x];
            }
            __syncthreads();
        }
        
        C[row * K + col] = tmp;
    }
}

void LaunchSharedGemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
  const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  SharedGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaPeekAtLastError());
}
```

<img width="815" height="718" alt="Image" src="https://github.com/user-attachments/assets/d130bd02-ae35-4c31-bea7-454b5efc7000" />

**最终结果** 3322 GFLOPS，约为cuBlas的27%，可以看到，实际上相比naive实现提升了13.56倍。

<img width="587" height="216" alt="Image" src="https://github.com/user-attachments/assets/f1e9ff3e-9608-4ac0-9454-52b2b03960f0" />