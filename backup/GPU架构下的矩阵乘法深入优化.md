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
constexpr int BLOCK_SIZE = 32;

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

&emsp;&emsp;注意，在这里使用了行m为BlockX，列n为BlockY，因此是对m和n进行了并行，对每个thread只需要考虑计算k即可，并且BlockX是threadIdx中优先变化的，当x变化到blockDim-1时，y才会加1，也就是对二维并行来说变化效果是(0,0),(1,0),...,(m,0),(0,1),(1,1),...,(m,1),...。

&emsp;&emsp;对每个(x,y)坐标对来说，都需要在内部循环计算k，比如需要计算从(x, 0)到(x, k)和(0, y)到(k, y)的乘加，当计算完内部的k循环后，x加1，变成计算从(x+1, 0)到(x+1, k)和(0, y)到(k, y)的乘加，这个过程中x从0到m的计算，y都是不变的，索引变化如图。

<img width="812" height="708" alt="Image" src="https://github.com/user-attachments/assets/92d04125-9fa4-471f-b02e-01607b2def59" />

&emsp;&emsp;一定要记住 **GPU是并行** 的，可以这么理解，GPU的 $m \times n$ 个线程在同一时刻完成了K循环的同一步，K循环是一个时间的步数，每一步里都有 $m \times k$个线程在访存和运算。

&emsp;&emsp;因此这个时刻对A矩阵来说就是获取了从(0,k)到(m,k)的一列数据，一列数据需要访存m次，一共k列，就是 $m \times k$ 次访存，这个时刻对B矩阵来说就是获取了(k,y)这一个数据，一共需要访问k行，就是 $k$ 次访存。A矩阵和B矩阵加起来进行了 $k + m \times k$ 次访存。

**最终结果**  245 GFLOPS，约为cuBlas的2%。

<img width="589" height="212" alt="Image" src="https://github.com/user-attachments/assets/7561f6f6-6e18-49cf-9f2a-3f5a3316e908" />

---
### 1. y优先kernel实现

```cu
constexpr int BLOCK_SIZE = 32;

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

&emsp;&emsp;使用列n为BlockX，行m为BlockY，这时列变成了threadIdx中优先变化的量，也就是先计算从(y, 0)到(y, k)和(0, x)到(k, x)的乘加，当计算完内部的k循环后，x加1，变成计算从(y, 0)到(y, k)和(0, x+1)到(k, x+1)的乘加，索引变化如图。

<img width="851" height="710" alt="Image" src="https://github.com/user-attachments/assets/00dcfb25-d387-4528-bd67-b4af06d85fb8" />

&emsp;&emsp;在这种情况下，GPU在同一时刻完成K循环的同一步，这个时刻对A矩阵来说就是获取了(y,k)这一个数据，一共需要访问m行，就是 $m$ 次访存，这个时刻对B矩阵来说就是获取了从(k,0)到(k,n)的一行数据，yfirst的 **优化点** 就在这，由于同一个GPU时间刻的连续thread在访问连续数据，因此会按warp合并访问，一次访存就可以访问32B数据即8个FP32数据，一行数据需要访存 $n \div 8$ 次，一共k行，就是 $\frac{n \times k}{8}$ 次访存。A矩阵和B矩阵加起来进行了 $m + \frac{n \times k}{8}$ 次访存。

&emsp;&emsp;由于本示例中 $m = n = k$，因此理论上性能相比naive实现会提升8倍。

**最终结果**  1894 GFLOPS，约为cuBlas的15%，可以看到，实际上相比naive实现提升了7.73倍，也就是约8倍。

<img width="583" height="211" alt="Image" src="https://github.com/user-attachments/assets/bee89804-f51c-4e19-8395-f7881ce6ecae" />

---
### 2. shared memory kernel实现

```cu
constexpr int BLOCK_SIZE = 32;

__global__ void SharedGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float blockA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE * BLOCK_SIZE];

    uint32_t blockARow = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t blockBCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockARow < M && blockBCol < N) {
        float tmp = 0.0f;

        for (uint32_t i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            int blockACol = i * BLOCK_SIZE + threadIdx.x;
            int blockBRow = i * BLOCK_SIZE + threadIdx.y;

            int innerRow = threadIdx.y * BLOCK_SIZE;
            int innerCol = threadIdx.x;

            blockA[innerRow + innerCol] = (blockACol < K) ? A[blockARow * K + blockACol] : 0.0f;
            blockB[innerRow + innerCol] = (blockBRow < K) ? B[blockBRow * N + blockBCol] : 0.0f;

            __syncthreads();

            for (int32_t j = 0; j < BLOCK_SIZE; j++) {
                tmp += blockA[innerRow + j] * blockB[j * BLOCK_SIZE + innerCol];
            }
            __syncthreads();
        }

        C[blockARow * N + blockBCol] = tmp;
    }
}

void LaunchSharedGemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    SharedGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaPeekAtLastError());
}
```

&emsp;&emsp;由于shared memory的速度远大于global memory，因此我们在一个block内先用shared memory缓存该block的所有数据，然后再进行计算，计算时只会读写shared memory，最后再将结果写回global memory。

<img width="815" height="718" alt="Image" src="https://github.com/user-attachments/assets/d130bd02-ae35-4c31-bea7-454b5efc7000" />

&emsp;&emsp;由于使用了行m为BlockX，列n为BlockY，k进行内循环，因此对k不会再一个一个计算，而是以block长度统一计算。

&emsp;&emsp;首先要找到每个block的起点位置，对A矩阵来说，由于其block的起点行就是blockIdx.y * blockDim.y，这只是找到了block的起点行，为了找到thread还要加上threadIdx.y，而A矩阵的block起点列则需要通过内循环的K来获得，同样地，还要加上threadIdx.x定位thread。

&emsp;&emsp;对B矩阵来说，其block的起点行需要通过内循环的K来获得，为了定位thread需要加上threadIdx.y，而B矩阵的block的起点列则是blockIdx.x * blockDim.x，同样地，还要加上threadIdx.x定位thread。

&emsp;&emsp;要时刻注意，**GPU是并行的**，什么意思呢？看下面这个代码：

```cu
blockA[innerRow + innerCol] = A[blockARow * K + blockACol];
```

&emsp;&emsp;这里只获取了一个地址的值，为什么就对整个blockA赋值完成了？原因就在于blockARow中有threadIdx.y，而blockACol中有threadIdx.x，GPU是并行的，也就是在第K个时间步中同时对 $blockIdx.x \times threadIdx.x$ 个地址取值，那么就对整个blockA赋值了。blockB同理。

&emsp;&emsp;当进入到K+1个时间步时，A矩阵的blockARow没变，但是blockACol向右移动了整个BLOCK_SIZE，B矩阵的blockBCol没变，但是blockBRow向下移动了整个BLOCK_SIZE，此时获得的 $blockIdx.x \times threadIdx.x$ 个地址的值覆盖了blockA和blockB，进入了下一轮计算。

&emsp;&emsp;回到第K个时间步，由于GPU对整个block同时取值了，那么计算也需要对整个block进行计算，计算后的值临时存储在tmp中，进行所有时间步后也就计算完毕了A的K列和B的K行数据，此时再写回C矩阵即可。

&emsp;&emsp;这里又有一个问题：对整个block计算为什么只需要一个临时变量就可以了？不应该需要BLOCK_SIZE个吗？因为**GPU是并行的**，对blockA和blockB取值时用的innerRow和innerCol也有threadIdx，因此实际上在这个时间步是一整个BLOCK_SIZE在进行计算，使用的tmp变量是block那么多个thread自身私有的临时变量。

&emsp;&emsp;看上去for j循环是在进行blockA一行和blockB一列的计算，实际上innerRow和innerCol都是并行的，也就是for j循环是在同时对blockA的所有行和blockB的所有列进行计算，每一个j的时间步中，取得blockA的所有行的值和blockB的所有列的值进行乘加操作再放回各自thread的私有tmp变量中，当完成for j循环后，整个block的矩阵乘法就全部算出来了，并且每个对应位置的thread的私有tmp变量就存放着对应block位置的结果值，将其写回C即可完成计算。

&emsp;&emsp;shared memory的提升来源于多个方面，首先是A 和 B block 被缓存，只在global memory中访问一次，其次是内部计算全在寄存器和shared memory中，最后是warp内线程可以并行累加同一 block的元素。同时我们还可以看到，在x增加的过程中y是不变的，也就是说，x的每个值都会获取一次y的数据，x一共有n个，就会重复获取n次y的数据，当x增长到n-1之后，y才会加1，这是下一个优化点的问题来源。

**最终结果** 3322 GFLOPS，约为cuBlas的27%，相比naive实现提升了13.56倍。

<img width="587" height="216" alt="Image" src="https://github.com/user-attachments/assets/f1e9ff3e-9608-4ac0-9454-52b2b03960f0" />

---
### 3. tiling block kernel实现

```cu
constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 4;

__global__ void TilingGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE][BLOCK_SIZE];
    float tileC[TILE_SIZE][TILE_SIZE] = {0.0f};
    
    uint32_t blockARowStart = blockIdx.y * BLOCK_SIZE;
    uint32_t blockBColStart = blockIdx.x * BLOCK_SIZE;

    for (uint32_t i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        int blockAColStart = i * BLOCK_SIZE;
        int blockBRowStart = i * BLOCK_SIZE;
        
        #pragma unroll
        for (uint32_t j = 0; j < TILE_SIZE; j++) {
            for (uint32_t k = 0; k < TILE_SIZE; k++) {
                int innerRow = threadIdx.y * TILE_SIZE + j;
                int innerCol = threadIdx.x * TILE_SIZE + k;

                // Load matrix A sub-tile into Shared Memory with safe bounds check
                if ((blockARowStart + innerRow) < M && (blockAColStart + innerCol) < K)
                    blockA[innerRow][innerCol] = A[(blockARowStart + innerRow) * K + blockAColStart + innerCol];
                else
                    blockA[innerRow][innerCol] = 0.0f;
                
                // Load matrix B sub-tile into Shared Memory with safe bounds check
                if ((blockBRowStart + innerRow) < K && (blockBColStart + innerCol) < N)
                    blockB[innerRow][innerCol] = B[(blockBRowStart + innerRow) * N + blockBColStart + innerCol];
                else
                    blockB[innerRow][innerCol] = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++) {
            #pragma unroll
            for (uint32_t j = 0; j < TILE_SIZE; j++) {
                #pragma unroll
                for (uint32_t k = 0; k < TILE_SIZE; k++) {
                    tileC[j][k] += blockA[threadIdx.y * TILE_SIZE + j][dotIdx] * blockB[dotIdx][threadIdx.x * TILE_SIZE + k];
                }
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (uint32_t j = 0; j < TILE_SIZE; j++) {
        for (uint32_t k = 0; k < TILE_SIZE; k++) {
            int finalRow = blockARowStart + threadIdx.y * TILE_SIZE + j;
            int finalCol = blockBColStart + threadIdx.x * TILE_SIZE + k;
            if (finalRow < M && finalCol < N) {
                C[finalRow * N + finalCol] = tileC[j][k];
            }
        }
    }
}

void LaunchTilingGemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const dim3 block(BLOCK_SIZE / TILE_SIZE, BLOCK_SIZE / TILE_SIZE);
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    TilingGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaPeekAtLastError());
}
```

**最终结果** 4401 GFLOPS，约为cuBlas的31.84%，相比naive实现提升了17.96倍。

<img width="577" height="211" alt="Image" src="https://github.com/user-attachments/assets/0f99255a-a341-4837-b70a-ff5e8b7b3ce1" />
