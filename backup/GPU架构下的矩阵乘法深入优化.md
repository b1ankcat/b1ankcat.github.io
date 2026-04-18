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

                if ((blockARowStart + innerRow) < M && (blockAColStart + innerCol) < K)
                    blockA[innerRow][innerCol] = A[(blockARowStart + innerRow) * K + blockAColStart + innerCol];
                else
                    blockA[innerRow][innerCol] = 0.0f;
                
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

&emsp;&emsp;在shared memory gemm kernel中，每个thread为了执行一次计算需要从shread memory中读取2个float，此时的计算访存比是1:2，虽然shread memory很快，但是当所有线程都在进行高频小数据读取时，shread memory的带宽还是很容易成为瓶颈，因此为了提高对shread memory的访存比，我们模仿shread memory缓存global memory的方式，用registers来缓存shread memory。

&emsp;&emsp;这个方法被称为Tiling算法，也就是在Block中再划分出一个小Block成为Tile，每个线程现在不再只计算一个元素，而是对这个Tile的所有元素进行计算，我们设定的Tile大小是4，也就是16次计算，这时只需要8次访存，计算访存比达到了2:1，相比shared memory提升了四倍，其索引取值如图所示。

<img width="1077" height="1053" alt="Image" src="https://github.com/user-attachments/assets/dc5b9aad-7ea2-479d-8436-cf440aac6001" />

&emsp;&emsp;可以看到，先用blockIdx来定位Block的起点，由于blockIdx是在M和N上并行，而K是内部并行，用blockIdx.y和K来定位Block A的起点坐标，用K和blockIdx.x来定位Block B的起点坐标。另一方面，由于每个线程不再只计算一个元素，而是计算TILE_SIZE长宽高的Tile，因此blockDim也不再是BLOCK_SIZE了，而是BLOCK_SIZE / TILE_SIZE，由于这个原因，在定位Block起点的时候就不能乘以blockDim了，因为在逻辑上每个Block还是处理了BLOCK_SIZE长宽高的数据，只是blockDim保存的BLOCK_SIZE / TILE_SIZE，所以BlockStart是用blockIdx * BLOCK_SIZE表示。

&emsp;&emsp;在加载一个Block数据到shared memory中时，依然对K进行内循环，通过K来确定Block A的列坐标和Block B的heng坐标，这样就能获取Block的左上角起点坐标了。

&emsp;&emsp;下一步是获取Block内部每个Tile的左上角起点坐标。由于每个Block现在不再有BLOCK_SIZE个thread，而是BLOCK_SIZE / TILE_SIZE个thread，每个threadY要处理TILE_SIZE行，每个threadX要处理TILE_SIZE列，因此每个Tile的左上角起点坐标就是threadIdx * TILE_SIZE的坐标。

&emsp;&emsp;现在可以加载数据了，加载用的for j和for k循环次数用的常数TILE_SIZE，也就是GPU会执行TILE_SIZE * TILE_SIZE次加载，每一个时间步中，所有thread同时对整个Block的所有Tile加载一个元素，执行完时间步后就加载完了整个Block，此时的加载逻辑就只需要将shared memory的Block和A矩阵、B矩阵的对应地址元素对应即可。

&emsp;&emsp;对加载来说，每个thread都是从tile的左上角开始，每个时间步移动一个元素，经过TILE_SIZE*TILE_SIZE个时间步后加载完正片Tile，每个时间步中threadX和threadY并发进行，也就是同时对一个Block的所有Tile进行加载，并且在每个时间步中同时对A矩阵和B矩阵进行加载。

&emsp;&emsp;而到了计算的时候，tileC其实是每个thread私有的一片reg空间，并且是计算的是最后的结果，那么每个对应tileC[j][k]的位置都需要一整行的blockA和一整列的blockB才能得到最后结果，并且每个thread需要计算TILE_SIZE次，因此还需要进行for dotIdx循环BLOCK_SIZE次，才能获取K轴的全部数据，并且每个线程计算了整个Tile的结果。

&emsp;&emsp;具体到代码中来说，每个thread在循环中计算blockA的对应Tile的第j行所有数据，因此第一个维度填写j，而第二个维度填写dotIdx，因为blockA在行维度并行，因此第一个维度还要加上threadIdx.y * TILE_SIZE来定位具体的Tile。同理，对blockB来说计算对应Tile的第k列所有数据，因此第一个维度填写dotIdx，而第二个维度填写k加上列并行的threadIdx.x * TILE_SIZE。

&emsp;&emsp;以GPU并行的视角来看，相当于每个thread进行了BLOCK_SIZE次TILE_SIZE * TILE_SIZE个时间步，在每个时间步中，每进行BLOCK_SIZE次时间步，thread就会计算出Block中所有Tile的一个值，一共进行TILE_SIZE * TILE_SIZE次BLOCK_SIZE时间步最终计算完Block中所有Tile的所有值，在BLOCK_SIZE次时间步中，thread是在对block的K轴数据进行乘加，而在TILE_SIZE * TILE_SIZE时间步中，thread是在计算需要完成的TILE_SIZE行TILE_SIZE列数据。

&emsp;&emsp;最后需要将数据写回global memory，此时对应Block的对应Tile的数据就是正确数据了，因此不用再对K轴进行内部循环，直接将对应thread的Tile值放回C矩阵即可，同样地，所有thread会在同一时刻对Block内的所有Tile进行写回操作，只需要进行TILE_SIZE * TILE_SIZE次时间步就可以完成Block内所有Tile的所有元素写回。

**最终结果** 4401 GFLOPS，约为cuBlas的31.84%，相比naive实现提升了17.96倍。

<img width="577" height="211" alt="Image" src="https://github.com/user-attachments/assets/0f99255a-a341-4837-b70a-ff5e8b7b3ce1" />

---
### 4. vector load kernel实现

```cu
constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 4;

__global__ void VecGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float blockB[BLOCK_SIZE][BLOCK_SIZE + 1];
    float tileC[TILE_SIZE][TILE_SIZE] = {0.0f};

    uint32_t blockARowStart = blockIdx.y * BLOCK_SIZE;
    uint32_t blockBColStart = blockIdx.x * BLOCK_SIZE;
    
    for (uint32_t i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        int blockAColStart = i * BLOCK_SIZE;
        int blockBRowStart = i * BLOCK_SIZE;

        #pragma unroll
        for (uint32_t j = 0; j < TILE_SIZE; j++) {
            for (uint32_t k = 0; k < TILE_SIZE / 4; k++) {
                int innerRow = threadIdx.y * TILE_SIZE + j;
                int innerCol = threadIdx.x * TILE_SIZE + k * 4;

                if ((blockARowStart + innerRow) < M && (blockAColStart + innerCol) < K){
                    float4 tmpA = reinterpret_cast<const float4*>(&A[(blockARowStart + innerRow) * K + blockAColStart + innerCol])[0];
                    blockA[innerRow][innerCol + 0] = tmpA.x;
                    blockA[innerRow][innerCol + 1] = tmpA.y;
                    blockA[innerRow][innerCol + 2] = tmpA.z;
                    blockA[innerRow][innerCol + 3] = tmpA.w;
                }
                else{
                    blockA[innerRow][innerCol + 0] = 0.0f;
                    blockA[innerRow][innerCol + 1] = 0.0f;
                    blockA[innerRow][innerCol + 2] = 0.0f;
                    blockA[innerRow][innerCol + 3] = 0.0f;
                }
                
                if ((blockBRowStart + innerRow) < K && (blockBColStart + innerCol) < N){
                    float4 tmpB = reinterpret_cast<const float4*>(&B[(blockBRowStart + innerRow) * N + blockBColStart + innerCol])[0];
                    blockB[innerRow][innerCol + 0] = tmpB.x;
                    blockB[innerRow][innerCol + 1] = tmpB.y;
                    blockB[innerRow][innerCol + 2] = tmpB.z;
                    blockB[innerRow][innerCol + 3] = tmpB.w;
                }
                else{
                    blockB[innerRow][innerCol + 0] = 0.0f;
                    blockB[innerRow][innerCol + 1] = 0.0f;
                    blockB[innerRow][innerCol + 2] = 0.0f;
                    blockB[innerRow][innerCol + 3] = 0.0f;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++) {
            float regA[TILE_SIZE];
            float regB[TILE_SIZE];

            #pragma unroll
            for (int j = 0; j < TILE_SIZE; j++) {
                regA[j] = blockA[threadIdx.y * TILE_SIZE + j][dotIdx];
                regB[j] = blockB[dotIdx][threadIdx.x * TILE_SIZE + j];
            }

            #pragma unroll
            for (int j = 0; j < TILE_SIZE; j++) {
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; k++) {
                    tileC[j][k] += regA[j] * regB[k];
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

void LaunchVecGemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const dim3 block(BLOCK_SIZE / TILE_SIZE, BLOCK_SIZE / TILE_SIZE);
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    VecGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaPeekAtLastError());
}
```
&emsp;&emsp;首先，为了加速数据加载，我们使用 float4 进行矢量化。原本在 K 轴的循环需要进行 TILE_SIZE 次，现在每次可以一次性加载连续的 4 个 float，因此循环次数可除以 4，同时 innerCol 也按 stride 进行调整，每个线程一次性加载一行中的四列数据。

&emsp;&emsp;每个线程只需进行 TILE_SIZE 次循环，每次循环将对应 Tile 的一列 A 和一行 B 加载到寄存器中，然后执行外积计算。通过这种方式，每次循环就完成了 Tile 的一个时间步计算，即 (TILE_SIZE, 1) 与 (1, TILE_SIZE) 的乘法累加操作。经过 BLOCK_SIZE 次循环后，整个 Tile 的 K 轴累加完成，Tile 内所有元素的最终值也随之计算完成，从而完成矩阵乘法的一部分。

**最终结果** 7657 GFLOPS，约为cuBlas的60.34%，相比naive实现提升了31.25倍。

<img width="618" height="213" alt="Image" src="https://github.com/user-attachments/assets/52cdf8d3-ce03-4ab6-a67d-d8bfbf49732d" />

---
### 5. warp tiling + double buffer kernel实现

```cu
```

**最终结果** 7657 GFLOPS，约为cuBlas的60.34%，相比naive实现提升了31.25倍。