### **本文是CppCon演讲《Matrix Multiplication Deep Dive》的文字整理和一些自己的理解**

---
### 矩阵setup

1. std::vector<double>并且对齐64 bytes的cpu cache
2. 行优先存储
3. shape为2880 x 2880
4. 大小为63 MB，大于L3 cache size (6 MB)
5. 峰值算力公式如下:

$$
\text{FLOPS} = \text{cores} \times \frac{\text{cycles}}{\text{seconds}} \times \frac{\text{FLOPs}}{\text{cycles}}
$$

---
### 0. 朴素c++实现

```c++
void matmul_naive(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            for (int k = 0; k < K; ++k){
                C(i, j) += A(i, k) * B(k, j);  // 最内层按一行A乘以一列B得到一个C的顺序计算
            }
        }
    }
}
```

<img width="897" height="236" alt="Image" src="https://github.com/user-attachments/assets/9040bb6f-5244-4755-aa70-d9b6be699953" />

---
### 1. 改变循环顺序

```c++
void matmul_change_order(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    for (int i = 0; i < M; ++i){
        for (int k = 0; k < K; ++k){
            for (int j = 0; j < N; ++j){
                C(i, j) += A(i, k) * B(k, j);  // 最内层按一个A乘以一行B得到一行C的顺序计算
            }
        }
    }
}
```

在 **i-j-k** 顺序下，在最内层的循环中，可以看到对 **A(i, k)** 的访问是连续的，但是 **B(k, j)** 的访问会有一个 **j** 大小的Strided。

<img width="491" height="499" alt="Image" src="https://github.com/user-attachments/assets/fdd96abf-3291-4968-af63-33f94aba5efc" />

改为 **i-k-j** 顺序后，在最内层的循环中，可以看到对于A矩阵，**A(i, k)** 的访问还是连续的，而此时对 **B(k, j)** 和 **C(i, j)** 的访问也会变成连续的的了。这不仅最大化了缓存行利用率，还能有效触发硬件预取器。

<img width="485" height="488" alt="Image" src="https://github.com/user-attachments/assets/5d5739ca-d0fc-41c2-9a62-2d109b37f06a" />

**最终结果** 性能相比Naive实现提升了约12.6倍。

---
### 2. Tiling分片

```c++
void matmul_tiling(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    constexpr auto BLOCK = 64;
    
    for (int ib = 0; ib < M; ib += BLOCK){
        for (int kb = 0; kb < K; kb += BLOCK){
            for (int jb = 0; jb < N; jb += BLOCK){
                const double* a = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       c = &C(ib, jb);
                
                for (int i = 0; i < BLOCK; ++i, c+=N, a+=K){
                    const double* b = mb;
                    for (int k = 0; k < BLOCK; ++k, b+=N){
                        for (int j = 0; j < BLOCK; ++j){
                            c[j] += a[k] * b[j];
                        }
                    }
                }
            }
        }
    }
}
```

将块外循环和块内循环分离，通过增加数据的复用率来突破Memory Wall，大幅提升Arithmetic Intensit.：
```c++
template<int BLOCK>
void matmul_block_kernel(const double* a, const double* mb, double* c, int N, int K)
{
    /** 
    * 对BLOCK内的i循环时，i移动时，c和a也要跟着移动
    * i是逻辑计数，限制只会进行BLOCK这么多行循环
    * 而i在逻辑上移动了行后，a和c的基址还需要再内存中移动一行
    **/
    for (int i = 0; i < BLOCK; ++i, c+=N, a+=K){  
        const double* b = mb;
        for (int k = 0; k < BLOCK; ++k, b+=N){  // 同理，k移动一列，内存b的基址也要移动一列
            for (int j = 0; j < BLOCK; ++j){
                c[j] += a[k] * b[j];
            }
        }
    }
}

void matmul_tiling(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    constexpr auto BLOCK = 64;
    
    for (int ib = 0; ib < M; ib += BLOCK){
        for (int kb = 0; kb < K; kb += BLOCK){
            for (int jb = 0; jb < N; jb += BLOCK){
                const double* a = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       c = &C(ib, jb);
                matmul_block_kernel<BLOCK>(a, mb, c, N, K);
            }
        }
    }
}
```

<img width="426" height="491" alt="Image" src="https://github.com/user-attachments/assets/58ab61c1-c7f8-4f20-a3c4-3fe4c24f75ff" />

**最终结果** 性能相比Naive实现提升了约38.9倍。

---
### 3. 矢量化

<img width="962" height="365" alt="Image" src="https://github.com/user-attachments/assets/58b4578a-8ee8-45e7-97d9-439704012887" />

```c++
template<int BLOCK>
void matmul_avxblock_kernel(const double* a, const double* mb, double* c, int N, int K)
{
    constexpr int avx_doubles = 256 / (sizeof(double) * 8);
    for (int i = 0; i < BLOCK; ++i, c+=N, a+=K){
        const double* b = mb;
        for (int k = 0; k < BLOCK; ++k, b+=N){
            __m256d a_reg = _mm256_broadcast_sd(&a[k]);  // 将一个a广播为一个矢量
            for (int j = 0; j < BLOCK; j+=avx_doubles){
                __m256d b_reg = _mm256_loadu_pd(&b[j]);  // 加载avx_doubles个b和c进reg
                __m256d c_reg = _mm256_loadu_pd(&c[j]);
                c_reg  = _mm256_fmadd_pd(a_reg, b_reg, c_reg);
                _mm256_storeu_pd(&c[j], c_reg);  // 将矢量乘加后的reg写回c
            }
        }
    }
}
```

**最终结果** 性能相比Naive实现提升了约48.13倍。

---
### 4. 基于缓存的tiling分块

<img width="565" height="797" alt="Image" src="https://github.com/user-attachments/assets/4a95add4-8108-4a2a-903c-485c756eb7f5" />

将大矩阵切分为适合装入 L1 和 L2 Cache 的小块。

```c++
template<int Nr, int Mr, int Kc, int Nc>
void matmul_cache_block_kernel(const double* a, const double* mb, double* c, int N, int K)
{
    constexpr int avx_doubles = 256 / (sizeof(double) * 8);
    for (int i = 0; i < Mr; ++i, c+=N, a+=K){
        const double* b = mb;
        for (int k = 0; k < Kc; ++k, b+=N){
            __m256d a_reg = _mm256_broadcast_sd(&a[k]);  // 将一个a广播为一个矢量
            for (int j = 0; j < Nr; j+=avx_doubles){
                __m256d b_reg = _mm256_loadu_pd(&b[j]);  // 加载avx_doubles个b和c进reg
                __m256d c_reg = _mm256_loadu_pd(&c[j]);
                c_reg  = _mm256_fmadd_pd(a_reg, b_reg, c_reg);
                _mm256_storeu_pd(&c[j], c_reg);  // 将矢量乘加后的reg写回c
            }
        }
    }
}

void matmul_cache_tiling(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    /** 
    * Nc, Mc, Kc是block cahce的大小
    * Nr, Mr是block register的大小
    **/
    constexpr auto Mc = 180, Nc = 96, Kc = 240, Nr = 12, Mr = 4;
    
    /** 
    * 前三层循环先用Nc, Mc, Kc替换掉BLOCK
    * 这样可以让L2和L3 Cache保存最大的矩阵块
    **/
    for (int ib = 0; ib < M; ib += Mc){
        for (int kb = 0; kb < K; kb += Kc){
            for (int jb = 0; jb < N; jb += Nc){
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);
                /** 
                * 进一步将 Cache 块细分为更小的 Register Block, 大小为 Mr x Nr
                * 存储C矩阵需要 Mr x (Nr / 4) 个寄存器
                * 加载B矩阵需要 Nr / 4 个寄存器
                * 广播 A 元素需要1个寄存器
                * 当寄存器总数为16时，Mr = 4, Nr = 12
                **/
                for (int i2 = 0; i2 < Mc; i2+=Mr){
                    for (int j2 = 0; j2 < Nc; j2+=Nr){
                        const double* a = &ma[i2 * K];  // ma的基地址加上已经移动到第i2行，每行K个元素
                        const double* b = &mb[j2];  // mb向右移动j2列
                        double*       c = &mc[i2 * N + j2];  // mc既要移动行，也要移动列
                        matmul_cache_block_kernel<Nr, Mr, Kc, Nc>(a, b, c, N, K);
                    }
                }
            }
        }
    }
}
```

**最终结果** 性能相比Naive实现提升了约56.1倍。

---
### 5. 寄存器分配

在最内层内核中，显式地向编译器提供寄存器使用提示，确保充分利用 CPU 的所有向量寄存器。这极大地减少了 Load/Store 指令，避免了 Register Spilling（寄存器溢出），从而将指令吞吐量推向理论峰值。

```c++
template<int Nr, int Mr, int Kc, int Nc>
void matmul_reg_block_kernel(const double* ma, const double* b, double* c, int N, int K)
{
    constexpr int avx_doubles = 256 / (sizeof(double) * 8);
    /** 
    * std::array<__m256d, CREG_CNT> res 实际上是告诉编译器, 
    * 需要 CREG_CNT 个 YMM 寄存器来暂存中间结果
    **/
    constexpr int CREG_CNT{Mr * Nr / avx_doubles};
    std::array<__m256d, CREG_CNT> res;
    for (int idx = 0; idx < CREG_CNT; ++idx){
        res[idx] = __mm256_setzero_pd();
    }
    
    /** 
    * 在矩阵乘法的Naive实现中，每次计算都需要访存
    * 现在C 的计算结果始终驻留在寄存器中
    **/
    for (int k = 0; k < Kc; ++k, b+=N){
        const double* a = ma;
        int idx = 0;
        for (int i = 0; i < Mr; ++i, a+=K){
            __m256d a_reg = _mm256_broadcast_sd(&a[k]); 
            for (int j = 0; j < Nr; j+=avx_doubles, ++idx){
                res[idx] = _mm256_fmadd_pd(a_reg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
        }
    }

   /** 
    * 现在才开始批量写回内存
    **/ 
    int idx = 0;
    for (int i = 0; i < Mr; ++i, c+=N){
        for (int j = 0; j < Nr; j+=avx_doubles, ++idx){
            load_inc_store_double(&c[j], res[idx]);
        }
    }
}
```

**最终结果** 性能相比Naive实现提升了约108倍。

---
### 6. 多线程并行

将计算任务按 C 的行进行切分，分配给多个 CPU 核心。

```c++
#pragma omp parallel for   // 线程间不需要同步
    for (int ib = 0; ib < M; ib += Mc){
        for (int kb = 0; kb < K; kb += Kc){
            for (int jb = 0; jb < N; jb += Nc){
               // ...
            }
        }
    }
```

<img width="418" height="207" alt="Image" src="https://github.com/user-attachments/assets/345dde22-fc33-4cff-aca3-cf141eaa32f9" />

**最终结果** 性能相比Naive实现提升了约326.1倍。

---
### 7. 矩阵tile布局

在进入内核计算之前，将具有 Strided 内存布局的子 Tile，预先 Copy 到一段连续的内存 Buffer 中。这样可以消除最内层循环中的 Stride 计算，确保在读取 A 和 B 矩阵块时拥有连续访问。

<img width="284" height="451" alt="Image" src="https://github.com/user-attachments/assets/e706d5a4-d5fa-4850-95bb-923a2e72fc6d" />

```c++
template<int Nr, int Mr, int Kc>
void matmul_reg_block_reordered_kernel(const double* a, const double* b, double* c, int N)
{
    constexpr int avx_doubles = 256 / (sizeof(double) * 8);
    constexpr int CREG_CNT{Mr * Nr / avx_doubles};
    std::array<__m256d, CREG_CNT> res;
    for (int idx = 0; idx < CREG_CNT; ++idx){
        res[idx] = __mm256_setzero_pd();
    }
    
   /** 
    * 原本属于A矩阵不同列的元素已经成为连续存储的块, 
    * 其宽度正好等于寄存器的大小 Mr
    * B矩阵同理，成为连续的块后宽度为 Nr
    **/ 
    for (int k = 0; k < Kc; ++k, b+=Nr, a+=Mr){
        int idx = 0;
        for (int i = 0; i < Mr; ++i){
            __m256d a_reg = _mm256_broadcast_sd(&a[i]); 
            for (int j = 0; j < Nr; j+=avx_doubles, ++idx){
                res[idx] = _mm256_fmadd_pd(a_reg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
        }
    }
    
    int idx = 0;
    for (int i = 0; i < Mr; ++i, c+=N){
        for (int j = 0; j < Nr; j+=avx_doubles, ++idx){
            load_inc_store_double(&c[j], res[idx]);
        }
    }
}

void matmul_packing_tiling(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    constexpr auto Mc = 180, Nc = 96, Kc = 240, Nr = 12, Mr = 4;

   /** 
    * 分配连续的 Buffer，用于存放当前 Cache Block 的数据
    **/ 
    std::vector<double> a_buffer(Mc * Kc), b_buffer(Kc * Nc);

    for (int ib = 0; ib < M; ib += Mc) {
        for (int kb = 0; kb < K; kb += Kc) {
            /** 
            * 将 A 的子块 (Mc x Kc) 复制到连续内存, 确保 Mr 维度的元素连续
            **/ 
            for (int i2 = 0; i2 < Mc; i2 += Mr) {
                for (int k = 0; k < Kc; ++k) {
                    for (int i = 0; i < Mr; ++i) {
                        a_buffer[(i2 * Kc) + (k * Mr) + i] = A(ib + i2 + i, kb + k);
                    }
                }
            }
            
            for (int jb = 0; jb < N; jb += Nc) {
                /** 
                * 将 B 的子块 (Kc x Nc) 复制到连续内存, 确保 Nr 维度的元素连续
                **/ 
                for (int j2 = 0; j2 < Nc; j2 += Nr) {
                    for (int k = 0; k < Kc; ++k) {
                        for (int j = 0; j < Nr; ++j) {
                            b_buffer[(j2 * Kc) + (k * Nr) + j] = B(kb + k, jb + j2 + j);
                        }
                    }
                }

                for (int i2 = 0; i2 < Mc; i2 += Mr) {
                    for (int j2 = 0; j2 < Nc; j2 += Nr) {
                        // 指向 Packing 后的连续内存地址
                        const double* a = &a_buffer[i2 * Kc];
                        const double* b = &b_buffer[j2 * Kc];
                        // C 矩阵通常不进行 Packing，直接更新原内存以减少开销
                        double* c = &C(ib + i2, jb + j2);
                        matmul_reg_block_reordered_kernel<Nr, Mr, Kc>(a, b, c, N);
                    }
                }
            }
        }
    }
}
```

**最终结果** 性能相比Naive实现提升了约422.2倍。

---
### 8. 使用c++26的SIMD指令

```c++
template<int Nr, int Mr, int Kc>
void matmul_reg_block_reordered_simd_kernel(const double* a, const double* b, double* c, int N)
{
    constexpr int avx_doubles = 256 / (sizeof(double) * 8);
    constexpr int CREG_CNT{Mr * Nr / avx_doubles};
    std::array<simd_d, CREG_CNT> c_reg{};
    
    for (int k = 0; k < Kc; ++k, b+=Nr, a+=Mr){
        int idx = 0;
        for (int i = 0; i < Mr; ++i){
            simd_d a_reg(a[i]); 
            for (int j = 0; j < Nr; j+=avx_doubles, ++idx){
                c_reg[idx] += a_reg * simd_d(&b[j], stdx::element_aligned);
            }
        }
    }
    
    int idx = 0;
    for (int i = 0; i < Mr; ++i, c+=N){
        for (int j = 0; j < Nr; j+=avx_doubles, ++idx){
            load_inc_store_double(&c[j], c_reg[idx]);
        }
    }
}
```

**最终结果** 零开销抽象。

---
### 9. 循环展开

彻底展开最内层循环，消除循环的分支跳转开销。展开后的长指令流使得编译器能拥有更大的指令窗口，从而进行更好的指令调度，提升指令级并行度（ILP）。为了避免手写大量展开代码丧失灵活性，利用std::make_index_sequence 与折叠表达式可以在编译期动态生成展开后的汇编代码。

```c++
/** 计算循环展开 **/
template<typename T, size_t... J>
void compute_row(const simd<T>& a, const simd<T>* b, simd<T>* r, std::index_sequence<J...>)
{
    /** 
    * 1. 这行代码在编译时会被展开为J行指令, 
    *     从根本上消除了循环计数器自增指令和条件跳转指令, 
    *     指令流是完全线性的, 指令分发器可以看到后续数十条互不相关的 FMA 指令。
    * 2. 展开后, r[0], r[1], r[2]... 对应不同的 SIMD 寄存器, 
    *     现代 CPU 的指令发射队列可以同时执行这些 FMA 指令。
    * 3. 因为 J 是编译期常量，b[J * size] 这种地址偏移在编译阶段就计算完成, 
    *     最终汇编直接体现为 [rax + 64] 这种立即数偏移形式, 
    *     节省了运行时的乘法和加法开销。
    **/ 
    (..., (r[J] += a * b[J]));
}

template<typename T, size_t... I, size_t... J>
void compute_kernel(const T* a, const T* b, simd<T>* r, std::index_sequence<I...>, std::index_sequence<J...>)
{
    constexpr auto Nrs= sizeof...(J);
    constexpr auto seq= std::make_index_sequence<Nrs>{};
    constexpr auto align= stdx::element_aligned;
    
    simd<T> bs[Nrs] = {simd<T>(&b[J * simd<T>::size()], align)...};
    (..., (compute_row(simd<T>(a[I]), bs, &r[I * Nrs], seq)));
}

template<int Mr, int Nrs, typename T>
void compute_kernel(const T* a, const T* b, simd<T>* r)
{
    compute_kernel(a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
}

/** 读写循环展开 **/
template<size_t RowIdx, typename T, int WIDTH, size_t... I>
void store_row(T* c, simd<T, WIDTH>* r, std::index_sequence<I...>)
{
    /** 
    * 编译器明确知道 r[0], r[1], r[2]... 是独立的存储位置,
    * 从而大胆进行指令重排, 将 Load 指令尽可能的提前, 
    * 这样可以掩盖内存延迟。
    **/
    (..., (load_inc_store(&c[I * WIDTH], r[RowIdx * sizeof...(I) + I])));
}

template<int Nrs, typename T, int WIDTH, size_t... RowIndices>
void store_kernel(T* c, simd<T, WIDTH>* r, int N, std::index_sequence<RowIndices...>)
{
    constexpr auto seq= std::make_index_sequence<Nrs>{};
    (..., (store_row<RowIndices>(c, r, seq), c+=N));
}

template<int Mr, int Nrs, typename T, int WIDTH>
void store_kernel(T* c, simd<T, WIDTH>* r, int N)
{
    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}

/** 循环展开后的完整kernel函数 **/
template<int Nr, int Mr, int Kc, typename T>
void matmul_unrolling_kernel(const T* a, const T* b, T* c, int N)
{
    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;
    constexpr int Nrs{Nr / num_of_elems_in_reg };
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg ");
    
    fix_simd<T, num_of_elems_in_reg> r[Nrs*Mr] = {};
    for (int k = 0; k < Kc; ++k, b+=Nr, a+=Mr){
        compute_kernel<Mr, Nrs>(a, b, r);
    }
    store_kernel<Mr, Nrs>(c, r, N);
}
```

**最终结果** 性能相比Naive实现提升了约434.28倍。

---
### 10. 预取

在代码中手动插入预取指令，提前将下一轮所需的 Cache Line 取入缓存，用来掩盖内存访问延迟。尽管硬件预取已经很强，但对于跨步距的矩阵 C 写入，显式软件预取仍带来了小幅性能提升。

```c++
__mm_prefetch(c, _MM_HINT_NTA);
```

**最终结果** 性能相比Naive实现提升了约452.4倍。
与OpenBlas相比，OpenBlas的为Naive实现的约490.3倍。









