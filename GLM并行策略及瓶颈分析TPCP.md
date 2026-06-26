# DSA/NSA TP+CP 并行策略说明

本文按 **Megatron Sequence Parallel (SP) + Tensor Parallel (TP)** 的视角解释技术文档中的 `TP+CP`。这里的 `CP` 在长 prefill 场景中更接近 sequence/context 维度的 activation 分片；文档里的第三个 AllGather 和最后的 ReduceScatter 应按 Megatron SP 的通信语义理解。

## 1. 应用场景和适用模型

文档中的启动方式是：

```bash
--tp 8
--enable-nsa-prefill-context-parallel
--nsa-prefill-cp-mode round-robin-split
```

在当前 SGLang 中，`nsa` 是旧名，对应：

```bash
--tp-size 8
--enable-dsa-prefill-context-parallel
--dsa-prefill-cp-mode round-robin-split
```

这套策略主要面向 DSA/NSA 长上下文模型：

- `GlmMoeDsaForCausalLM` / `glm_moe_dsa`，例如 `GLM-5-FP8`、`GLM-5.1-FP8`、`GLM-5.2-FP8`
- DeepSeek V3.2 这类带 DSA/NSA sparse attention/indexer 的模型
- config 中有 `index_topk`、`indexer_types`、`index_n_heads` 等结构的模型

它不适合直接套到普通 dense attention 模型上，例如普通 Qwen、Llama、GLM-4.5-Air。原因是普通模型没有 DSA/NSA indexer，无法产生 sparse KV top-k，也没有对应的 sparse attention/cache 路径。

### TP 和 TP+CP 的使用场景

只用 TP 适合：

- prompt 长度不极端，prefill token 数不大
- 主要瓶颈是权重显存、GEMM 或普通 attention/MLP 计算
- cache hit 后每次新增 token 很少，CP/SP 的额外通信收益不足
- 希望避免 sequence activation 的 AllGather/ReduceScatter 通信

TP+DSA/NSA CP 适合：

- `Prefill 100K + 1` 这类超长上下文 prefill
- 共享长前缀、radix cache、cache hit 场景下仍有大量长上下文 cache/KV 访问
- DSA/NSA sparse attention 可以用 indexer/top-k 降低长上下文 attention 代价
- 单纯 TP 下每张卡仍要承受完整长上下文 prefill/cache 压力

在 `round-robin-split` 下，sequence/context 维度按 token index 取模分片：

```text
GPU0: token 0, 8, 16, ...
GPU1: token 1, 9, 17, ...
...
GPU7: token 7, 15, 23, ...
```

即：

```text
owner_rank(token_i) = i % cp_size
```

这个 CP 更像 Megatron SP：activation 按 sequence/token 维度切开；进入需要 full sequence 的 TP 计算前 AllGather；计算后用 ReduceScatter 求和并切回 sequence shard。

## 2. Attention 和 MLP/MoE 的权重切分

这是理解这套策略的关键。

### Attention 权重：基本复制，不按 TP 切 head

在 DSA/NSA prefill CP 路径中，SGLang 会保持：

```text
attn_tp_size = 1
attn_cp_size = tp_size
```

所以虽然命令行写的是 `--tp 8`，但在 DSA prefill attention 阶段，这 8 个 rank 主要被用作 context/sequence parallel ranks，而不是普通 attention head-wise TP ranks。

因此 attention 阶段可以理解为：

```text
Q/KV/O 投影权重：每张卡基本完整复制
activation：按 token/sequence 切分
```

每张卡拿到自己的 sequence shard：

```text
X_r = X[token_i where i % 8 = r]
```

然后本地用完整 attention 权重计算：

```text
Q_r  = X_r W_Q
KV_r = X_r W_KV
```

通过 DSA/NSA cache/KV AllGather 拿到全局 KV 视图后，每张卡只对自己的 query 做 sparse attention：

```text
A_r = SparseAttention(Q_r, KV_full)
O_r = A_r W_O
```

所以 attention 输出仍是 sequence-sharded：

```text
GPU0: O(token0, token8, ...)
GPU1: O(token1, token9, ...)
...
```

这里不是普通 TP 中的：

```text
GPU0: 一部分 heads
GPU1: 一部分 heads
...
```

而是：

```text
GPU0: 一部分 tokens 的完整 attention
GPU1: 另一部分 tokens 的完整 attention
...
```

### MLP/MoE 权重：按 TP/MoE 切分

MLP/MoE 部分仍体现 TP：

```text
gate/up: ColumnParallel
 down:  RowParallel
```

MoE expert 还可能由 fused MoE runner 或 expert-parallel 相关 backend 管理，但从通信语义看，核心是：每张卡用自己的 MLP/MoE 权重分片对 full sequence activation 计算 partial output。

因此进入 MLP/MoE 前需要 SP AllGather，把 sequence-sharded activation 恢复成 full sequence：

```text
H_full = AllGather(H_0, H_1, ..., H_7)
```

然后每张卡计算自己的 TP partial：

```text
Y_r_partial = MLP_or_MoE_r(H_full)
```

最后 ReduceScatter：

```text
Y_full = sum_r Y_r_partial
```

并直接切回：

```text
GPU0: Y_full(token0, token8, ...)
GPU1: Y_full(token1, token9, ...)
...
```

这就是 Megatron SP 中 `AllGather -> TP compute -> ReduceScatter` 的模式。

## 3. 通信算子、生产者和消费者

`prefill_100k_cptp.pdf` 中一层 prefill 的主要通信序列是：

```text
AllGather #1
AllGather #2
AllGather #3
ReduceScatter
```

设：

```text
P = 8                       # TP/CP world size
L = prefill token 数
n = L / P
r = 当前 rank
X_r = X[token_i where i % P = r]
```

在 `round-robin-split` 下，每张卡持有交错 token：

```text
GPU0: token 0, 8, 16, ...
GPU1: token 1, 9, 17, ...
...
GPU7: token 7, 15, 23, ...
```

NCCL AllGather 的原始输出通常是 rank 顺序拼接：

```text
[rank0 tokens][rank1 tokens]...[rank7 tokens]
```

SGLang 的 CP helper 会再做 rerange，把它恢复成原始 token 顺序：

```text
token0, token1, token2, ..., tokenL-1
```

所以后文的 `AllGather(...)` 都默认包含这个 round-robin rerange。

### AllGather #1：DSA/NSA indexer 全局视图

trace 中大致位置是：

```text
elementwise
fast_hadamard_transform_kernel
fast_hadamard_transform_kernel
ncclDevKernel_AllGather_RING_LL
_act_quant_kernel
fused_store_indexer_cache
deep_gemm sm100_bf16_gemm_impl
triton_poi_fused_mul_unsqueeze
_get_k_and_s_triton_kernel
deep_gemm sm100_fp8_mqa_logits
topk_transform_prefill_kernel
```

这个 AllGather 不是 SP，也不是最终 attention 的 KV AllGather。它服务于 DSA/NSA indexer，用来让每张卡得到全局 token 的 indexer/cache 视图，从而为自己的 query token 选择 sparse KV 范围。

#### 通信前谁生产

生产者是每张卡本地的 attention/indexer 前置算子。

每张卡先从自己的 sequence shard 计算 latent：

```text
qkv_latent_r = X_r W_a
q_lora_r, latent_cache_r = split(qkv_latent_r)
latent_cache_r = [kv_lora_r, k_rope_raw_r]
```

其中 `q_lora_r` 进入两条路径：

```text
1. attention query 路径：
   q_lora_r -> q_a_layernorm -> q_b_proj -> Q_r

2. DSA/NSA indexer 路径：
   X_r, q_lora_r, position_r -> indexer precompute
```

第一个 AllGather 前的 elementwise 和 `fast_hadamard_transform_kernel` 可以理解为 indexer 路径的前置变换。抽象写作：

```text
I_r = f_indexer_pre(X_r, q_lora_r, position_r)
```

这里的 `I_r` 不是模型里正式命名的单一张量，而是 indexer query/key/cache 写入前需要的中间表示。它可能包含或参与生成：

```text
indexer query feature
indexer key feature
position/rope 相关特征
后续量化和 cache 写入需要的局部状态
```

#### 通信内容

第一个 AllGather 通信的是每张卡本地生成的 indexer 前置中间表示：

```text
GPU0: I(token0, token8, token16, ...)
GPU1: I(token1, token9, token17, ...)
...
GPU7: I(token7, token15, token23, ...)
```

通信后每张卡都有：

```text
I_full = I(token0, token1, ..., tokenL-1)
```

#### 通信后谁消费

消费者是 DSA/NSA indexer 后续算子：

```text
_act_quant_kernel
fused_store_indexer_cache
deep_gemm sm100_bf16_gemm_impl
triton_poi_fused_mul_unsqueeze
_get_k_and_s_triton_kernel
deep_gemm sm100_fp8_mqa_logits
topk_transform_prefill_kernel
```

它们完成的计算可以拆成：

```text
1. _act_quant_kernel
   把 indexer query/key 或相关激活量化成 FP8。

2. fused_store_indexer_cache
   把全局 token 顺序下的 indexer key 写入 indexer cache。

3. deep_gemm sm100_bf16_gemm_impl
   计算 indexer 的 dense 投影、gate、score 前置项等。

4. triton_poi_fused_mul_unsqueeze / _get_k_and_s_triton_kernel
   生成 DSA/NSA sparse selection 需要的 key、scale、metadata。

5. deep_gemm sm100_fp8_mqa_logits
   用本 rank query 对全局 indexer cache 算 logits。

6. topk_transform_prefill_kernel
   从 logits 里得到每个 query token 要访问的 top-k KV block/token index。
```

最终产物是 sparse attention 需要的选择结果：

```text
topk_indices_r = TopK(IndexerLogits(Q_index_r, K_index_full))
```

注意消费者只需要为本 rank 的 query token 产生 `topk_indices_r`，但它必须能看到全局 indexer cache，所以需要 AllGather #1。

### AllGather #2：MLA/DSA attention 的 latent KV cache

trace 中大致位置是：

```text
topk_transform_prefill_kernel
Memcpy DtoD
GEMM / fused_rope_kernel
elementwise
ncclDevKernel_AllGather_RING_LL
elementwise
set_mla_kv_buffer_kernel
concat_mla_absorb_q_kernel
sparse_attn_fwd_kernel
```

这个 AllGather 通信的是真正给 attention 消费的 latent KV cache，而不是第一个 AllGather 的 indexer 中间表示。

#### 通信前谁生产

生产者是本地 KV latent 处理路径。

从前面的投影得到：

```text
latent_cache_r = [kv_lora_r, k_rope_raw_r]
```

本地计算：

```text
k_nope_r = RMSNorm(kv_lora_r)
k_pe_r   = RoPE(k_rope_raw_r)
```

然后打包回 latent cache：

```text
KV_latent_r = concat(k_nope_r, k_pe_r)
```

代码逻辑上对应 `rebuild_cp_kv_cache`：先把本 rank 的 `k_nope` 和 `k_pe` 写入 `latent_cache`，再调用 CP AllGather。

#### 通信内容

第二个 AllGather 通信的是 attention 用的 latent KV cache：

```text
GPU0: KV_latent(token0, token8, token16, ...)
GPU1: KV_latent(token1, token9, token17, ...)
...
GPU7: KV_latent(token7, token15, token23, ...)
```

通信后每张卡都有：

```text
KV_latent_full = KV_latent(token0, token1, ..., tokenL-1)
```

其中可以理解为：

```text
KV_latent_full = [K_nope_full, K_pe_full]
```

#### 通信后谁消费

消费者是 MLA sparse attention 前后的算子：

```text
set_mla_kv_buffer_kernel
concat_mla_absorb_q_kernel
sparse_attn_fwd_kernel
```

具体计算是：

```text
1. set_mla_kv_buffer_kernel
   把 AllGather 后的 KV_latent_full 写入 MLA KV buffer。

2. concat_mla_absorb_q_kernel
   拼接本 rank 的 q_nope / q_pe，形成本 rank query。

3. sparse_attn_fwd_kernel
   用本 rank 的 Q_r、全局 KV_latent_full，以及 AllGather #1 后产生的 topk_indices_r 做 sparse attention。
```

因此每张卡计算的是：

```text
O_r = SparseAttention(Q_r, KV_latent_full, topk_indices_r)
```

这里的关键点是：

```text
Q_r: 只包含本 rank 的 query token
KV_latent_full: 包含全局 token 的 KV/cache
topk_indices_r: 只对应本 rank query 的 sparse selection
```

所以 attention 输出仍然是 sequence-sharded：

```text
GPU r: O(token_i where i % P = r)
```

### AllGather #3：SP activation AllGather，给 TP MLP/MoE 消费

trace 中大致位置是：

```text
sparse_attn_fwd_kernel
GEMM / o_proj
FusedAddRMSNorm
ncclDevKernel_AllGather_RING_LL
per_token_group_quant_fp8
deep_gemm sm100_fp8_gemm_1d1d_impl
moe::dev::routing::routingDeepSeek::routingMainKernel
moe::dev::routing::routingDeepSeek::routingIndicesCluster
moe::dev::activation::activationDeepSeekKernel
moe::dev::finalize::finalizeKernel
```

这个 AllGather 和前两个不同。它不是 DSA/NSA cache 通信，而是 Megatron SP 语义：把 sequence-sharded activation 聚成 full sequence activation，供 TP 切分的 MLP/MoE 权重使用。

#### 通信前谁生产

生产者是 attention 输出后的残差和 norm 路径：

```text
O_r = SparseAttention(Q_r, KV_latent_full, topk_indices_r)
H_r = RMSNorm(residual_r + O_r W_O)
```

在这套 DSA/NSA CP 路径里，attention 权重基本复制，`W_O` 也不按 8 卡 TP 切 head，所以 `H_r` 仍然只包含本 rank 的 token shard。

#### 通信内容

第三个 AllGather 通信的是 MLP/MoE 输入 activation：

```text
GPU0: H(token0, token8, token16, ...)
GPU1: H(token1, token9, token17, ...)
...
GPU7: H(token7, token15, token23, ...)
```

通信后每张卡都有：

```text
H_full = H(token0, token1, ..., tokenL-1)
```

#### 通信后谁消费

消费者是 TP 切分的 MLP/MoE。

对于普通 SwiGLU MLP，完整公式是：

```text
U = H_full W_up
G = H_full W_gate
A = SiLU(G) * U
Y = A W_down
```

TP=8 时，权重按 intermediate 维度切：

```text
W_up_i   = W_up[:,   i*D/P : (i+1)*D/P]
W_gate_i = W_gate[:, i*D/P : (i+1)*D/P]
W_down_i = W_down[i*D/P : (i+1)*D/P, :]
```

第 `i` 张卡消费 `H_full`，计算自己的 partial output：

```text
U_i = H_full W_up_i              # [L, D/P]
G_i = H_full W_gate_i            # [L, D/P]
A_i = SiLU(G_i) * U_i            # [L, D/P]
Y_i_partial = A_i W_down_i       # [L, H]
```

对 MoE 层，`H_full` 还会先被 router 消费：

```text
router_logits = H_full W_router
expert_ids, expert_weights = TopK(router_logits)
```

trace 中对应：

```text
moe::dev::routing::routingDeepSeek::routingMainKernel
moe::dev::routing::routingDeepSeek::routingIndicesCluster
```

它们做：

```text
1. 对每个 token 选择 top-k expert
2. 生成 token -> expert 的映射
3. 把 token 按 expert 聚类/重排
4. 生成 grouped GEMM 所需 metadata
```

随后 expert FFN 计算：

```text
U_e_i = X_e W_up_e_i
G_e_i = X_e W_gate_e_i
A_e_i = SiLU(G_e_i) * U_e_i
Y_e_i_partial = A_e_i W_down_e_i
```

其中 `X_e` 是被路由到 expert `e` 的 token。trace 中的：

```text
deep_gemm sm100_fp8_gemm_1d1d_impl
moe::dev::activation::activationDeepSeekKernel
moe::dev::finalize::finalizeKernel
```

分别对应 expert GEMM、SwiGLU 激活、expert 输出按 router weight 加权并还原 token 顺序：

```text
Y_i_partial[token] = sum_{e in topk(token)} alpha[token,e] * Expert_e_i(H_full[token])
```

这里的 `_i` 表示第 `i` 个 TP rank 的 partial contribution。它还不是最终 MLP/MoE 输出。

### ReduceScatter：TP partial reduce + SP scatter

trace 中大致位置是：

```text
moe::dev::finalize::finalizeKernel
ncclDevKernel_ReduceScatter_Sum_bf16
```

#### 通信前谁生产

生产者是每张 TP rank 上的 MLP/MoE partial output：

```text
GPU0: Y_0_partial(token0..tokenL-1)
GPU1: Y_1_partial(token0..tokenL-1)
...
GPU7: Y_7_partial(token0..tokenL-1)
```

每个 `Y_i_partial` 形状都是 full sequence：

```text
[L, H]
```

但它只是完整 MLP/MoE 输出的一部分。以普通 MLP 为例：

```text
Y_full = A W_down
       = A_0 W_down_0 + A_1 W_down_1 + ... + A_7 W_down_7
       = sum_i Y_i_partial
```

MoE 也是同理：每张卡算的是自己的 expert 权重分片贡献，finalize 后仍需要跨 TP rank 求和。

#### 通信内容

ReduceScatter 通信的是所有 TP rank 的 partial output：

```text
Y_i_partial: [L, H]
```

它等价于先 AllReduce 求和，再按 sequence 维度 scatter：

```text
Y_full = sum_i Y_i_partial
Y_r = Y_full[token_j where j % P = r]
```

#### 通信后谁消费

通信后的消费者是下一层 Transformer block 的 attention/indexer 前置计算。

ReduceScatter 后每张卡重新只持有自己的 sequence shard：

```text
GPU0: Y(token0, token8, token16, ...)
GPU1: Y(token1, token9, token17, ...)
...
GPU7: Y(token7, token15, token23, ...)
```

下一层从：

```text
X_r_next = Y_r
```

继续重复：

```text
local attention/indexer precompute
AllGather #1
AllGather #2
local sparse attention
AllGather #3
TP MLP/MoE
ReduceScatter
```

## 4. 一层的完整分布式逻辑

按“生产者 -> 通信 -> 消费者”的方式，一层 DSA/NSA TP+CP prefill 可以概括为：

```text
1. 输入布局
   GPU r 持有 X_r = X[token_i where i % 8 = r]

2. 本地 attention/indexer 前置计算
   生产 Q_r、latent_cache_r、indexer precompute I_r

3. AllGather #1
   通信 I_r
   消费者：indexer quant/cache/logits/topk
   产物：topk_indices_r

4. 本地 KV latent 处理
   生产 KV_latent_r = [k_nope_r, k_pe_r]

5. AllGather #2
   通信 KV_latent_r
   消费者：set_mla_kv_buffer、concat_q、sparse_attn
   产物：O_r

6. attention 输出和 norm
   生产 H_r = RMSNorm(residual_r + O_r W_O)

7. AllGather #3
   通信 H_r
   消费者：TP MLP/MoE router、expert GEMM、activation、finalize
   产物：Y_i_partial(full sequence)

8. ReduceScatter
   通信 Y_i_partial
   做 sum_i Y_i_partial，并按 token 维度 scatter
   产物：下一层输入 X_r_next
```

一句话总结：

```text
前两个 AllGather 是 DSA/NSA 的 indexer/cache/KV 全局视图；
第三个 AllGather 是 SP activation AllGather；
ReduceScatter 是 TP partial output 求和 + SP sequence scatter。
```

## 5. Trace 中观察到的瓶颈

### 主要瓶颈

文档中的 trace 指向的主要瓶颈是：

```text
NCCL collective 等待 + GPU command queue/backpressure + rank 间到达不均衡
```

主要集中在 AllGather 以及后续 kernel launch 上。

### 实验证据

1. DecoderLayer 逐层耗时存在尖峰

大多数 layer 耗时约 `10 ms`，但不同 rank 上出现明显尖峰。部分 rank，例如 `rank1 / rank2 / rank5 / rank7`，在个别请求中前约 40 层耗时约 `5 ms`，后续进入更长耗时阶段，形成阶梯状变化。

2. `cuLaunchKernelEx` 异常变长

文档中 `rank5` 的局部 trace 显示，某处 `cuLaunchKernelEx` 达到约 `2 ms`。正常 kernel launch 通常只有几微秒到几十微秒，说明 launch 侧发生阻塞。

3. `Command Buffer Full` 与异常 launch 对齐

异常变长的 `cuLaunchKernelEx` 与 PyTorch profiler 的 `Command Buffer Full` 标记基本重合，结束点又和前序 `ncclDevKernel_AllGather_RING_LL` 对齐。这说明 GPU work queue 可能积压，后续 launch 被 backpressure 放大。

4. AllGather launch delay 存在 rank 不均衡

文档统计了 NCCL AllGather 从 CUDA API 结束到 GPU kernel 实际开始的 delay。大部分事件接近 0，但 `rank5` 和 `rank7` 在多个区间达到 `170-230 ms`，`rank1 / rank2` 也有峰值。这说明等待不是全 rank 同步发生，而是存在 rank 间不均衡。

5. 每层通信密度高

`prefill_100k_cptp.pdf` 中一层可见约 3 次 AllGather 和 1 次 ReduceScatter。模型约 78 层，一个请求会产生大量 collective。单个 rank 的 delay 高峰会映射到 layer 耗时阶段性升高。

6. CP+TP 单并发可能慢于纯 TP

缓存命中实验中，TP+CP 在单并发请求下吞吐和 TTFT 约慢 `~10%`，说明 sequence/cache 汇聚通信在小并发或 cache hit 场景下可能抵消计算收益。并发升高或 long prefill 压力变大时，TP+CP 才更容易体现收益。

### 可能原因

1. rank 间到达 collective 的时间不一致

AllGather/ReduceScatter 是 collective。一个 rank 先提交 NCCL kernel 后，也要等待其他 rank 到达同一 collective 才能推进。rank5/rank7 的长 delay 说明 CPU launch 进度或 GPU 前序 kernel 完成时间不一致。

2. GPU command queue 积压

`Command Buffer Full` 和 `cuLaunchKernelEx ~2 ms` 对齐，说明 CPU launch 被 GPU 队列容量或前序 kernel 未完成限制。大量小 kernel、GEMM、NCCL collective 交织，容易形成 backpressure。

3. 每层多次恢复 full sequence/cache 视图

前两个 AllGather 是 DSA/NSA cache/KV 必需通信，第三个 AllGather 和 ReduceScatter 是 SP+TP 通信。100K prefill 下单次通信张量大，78 层重复后通信成本显著。

4. 负载和调度不完全均衡

round-robin token 切分能平均 token 数，但仍可能受到 padding、cache hit 后剩余 token、MoE routing、不同 rank kernel 路径和 stream 调度差异影响。

5. CPU 调度或单线程 runtime 扰动

文档中 CPU idle 很高，说明不是全机 CPU 饱和；但仍可能存在单线程 scheduler、Python runtime、CUDA API launch 顺序等局部瓶颈。

### 可能解决方案

1. 动态选择是否启用 CP/SP

cache hit 后只剩少量 new token 时，CP/SP 的 AllGather/ReduceScatter 可能收益不足。可以按 unique prefill token 数动态选择纯 TP 或 TP+CP。

2. 调整 chunked prefill 粒度

实验中使用 `--chunked-prefill-size 65536`。可以测试 32768 等更小 chunk，观察 AllGather duration、launch delay、TTFT 和吞吐变化。

3. 继续拆分 NCCL 时间线

建议区分：

```text
rank 到达 CPU 时间
CUDA API enqueue 时间
NCCL GPU kernel start 时间
NCCL GPU kernel end 时间
```

如果 CPU 到达不齐，优化 scheduler/launch；如果 GPU start delay 高，优化队列积压和前序 kernel。

4. 验证 CUDA launch queue 影响

可按文档建议测试 `CUDA_SCALE_LAUNCH_QUEUES`，验证 `Command Buffer Full` 和 `cuLaunchKernelEx` 长尾是否缓解。

5. 优化通信计算重叠

尝试让 cache/KV AllGather、SP AllGather/ReduceScatter 与局部 GEMM、indexer、MoE 计算重叠，减少 collective 暴露在关键路径上的时间。

6. 降低通信张量大小

评估 FP8 KV cache、压缩中间表示、避免重复 full sequence/cache 汇聚、复用 indexer/KV 中间结果等方式。所有改动都需要验证数值一致性。

7. 多次复现实验区分稳定瓶颈和随机扰动

如果尖峰位置随机，优先排查系统扰动；如果稳定出现在相同 layer/rank，优先排查模型路径、通信顺序和 kernel 调度。

## 6. 简短结论

文档里的 `TP+CP` 更准确地说是：

```text
DSA/NSA cache/KV 全局汇聚 + Megatron SP 风格的 sequence activation 分片 + TP 权重分片
```

其中：

```text
Attention 权重：基本复制，每卡处理自己的 sequence shard
前两个 AllGather：汇聚 DSA/NSA indexer/cache/KV
第三个 AllGather：SP activation AllGather，给 TP MLP/MoE 使用
ReduceScatter：TP partial output 求和，并切回 sequence shard
```

trace 暴露出的主要问题是高频 collective 带来的等待、rank 不均衡和 GPU command queue backpressure。优化应围绕动态 CP、chunk 粒度、rank 到达同步、通信计算重叠和减少冗余 full sequence/cache 汇聚展开。
