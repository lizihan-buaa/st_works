# DSA/NSA TP+CP 并行策略说明

本文重新按 **Megatron Sequence Parallel (SP) + Tensor Parallel (TP)** 的视角解释技术文档中的 `TP+CP`。这里的 `CP` 在长 prefill 场景中更接近 sequence/context 维度的 activation 分片；文档里的第三个 AllGather 和最后的 ReduceScatter 应按 Megatron SP 的通信语义理解。

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

## 3. 通信算子和张量含义

`prefill_100k_cptp.pdf` 中一层 prefill 的主要通信序列是：

```text
AllGather #1
AllGather #2
AllGather #3
ReduceScatter
```

另外 trace 中可见 `sglang::cross_device_reduce_2stage<bf16>`，它属于 TP partial output reduce 的实现细节或优化路径。

设：

```text
P = 8
L = prefill token 数
n = L / P
X_r = rank r 持有的 sequence shard，约 [n, hidden]
```

### AllGather #1：汇聚 DSA/NSA indexer/cache 数据

出现位置大致在：

```text
fast_hadamard_transform
AllGather
fused_store_indexer_cache
fp8_mqa_logits
topk_transform_prefill
```

通信前：

```text
GPU0: I(token0, token8, token16, ...)
GPU1: I(token1, token9, token17, ...)
...
GPU7: I(token7, token15, token23, ...)
```

这里 `I` 是 DSA/NSA indexer、top-k、indexer cache 相关中间数据。

通信后，每张卡获得全局 token 视图：

```text
GPU0..7: I_full = I(token0, token1, ..., tokenL-1)
```

NCCL AllGather 原始结果是按 rank 拼接的，所以 SGLang 会做 reshape/transpose，把 `[rank0][rank1]...` 恢复成原始 token 顺序。

### AllGather #2：汇聚 KV / latent KV cache

出现位置大致在：

```text
topk_transform_prefill
rope
AllGather
set_mla_kv_buffer
concat_mla_absorb_q
sparse_attn_fwd_kernel
```

通信前：

```text
GPU0: KV(token0, token8, token16, ...)
GPU1: KV(token1, token9, token17, ...)
...
GPU7: KV(token7, token15, token23, ...)
```

通信后：

```text
GPU0..7: KV_full = KV(token0, token1, ..., tokenL-1)
```

这一步的目的是让每张卡都能访问完整 KV/cache。随后每张卡只用自己的 `Q_r` attend 到全局 `KV_full`：

```text
GPU r:
  A_r = SparseAttention(Q_r, KV_full)
```

所以注意力输出仍是本 rank 的 sequence shard。

### AllGather #3：Megatron SP 风格的 activation AllGather

出现位置大致在：

```text
sparse_attn_fwd_kernel
o_proj
FusedAddRMSNorm
AllGather
per_token_group_quant
MLP / MoE routing / GEMM / finalize
```

这一步不是 DSA cache 汇聚，而是 TP+SP 主干通信。

通信前，attention 后的 hidden 是 sequence-sharded：

```text
GPU0: H(token0, token8, ...)
GPU1: H(token1, token9, ...)
...
GPU7: H(token7, token15, ...)
```

MLP/MoE 权重按 TP/MoE 分片。为了让每个 TP rank 用自己的权重分片处理完整 sequence activation，需要先 AllGather：

```text
GPU0..7: H_full = H(token0, token1, ..., tokenL-1)
```

然后各卡计算：

```text
Y_r_partial = MLP_or_MoE_r(H_full)
```

这里 `r` 是 TP rank 的 partial contribution，不应主要理解成 EP 后的专家贡献汇总。

### ReduceScatter：TP reduce + SP scatter

出现位置大致在：

```text
MoE finalize
ReduceScatter_Sum_bf16
```

通信前：

```text
GPU0: Y_0_partial(full sequence)
GPU1: Y_1_partial(full sequence)
...
GPU7: Y_7_partial(full sequence)
```

普通 TP 会 AllReduce：

```text
Y_full = sum_r Y_r_partial
```

并让每张卡都拿到完整 `Y_full`。但 SP 不希望保留完整 sequence activation，所以用 ReduceScatter 同时完成两件事：

```text
1. reduce:  对 TP partial output 求和
2. scatter: 按 sequence/token 维度切回各 rank
```

通信后：

```text
GPU0: Y_full(token0, token8, ...)
GPU1: Y_full(token1, token9, ...)
...
GPU7: Y_full(token7, token15, ...)
```

这一步把 activation 恢复成下一层需要的 CP/SP-local 布局。

## 4. 一层的完整分布式逻辑

按现在的理解，一层 DSA/NSA TP+CP prefill 可以概括为：

```text
1. 输入 activation 是 sequence-sharded：
   GPU r 持有 token r::8

2. Attention 权重基本复制：
   每张卡用完整 W_Q/W_KV/W_O 处理自己的 token shard

3. AllGather #1：
   汇聚 DSA/NSA indexer/top-k/cache 相关数据

4. AllGather #2：
   汇聚 KV / latent KV cache
   每张卡拿到 KV_full，但只计算自己的 Q_r

5. Sparse attention：
   GPU r 计算 SparseAttention(Q_r, KV_full)
   输出仍是 token r::8

6. AllGather #3：
   按 Megatron SP 逻辑，把 sequence-sharded hidden 聚成 H_full

7. TP MLP/MoE：
   每张卡用自己的 MLP/MoE 权重分片计算 Y_r_partial(H_full)

8. ReduceScatter：
   sum 所有 TP partial output，并按 sequence 维度切回 token r::8

9. 下一层继续从 sequence-sharded activation 开始
```

一句话总结：

```text
前两个 AllGather 是 DSA/NSA 为了恢复全局 cache/KV 视图；
第三个 AllGather + ReduceScatter 是 Megatron SP 风格的 TP activation 通信对。
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
