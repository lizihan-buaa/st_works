# DSA/NSA TP+CP 并行策略说明

本文总结技术文档中 `GLM-5-FP8 / GLM-5.x-FP8` 长上下文 prefill 实验使用的特殊并行策略，并结合 trace 说明通信算子、瓶颈现象和可能的优化方向。

## 1. 应用场景和适用模型

文档中的并行方式不是普通的 `TP + CP`，而是 **DSA/NSA prefill context parallel**：

```bash
--tp 8
--enable-nsa-prefill-context-parallel
--nsa-prefill-cp-mode round-robin-split
```

在当前 SGLang 版本中，`nsa` 参数是旧名，等价的新参数是：

```bash
--tp-size 8
--enable-dsa-prefill-context-parallel
--dsa-prefill-cp-mode round-robin-split
```

它主要面向以下模型：

- `GlmMoeDsaForCausalLM` / `glm_moe_dsa`，例如 `GLM-5-FP8`、`GLM-5.1-FP8`、`GLM-5.2-FP8`
- DeepSeek V3.2 这类带 DSA/NSA sparse attention/indexer 的模型
- 具备 `index_topk`、`indexer_types`、`index_n_heads` 等 sparse indexer 结构的长上下文模型

它不适合直接套到普通 dense attention 模型上，例如普通 Qwen、Llama、GLM-4.5-Air。原因是这些模型没有 DSA/NSA indexer，无法定义 sparse KV top-k，也没有对应的 sparse attention 执行路径。

### 什么时候用 TP

只用 TP 适合：

- prompt 长度中等，prefill token 不极端长
- 主要瓶颈是模型权重显存或矩阵乘计算
- 不希望引入额外 sequence 切分和跨 rank token 通信
- cache hit 后每次新增 token 很少，CP 的切分收益不足

TP 的核心是切权重、attention head、MLP hidden/channel。每张卡持有部分权重和部分 head，通信主要用于线性层 partial result 的 all-reduce。

### 什么时候用 TP+DSA/NSA CP

用文档里的 TP+CP 适合：

- `Prefill 100K + 1` 这类超长上下文 prefill
- batch/concurrency 下有大量共享长前缀
- DSA/NSA 模型可用 sparse attention 降低长上下文 attention 计算
- 单纯 TP 下每张卡仍需要处理完整长序列，prefill latency 和显存压力过高

在 `round-robin-split` 下，CP 按 token 下标取模切分：

```text
GPU0: token 0, 8, 16, ...
GPU1: token 1, 9, 17, ...
...
GPU7: token 7, 15, 23, ...
```

也就是：

```text
owner_rank(token_i) = i % cp_size
```

这让 100K token 在 8 张卡上均匀分摊，主要优化 long prefill 阶段。decode 阶段通常不是这套 CP 的主要收益场景。

## 2. 通信算子和张量切分

`prefill_100k_cptp.pdf` 中一层 prefill 的主要通信序列可以概括为：

```text
AllGather #1
AllGather #2
AllGather #3
ReduceScatter
```

另一个 trace 视图中还能看到：

```text
sglang::cross_device_reduce_2stage<bf16>
```

下面按一次 decoder layer 的数据流说明。设：

```text
P = 8
L = prefill token 数
n = L / P
d = hidden size
X_r = 当前 rank 持有的 token shard，形状约为 [n, d]
```

### AllGather #1：indexer/top-k 前恢复全局 token 视图

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

`I` 是 indexer/top-k 需要的中间特征、位置相关信息或 indexer cache 写入数据。

AllGather 后，每张卡都得到完整 token 视图：

```text
GPU0..7: I_full = I(token0, token1, ..., tokenL-1)
```

由于 NCCL AllGather 原始结果通常按 rank 拼接，SGLang 还会做 reshape/transpose，把：

```text
[rank0 tokens][rank1 tokens]...[rank7 tokens]
```

恢复成原始 token 顺序：

```text
token0, token1, token2, ...
```

这一步保证每张卡都能基于全局 token 信息生成一致的 sparse top-k。

### AllGather #2：KV cache / sparse attention 前恢复全局 KV

出现位置大致在：

```text
topk_transform_prefill
rope
AllGather
set_mla_kv_buffer
concat_mla_absorb_q
sparse_attn_fwd_kernel
```

通信前，每张卡只有本地 token 的 K/V 或 MLA latent KV：

```text
GPU0: KV(token0, token8, token16, ...)
GPU1: KV(token1, token9, token17, ...)
...
GPU7: KV(token7, token15, token23, ...)
```

AllGather 后，每张卡得到完整 KV：

```text
GPU0..7: KV_full = KV(token0, token1, ..., tokenL-1)
```

随后 `set_mla_kv_buffer` 写 KV cache，`sparse_attn_fwd_kernel` 使用前面选出的 sparse top-k 从全局 KV 中取值。

注意：query 仍是本 rank 的 local token query。因此 attention 输出仍是 token-sharded：

```text
GPU0: A(token0, token8, ...)
GPU1: A(token1, token9, ...)
...
```

### cross_device_reduce / reduce：合并 TP partial output

trace 中可见：

```text
deep_gemm / o_proj
sglang::cross_device_reduce_2stage<bf16>
FusedAddRMSNorm
```

这类 reduce 合并的是 TP 线性层的 partial output，不改变 token 分布。

通信前：

```text
GPU0: O_0_partial(local tokens)
GPU1: O_1_partial(local tokens)
...
GPU7: O_7_partial(local tokens)
```

通信后：

```text
O(local tokens) = sum_r O_r_partial(local tokens)
```

每张卡仍只持有自己的 round-robin token shard。

### AllGather #3：进入 MoE 前收集 full token batch

出现位置大致在：

```text
sparse_attn_fwd_kernel
o_proj
FusedAddRMSNorm
AllGather
per_token_group_quant
MoE routing / expert bmm / finalize
```

通信前：

```text
GPU0: H(token0, token8, ...)
GPU1: H(token1, token9, ...)
...
```

MoE routing 通常需要在 MoE group 内看到完整 token batch，因此 AllGather 后：

```text
GPU0..7: H_full = H(token0, token1, ..., tokenL-1)
```

随后执行 routing、expert dispatch、expert GEMM、activation 和 finalize。

### ReduceScatter：MoE 后求和并切回 CP-local token

出现位置大致在：

```text
MoE finalize
ReduceScatter_Sum_bf16
```

通信前，每张卡持有 full token batch 的 MoE partial result：

```text
GPU0: M_0_full
GPU1: M_1_full
...
GPU7: M_7_full
```

数学上需要先求和：

```text
M_full = sum_r M_r_full
```

但下一层仍然要求 round-robin CP-local 布局，所以直接用 ReduceScatter 同时完成：

```text
reduce:  对 MoE partial result 求和
scatter: 把 full token batch 切回各 rank 的 token shard
```

通信后：

```text
GPU0: M(token0, token8, ...)
GPU1: M(token1, token9, ...)
...
GPU7: M(token7, token15, ...)
```

这样下一层可以继续从相同的 DSA/NSA CP token 布局开始。

## 3. Trace 中观察到的瓶颈

### 主要瓶颈

文档中的 trace 指向的主要瓶颈不是单纯 GPU 算力不足，而是：

```text
NCCL collective 等待 + GPU command queue/backpressure + rank 间到达不均衡
```

具体集中在 AllGather 和后续 kernel launch 上。

### 实验证据

1. `DecoderLayer` 逐层耗时存在尖峰

大部分 layer 耗时集中在约 `10 ms`，但不同 rank 上出现若干明显尖峰。部分 rank，例如 `rank1 / rank2 / rank5 / rank7`，在个别请求中前约 40 层耗时约 `5 ms`，后续进入更长耗时阶段，形成阶梯状变化。

2. `cuLaunchKernelEx` 异常变长

文档中 `rank5` 的局部 trace 显示，某处 `cuLaunchKernelEx` CUDA API 调用达到约 `2 ms`。正常 kernel launch 通常只有几微秒到几十微秒，这说明 launch 侧发生阻塞。

3. `Command Buffer Full` 与异常 launch 对齐

异常变长的 `cuLaunchKernelEx` 与 PyTorch profiler 的 `Command Buffer Full` 标记基本重合，而且结束点和前序 `ncclDevKernel_AllGather_RING_LL` 的结束时间对齐。这表明 GPU work queue 可能已经积压，后续 launch 被 backpressure 放大。

4. AllGather launch delay 存在 rank 不均衡

文档统计了 NCCL AllGather 从 CUDA API 结束到 GPU kernel 实际开始的 delay。大部分 AllGather delay 接近 0，但 `rank5` 和 `rank7` 在多个区间持续达到 `170-230 ms`，`rank1 / rank2` 也有较大峰值。这说明等待不是全 rank 同步发生，而是存在 rank 间不均衡。

5. 每层通信密度高

`prefill_100k_cptp.pdf` 中一层可见约 `3` 次 NCCL AllGather 和 `1` 次 ReduceScatter。模型约 78 层，一个请求会产生大量 collective。文档也指出每个请求约有 `3 * 78` 个 NCCL AllGather kernel，因此单个 rank 的延迟高峰会映射到 layer 耗时阶段性升高。

6. CP+TP 单并发可能慢于纯 TP

缓存命中实验中，TP+CP 在单并发请求下吞吐和 TTFT 约慢 `~10%`，说明 CP 的 sequence 切分、AllGather 和 ReduceScatter 在小并发或 cache hit 场景下可能抵消计算收益。并发升高时，TP+CP 的收益才更明显。

### 可能原因

1. rank 间到达 collective 的时间不一致

AllGather 是 collective。即使某个 rank 先提交了 NCCL kernel，也必须等其他 rank 到达对应 collective 才能真正推进。trace 中 rank5/rank7 的长 launch delay 表明不同 rank 的 CPU launch 进度或 GPU 前序 kernel 完成时间不一致。

2. GPU command queue 积压

`Command Buffer Full` 和 `cuLaunchKernelEx ~2 ms` 对齐，说明 CPU 侧 launch 可能被 GPU 队列容量或前序 kernel 未完成限制。短时间内提交大量小 kernel、NCCL collective 和 GEMM，容易形成 backpressure。

3. 通信频率高，AllGather 在每层重复出现

DSA/NSA CP 为了在 indexer、KV cache、MoE 阶段恢复 full token 视图，每层需要多次 AllGather。100K prefill 下单次通信张量大，78 层重复后通信开销显著。

4. CP/TP 负载不完全均衡

round-robin token 切分能缓解连续段长度差异，但仍可能存在 padding、cache hit 后剩余 token、expert routing 分布、不同 rank kernel 路径差异等问题。MoE routing 也可能导致专家负载不均。

5. CPU 调度或单线程调度路径扰动

文档中 CPU idle 很高，说明不是全机 CPU 饱和；但仍可能存在单线程 scheduler、Python runtime、CUDA API launch 顺序等局部瓶颈。一次性尖峰也可能来自运行时 CPU 扰动。

### 可能解决方案

1. 降低无效 CP 使用

对于 cache hit 后只剩少量 new token 的请求，CP 可能收益不足。可以根据 unique prefill token 数动态决定是否启用 CP，短输入或高度 cache hit 的场景优先走纯 TP。

2. 调整 chunked prefill 粒度

文档实验使用 `--chunked-prefill-size 65536`。可以测试更小或更均匀的 chunk，例如 32768，观察 AllGather duration、launch delay 和 TTFT 是否下降。目标是在计算并行度和通信同步成本之间折中。

3. 减少 rank 间不同步

通过 NVTX/trace 继续区分：

```text
rank 到达 CPU 时间
CUDA API enqueue 时间
NCCL kernel GPU start 时间
NCCL kernel GPU end 时间
```

如果主要是 CPU 到达不齐，优化 scheduler/launch 路径；如果主要是 GPU start delay，优化队列积压和前序 kernel。

4. 验证 CUDA launch queue 影响

文档建议尝试 `CUDA_SCALE_LAUNCH_QUEUES`，验证 kernel launch 队列容量是否影响 `Command Buffer Full` 和 `cuLaunchKernelEx` 长尾。

5. 优化通信和计算重叠

尝试让 AllGather 与局部 GEMM、indexer 或 MoE 计算重叠，减少 collective 暴露在关键路径上的时间。也可以检查是否存在不必要的同步、stream barrier 或 DtoD copy。

6. 优化 MoE 负载均衡

观察各 rank 的 expert routing 分布和 MoE finalize 前后的等待。如果 expert 分布不均，可考虑 expert parallel 参数、冗余 expert、负载均衡策略或调整 routing batch 组织。

7. 降低通信张量大小

可评估 FP8 KV cache、压缩中间表示、避免不必要的 full token AllGather、复用 indexer/KV 中间结果等方式。需要谨慎验证数值一致性和 kernel 支持。

8. 多次复现实验区分稳定瓶颈和随机扰动

文档中部分 layer 40-60 尖峰可能是一次性 CPU/runtime 扰动。应重复运行同一 benchmark，比较尖峰是否稳定出现在相同 layer/rank。如果位置随机，优先排查系统扰动；如果位置稳定，优先排查模型路径、通信顺序和 kernel 调度。

## 4. 简短结论

文档中的 TP+CP 是 DSA/NSA 模型专用的 long-prefill 并行策略。它用 TP 启动多卡，用 round-robin CP 分摊 100K token 的 prefill 计算，并在 indexer、KV cache、MoE 阶段通过 AllGather/ReduceScatter 在 full-token 视图和 CP-local token shard 之间切换。

这套策略在长上下文、多并发、共享前缀场景下有价值，但通信密度很高。trace 显示主要瓶颈集中在 NCCL AllGather 等 collective 的等待、rank 不均衡和 GPU command queue backpressure 上。优化方向应围绕减少不必要 CP、调整 chunk 粒度、降低 rank 间不同步、改善通信计算重叠和 MoE 负载均衡展开。
