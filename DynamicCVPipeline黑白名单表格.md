# DynamicCVPipeline 黑名单 / 白名单

> 基线：`origin/main` @ `c569abdac`（2026-11-24）

## 黑名单（命中 → 中断 / 回退 / 降级）

| 名单 | 干什么用 | 涉及算子 |
|---|---|---|
| `interceptrFunc` (AnalyzeDataFlow/AnalyzeName.cpp) | 命中 → 整线在 AnalyzeDataFlow 阶段**中断回退** | 8 个：`kernel_sdpa_bwd_kv`、`_sparse_decode_kernel`、`_sparse_decode_model1_kernel`、`fused_chunk_fwd_kernel`、`parallel_nsa_fwd_kernel`、`fused_chunk_bwd_kernel`、`parallel_nsa_bwd_kernel_dkv`、`flash_varlen_fwd_kernel` |
| `kBlacklistOpNames` (PreCheckAvailable/PreCheckBlacklist.cpp) | 见到 `scope.scope` 算子 → 模块已被昇腾专用 pass 优化过，**跳过 SSBUFFER 整线** | `scope.scope`（唯一按算子名而非函数名匹配的） |
| `kBlacklistFuncNames` (PreCheckAvailable/PreCheckDisablePreload.cpp) | **禁用 3-preload**，强制双缓冲 | 3 个：`_attn_bwd`、`_swa_paged_decode_kernel`、`paged_decode_fd_kernel` |
| `kDisableMergeCubeKernel` (ComputeBlockOpt/MergeCubeBlock.cpp) | **跳过 MergeCubeBlockPass** | 7 个：`_attn_bwd`、`kernel_sdpa_bwd_q`、`_sdpa_infer_kernel`、`pcb06_tc02_c2v2v2c_chain`、`flex_attention_backward_dq_kernel`、`parallel_deltaformer_bwd_kernel_qk`、`chunk_kda_bwd_kernel_intra` |
| `legacyKernel` (AddControlFlowCondition.cpp:354) | **不用新版控制流方案**，保留 upstream legacy 条件 | 2 个：`_sdpa_infer_kernel`、`fwd_kernel` |
| `selectDisabledFunc` (AllocMultiCache/AddMultiBufferInnerScope.cpp) | producer 生成走 **legacy chunk_o 路径** | `chunk_fwd_kernel_o_npu_kernel` |
| `isOutlineReuseBlacklistKernel` (SplitDataflow/Utils.cpp) | 原本跳过 flag id 复用分析，**现已停用**（`return false` 短路、名单被注释） | 空 |

## 白名单（命中 → 启用优化）

| 名单 | 干什么用 | 涉及算子 |
|---|---|---|
| `kEnableMergeComputeBlockKernels` (ComputeBlockOpt/MergeComputeBlockPass.cpp) | MergeComputeBlockPass 默认关，命中才跑 | 5 个：`flex_attention_backward_dkdv_kernel`、`flex_attention_backward_dkdv_kernel_tasklist`、`_swa_bwd_dkdv_kernel`、`kernel_sdpa_bwd_kv`、`kernel_da_bwd_kv_ul` |
| `hasMemDepSyncWhitelistKernel` (SplitDataflow/InterCoreTransferAndSync.cpp) | 启用增强 memdep 同步 | 2 个：`_hstu_attn_fwd`、`parallel_path_fwd_kernel` |
| `legacyFuncNames` (Common/Utils.cpp) | `isTensorComputeOp` 改走 legacy 判定 | `pcb10_tc01_kernel` |
