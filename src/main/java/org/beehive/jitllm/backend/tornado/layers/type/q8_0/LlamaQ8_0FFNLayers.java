package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LlamaQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, LlamaConfiguration> {

    public LlamaQ8_0FFNLayers(
            String taskGraphName,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        this(taskGraphName, state, weights, config, schedulerType, 0, config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     * The first layer of the range uploads the shared buffers that layer 0 uploads otherwise, and
     * the key/value cache of {@code state} holds just these layers.
     */
    public LlamaQ8_0FFNLayers(
            String taskGraphName,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType,
            int firstLayer,
            int endLayer) {
        super(taskGraphName, state, weights, config, schedulerType);
        restrictToLayers(firstLayer, endLayer);
        setupFFNLayers();
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (LlamaQ8FFNLayers)
     *
     * <p>══════════════════════════════════════════════════════════════════════════════ ATTENTION
     * BLOCK ══════════════════════════════════════════════════════════════════════════════
     *
     * <p>wrapX (FP32) │ ▼ ┌─────────────────┐ │ attn_rms_reduce │──▶ temp (partial sums)
     * └────────┬────────┘ │ ▼ (optional: NON_NVIDIA only) ┌──────────────────┐ │
     * attn_rms_finalize│──▶ temp (final scale) └────────┬─────────┘ │ ▼ ┌────────────────┐ │
     * attn_rms_apply │──▶ wrapXb (normalized, FP32) └───────┬────────┘ │ ▼ ┌────────────────┐
     * ┌─────────────────────────────┐ │ qkv_projection │──────▶│ wrapQ, wrapK, wrapV (FP32) │
     * └───────┬────────┘ └─────────────────────────────┘ │ ▼ ┌───────────────────┐
     * ┌─────────────────────────────────────┐ │ rope_and_kv_cache │───▶│ Q,K rotated + KeyCache,
     * ValueCache │ └─────────┬─────────┘ └─────────────────────────────────────┘ │ ▼ ┌───────────┐
     * │ attention │──▶ wrapXb (attention output) └─────┬─────┘ │ ▼ ┌──────────────────┐ │
     * attn_output_proj │──▶ wrapX += Wo · wrapXb (residual connection) └────────┬─────────┘ │
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ ▼
     * ┌────────────────┐ │ ffn_rms_reduce │──▶ tempFFN (partial sums) └───────┬────────┘ │ ▼
     * (optional: NON_NVIDIA only) ┌─────────────────┐ │ ffn_rms_finalize│──▶ tempFFN (final scale)
     * └────────┬────────┘ │ ▼ ┌─────────────────┐ │ rms_ffn_gate_up │──▶ wrapHb =
     * SiLU(RMSNorm(x)·W1) ⊙ (RMSNorm(x)·W3) └────────┬────────┘ (fully fused: RMS reduce/apply +
     * W1/W3 matmuls + SiLU + GLU) │ ▼ ┌──────────────┐ │ ffn_down_proj│──▶ wrapX += W2 · wrapHb
     * (residual connection) └──────┬───────┘ │ ▼ wrapX (FP32) ──▶ [next layer or logits]
     *
     * <p>══════════════════════════════════════════════════════════════════════════════
     *
     * <p>Task Count: 9 tasks (7 if NVIDIA, skipping rms_finalize steps)
     *
     * <p>Data Flow Summary: Input: wrapX (FP32) - hidden state from previous layer Output: wrapX
     * (FP32) - updated hidden state with residual connections
     *
     * <p>Key Fusion Points: • qkv_projection: Fused Q/K/V matmuls with Q8 dequantization (3→1
     * kernel) • rope_and_kv_cache: Fused RoPE rotation + cache write (2→1 kernel) •
     * rms_ffn_gate_up: Fully fused RMS norm + W1/W3 matmuls + SiLU + GLU (5→1 kernel)
     *
     * <p>Quantization: Q8_0 format (8-bit weights with block-wise scaling)
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        }
        Object[] layerWeights = {
            // Copy-in weights per layer for batched-layered layout (Q8 format)
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex],
            weights.wkLayered[layerIndex],
            weights.wvLayered[layerIndex],
            weights.woLayered[layerIndex],
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex],
            weights.w2Layered[layerIndex],
            weights.w3Layered[layerIndex]
        };
        String weightSrc = weightSourceGraphName(layerIndex);
        if (weightSrc != null) {
            unifiedLayer.consumeFromDevice(
                    weightSrc,
                    org.beehive.jitllm.backend.tornado.plan.PackedRepack.deviceArrays(
                            layerWeights));
        } else {
            state.workspace.packedRepack.upload(unifiedLayer, layerWeights);
        }
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
        boolean packed = PackedActivationTasks.enabled();
        if (packed) {
            Object[] buffers = PackedActivationTasks.buffers(state.workspace);
            if (layerIndex == firstLayer) {
                unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, buffers);
            } else {
                consume(unifiedLayer, predecessorGraphName(layerIndex), buffers);
            }
        }

        // === Attention Block ===
        // RMS Normalization
        unifiedLayer.task(
                "attn_rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.temp,
                state.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.temp,
                    config.dim(),
                    config.rmsNormEps());
        }

        if (packed) {
            // Packed integer dot products over the activation quantized once to Q8 blocks.
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    "attn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);
            unifiedLayer.task(
                    "qkv_projection",
                    TransformerComputeKernelsQ8_0DP4A::fusedQKVMatmulQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    config.kvDim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    "attn_rms_apply",
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);

            // QKV Projection (fused with Q8 dequantization)
            unifiedLayer.task(
                    "qkv_projection",
                    TransformerComputeKernelsLayered::fusedQKVMatmulQ8,
                    context,
                    state.workspace.wrapXb, // input (FP32)
                    state.workspace.wrapQ, // output Q
                    state.workspace.wrapK, // output K
                    state.workspace.wrapV, // output V
                    weights.wqLayered[layerIndex].asByteArray(), // Wq (Q8)
                    weights.wkLayered[layerIndex].asByteArray(), // Wk (Q8)
                    weights.wvLayered[layerIndex].asByteArray(), // Wv (Q8)
                    config.dim(), // dim
                    config.kvDim(), // kvDim
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // RoPE + KV Cache
        // Precomputed RoPE tables: the frequencies come from the model's own rope_theta (and
        // any Llama 3.1 frequency scaling) instead of a constant baked into the kernel.
        // The paged twin differs in the KV index only: the block-table walk replaces
        // layer*contextLength*kvDim + pos*kvDim.
        ropeAndKeyValueCache(unifiedLayer, layerIndex);

        // Attention
        configureAttention(unifiedLayer, layerIndex);

        // Output Projection (Wo) with residual
        if (packed) {
            // The attention output is a new activation: quantized fresh for the projection.
            PackedActivationTasks.quantize(
                    unifiedLayer, "attn_out_quantize", context, state.workspace, state.workspace.wrapXb);
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // === FFN Block ===
        // RMS Normalization
        unifiedLayer.task(
                "ffn_rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.tempFFN,
                state.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.tempFFN,
                    config.dim(),
                    config.rmsNormEps());
        }

        if (packed) {
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    "ffn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN);
            unifiedLayer.task(
                    "ffn_gate_up",
                    TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    PACKED_LOCAL);
            // SwiGLU's output, quantized fresh for the down projection.
            PackedActivationTasks.quantize(
                    unifiedLayer, "ffn_down_quantize", context, state.workspace, state.workspace.wrapHb);
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asByteArray(),
                    config.hiddenDim(),
                    config.dim(),
                    PACKED_LOCAL);
        } else {
            // Fully fused: RMS apply + Gate/Up projections + SiLU + GLU (Q8 dequantization)
            unifiedLayer.task(
                    "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fullyFusedRmsNormFFNGateUpQ8,
                    context,
                    state.workspace.wrapX, // raw input (FP32)
                    state.workspace.wrapHb, // output
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    weights.w1Layered[layerIndex].asByteArray(), // W1 (Q8)
                    weights.w3Layered[layerIndex].asByteArray(), // W3 (Q8)
                    config.dim(), // input dimension
                    config.hiddenDim(), // output dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC);

            // Down projection (W2) with residual (Q8 dequantization)
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    state.workspace.wrapHb,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asByteArray(),
                    config.hiddenDim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Keep activation X on device for next layer
        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    protected String predecessorGraphName(int layerIndex) {
        return null;
    }

    /** Lanes per output row of the packed projections: a multiple of 32. */
    private static final int PACKED_LOCAL = 64;

    /** {@code objects} consumed from {@code source}, or from the previous graph when it is null. */
    protected static void consume(TaskGraph graph, String source, Object... objects) {
        if (source != null) {
            graph.consumeFromDevice(source, objects);
        } else {
            graph.consumeFromDevice(objects);
        }
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        Object keyCache = keyCache();
        Object valueCache = valueCache();
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == firstLayer) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.positionHolder,
                    state.workspace.temp,
                    state.workspace.tempFFN); //
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, //
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2, //
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV, //
                    keyCache,
                    valueCache, //
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray()); //
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            if (attentionSplits() > 0) {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttSplit);
            }
        } else {
            // Subsequent layers: Consume data already on device from previous layer. Named when
            // the layer names its producer: a graph that consumes anything by name gets nothing
            // from an unnamed consume.
            String pred = predecessorGraphName(layerIndex);
            consume(
                    unifiedLayer,
                    pred,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2, //
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV, //
                    keyCache,
                    valueCache, //
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb, //
                    state.workspace.positionHolder //
                    ,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            consume(unifiedLayer, pred, state.workspace.wrapBlockTable);
            if (attentionSplits() > 0) {
                consume(unifiedLayer, pred, state.workspace.wrapAttSplit);
            }
        }
        return unifiedLayer;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        // === Worker Grid Definitions ===
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker =
                WorkerGridFactory.genericWorker(
                        configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused QKV: dim rows for Q + kvDim rows for K + kvDim rows for V
        int fusedQkvGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker =
                WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 512);

        WorkerGrid parallelAttentionWorker = attentionWorker();

        boolean packed = PackedActivationTasks.enabled();
        int qkvRows = config.dim() + 2 * config.kvDim();

        // === Per-Layer Grid Assignments (ordered by TaskGraph flow) ===
        for (int i = firstLayer; i < endLayer(config.numberOfLayers()); i++) {
            String l = "layer_" + i + ".";
            // --- Attention Block ---
            // RMS Normalization
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            if (packed) {
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_rms_apply", PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "qkv_projection",
                        PackedActivationTasks.projectionWorker(qkvRows, PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_out_quantize", PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_output_proj",
                        PackedActivationTasks.projectionWorker(config.dim(), PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_rms_apply", PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_gate_up",
                        PackedActivationTasks.projectionWorker(config.hiddenDim(), PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_quantize",
                        PackedActivationTasks.quantizeWorker(config.hiddenDim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_proj",
                        PackedActivationTasks.projectionWorker(config.dim(), PACKED_LOCAL));
            } else {
                tornadoForwardScheduler.addWorkerGrid(l + "attn_rms_apply", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid(l + "qkv_projection", fusedQkvWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_output_proj", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "rms_ffn_gate_up", configHiddenDimRowMajorWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_proj", configDimRowMajorGlobalWorker);
            }
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attention", parallelAttentionWorker);
            addAttentionCombineGrid(tornadoForwardScheduler, i);
            // --- FFN Block ---
            // RMS Normalization
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
        }

        return tornadoForwardScheduler;
    }

    /**
     * RoPE on Q and K, and K and V appended to the cache. Shared by every weight type of this
     * family, so a subclass cannot write one cache representation while attention reads another.
     */
    protected void ropeAndKeyValueCache(TaskGraph unifiedLayer, int layerIndex) {
        // The cache representation is independent of the weights: Q, K and V are FP32
        // activations here, so the FP16-cache kernels are the ones the FP16 weights use.
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedFP16Paged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ, // Q (in/out)
                    state.workspace.wrapK, // K (in/out)
                    state.workspace.wrapV, // V (in only)
                    state.workspace.wrapKeyCacheFP16, // Key cache (out, FP16)
                    state.workspace.wrapValueCacheFP16, // Value cache (out, FP16)
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    config.kvDim(),
                    config.headSize(),
                    keyValueLayer(layerIndex),
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedPaged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ, // Q (in/out)
                    state.workspace.wrapK, // K (in/out)
                    state.workspace.wrapV, // V (in only)
                    state.workspace.wrapKeyCache, // Key cache (out)
                    state.workspace.wrapValueCache, // Value cache (out)
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    config.kvDim(),
                    config.headSize(),
                    keyValueLayer(layerIndex),
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
    }

    /** The key cache every graph of this family binds: FP16 when the state holds one. */
    protected Object keyCache() {
        return useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** The value cache, following {@link #keyCache()}. */
    protected Object valueCache() {
        return useFp16KVCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    /**
     * Split-KV partitions per head for attention, or 0 for one workgroup per head. Asked for with
     * {@code -Djitllm.attention.splitKv=true} (and {@code .count}); taken on the NVIDIA path with
     * an FP16 cache, where the split kernels exist.
     */
    protected int attentionSplits() {
        var partitions = state.executionPolicy().splitKvPartitions();
        if (!useFp16KVCache() || schedulerType != SchedulerType.NVIDIA) {
            return 0;
        }
        if (partitions.isEmpty()) {
            // Split by default where the lane-cooperative kernel runs: one workgroup per head
            // leaves most of the device idle and walks the whole context serially.
            return laneAttention(config.headSize())
                    ? org.beehive.jitllm.inference.state.State.SPLIT_KV
                    : 0;
        }
        int splits = partitions.getAsInt();
        if (splits > org.beehive.jitllm.inference.state.State.SPLIT_KV) {
            throw new IllegalArgumentException(
                    "split-KV working partitions "
                            + splits
                            + " exceed the "
                            + org.beehive.jitllm.inference.state.State.SPLIT_KV
                            + " the attention scratch was sized for;"
                            + " raise jitllm.attention.splitKv.count, which is the capacity");
        }
        return splits;
    }

    /**
     * The worker grid of the attention task: one workgroup per head, or per head and partition when
     * attention is split.
     */
    protected WorkerGrid attentionWorker() {
        int splits = attentionSplits();
        if (splits > 0 && laneAttention(config.headSize())) {
            return LaneAttentionPolicy.worker(config.numberOfHeads() * splits, config.headSize());
        }
        return WorkerGridFactory.createAttentionWorker(
                splits > 0 ? config.numberOfHeads() * splits : config.numberOfHeads(),
                config.headSize());
    }

    /** Adds the combine pass's grid when attention is split. */
    protected void addAttentionCombineGrid(GridScheduler scheduler, int layer) {
        if (attentionSplits() > 0) {
            scheduler.addWorkerGrid(
                    "layer_" + layer + ".attention_combine",
                    WorkerGridFactory.createAttentionWorker(
                            config.numberOfHeads(), config.headSize()));
        }
    }

    /** Attention is dtype-independent — a Q4_0 sibling reuses this unchanged. */
    protected TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        int splits = attentionSplits();
        if (splits > 0) {
            // Flash-decoding: every head's positions are split across several workgroups, and a
            // combine pass merges their partial results. More workgroups than there are heads,
            // which is what keeps a GPU busy at short contexts.
            LaneAttentionPolicy.Kernel kernel =
                    laneAttention(config.headSize())
                            ? LaneAttentionPolicy.kernel(config.headSize())
                            : packedHalf2Attention
                                    ? TransformerPagedKvKernels
                                            ::processHeadsFlashAttentionSplitKVFP16PackedPaged
                                    : TransformerPagedKvKernels
                                            ::processHeadsFlashAttentionSplitKVFP16Paged;
            unifiedLayer.task(
                    "attention",
                    kernel,
                    context,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapAttSplit,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.workspace.positionHolder,
                    keyValueLayer(layerIndex),
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    splits);
            return unifiedLayer.task(
                    "attention_combine",
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    context,
                    state.workspace.wrapAttSplit,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    splits);
        }
        if (useFp16KVCache()) {
            // Flash attention over the half-precision cache, FP32 accumulation.
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.workspace.positionHolder,
                    keyValueLayer(layerIndex),
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
        if (schedulerType == SchedulerType.NVIDIA) {
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.workspace.positionHolder,
                    keyValueLayer(layerIndex),
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsParallelPaged,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    config.contextLength(),
                    state.workspace.positionHolder,
                    state.workspace.wrapAtt,
                    keyValueLayer(layerIndex),
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
    }

    // @formatter:on

    /**
     * The graph that has already uploaded this layer's weights, or {@code null} to upload them
     * here.
     *
     * <p>A weight array bound with {@code transferToDevice} in two task graphs of one execution
     * plan gets a device buffer in each, so the pool has to hold the whole model twice. See {@code
     * LlamaFP16FFNLayers.weightSourceGraphName} for the full note.
     */
    protected String weightSourceGraphName(int layerIndex) {
        return null;
    }
}
