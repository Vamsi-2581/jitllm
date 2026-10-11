package org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batched-prefill transformer-layer TaskGraphs for the unified batched prefill-decode plan ({@link
 * org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>One {@link ImmutableTaskGraph} per transformer layer, each processing {@code batchSize} tokens
 * simultaneously via {@link TransformerBatchPrefillKernels}.
 *
 * <p>Tensor-core layer pipeline (12 tasks, all GEMMs on MMA):
 *
 * <pre>
 *   batch_attn_rms        parallel RMS square-sum reduction (256 thr/token)
 *   batch_attn_rms_apply  RMS apply + FP16 quantize → wrapXbFP16Batch
 *   qkvProj               ONE fused MMA GEMM → packed qkvResultBatch [q|k|v]
 *   batch_rope_kv         RoPE over the packed buffer + KV cache write
 *   batch_attention       flash attention (register-partitioned P·V) → attnOutFP16
 *   woProj                MMA GEMM → woOut
 *   batch_ffn_rms         parallel RMS reduce FUSED with x += woOut
 *   batch_ffn_rms_apply   RMS apply + FP16 quantize → normedXFFNFP16
 *   gateUpProj            ONE fused MMA GEMM → packed gateUpResultBatch [gate|up]
 *   swiglu                SiLU(gate)*up over packed buffer → wrapHbFP16Batch
 *   w2Proj                MMA GEMM → w2Out
 *   w2Resid               x += w2Out
 * </pre>
 *
 * <p>KV cache ({@code wrapKeyCache}, {@code wrapValueCache}) is persisted on device after every
 * layer so the subsequent single-token decode layers can consume it.
 */
public class LlamaFP16LayersBatchPrefillMMA implements BatchPrefillTransformerLayerTaskGraphs {

    @Override
    public String describeProjections() {
        return "FP16 tensor-core MMA";
    }

    // Local size for the parallel RMS reductions (one workgroup per token).
    static final int RMS_LOCAL_SIZE = 256;

    protected final State state;
    private final boolean packedHalf2Attention;
    protected final TornadoWeights weights;
    protected final Configuration config;
    protected final KernelContext context = new KernelContext();
    protected final int batchSize;
    // GEMM M dimension rounded up to whole 128-row tiles (BM). The GEMM-adjacent
    // buffers in State are allocated at this padded size; rows >= batchSize are
    // computed but never consumed. All non-GEMM kernels use the true batchSize.
    protected final int paddedBatch;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    /**
     * The batched-prefill graphs only run on the CUDA backend (tensor-core gated), which is the
     * same backend the FP16 KV cache path targets.
     */
    private boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache();
    }

    public LlamaFP16LayersBatchPrefillMMA(
            State state, TornadoWeights weights, Configuration config, int batchSize) {
        this.state = state;
        // Resolved once from the session's policy, not read from a class constant.
        this.packedHalf2Attention = state.executionPolicy().packedHalf2Attention();
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        if (batchSize % 128 != 0
                && (Boolean.getBoolean("jitllm.verbose")
                        || Boolean.getBoolean("jitllm.EnableTimingForTornadoVMInit"))) {
            System.err.printf(
                    "[jitllm] prefill batch %d padded to %d for tensor-core tiles; "
                            + "GEMM efficiency is %d/%d — use a multiple of 128 for best throughput.%n",
                    batchSize, paddedBatch, batchSize, paddedBatch);
        }
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph batchPrefillLayer = new TaskGraph(graphName);

        Object keyCache =
                useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
        Object valueCache =
                useFp16KVCache()
                        ? state.workspace.wrapValueCacheFP16
                        : state.workspace.wrapValueCache;

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            // Allocate persistent GPU-side intermediates once
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache,
                    valueCache,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            // wrapXBatch produced by the prefillActivation graph and persists in device memory
            // to consume it from there we should use the explicit uniqueTaskGraph name
            // the no-arg form would use current graph name, which causes NPE without CUDA Graphs
            batchPrefillLayer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            // for the same reasons as above, we should use the explicit uniqueTaskGraph name to
            // consume
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            batchPrefillLayer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache,
                    valueCache,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
        }

        if (layerIndex == 0) {
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            batchPrefillLayer.consumeFromDevice(
                    "batchPrefillLayer_" + (layerIndex - 1), state.workspace.wrapBlockTable);
        }

        // Per-layer weights: upload once
        batchPrefillLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION, layerWeights(layerIndex));
        Object[] extraWeights = extraLayerWeights(layerIndex);
        if (extraWeights.length > 0) {
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, extraWeights);
        }

        int dim = config.dim();
        int kvDim = config.kvDim();
        int qDim = config.numberOfHeads() * config.headSize();

        atLayerStart(batchPrefillLayer, layerIndex);

        // ── Attention Block ────────────────────────────────────────────────────
        batchPrefillLayer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        batchPrefillLayer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        // Q, K, V in ONE tensor-core launch → packed [q|k|v] rows.
        // Grid spans (dim + 2*kvDim)/128 N-blocks: no grid starvation on the
        // skinny GQA projections, and the A operand is read once, not thrice.
        qkvProjection(batchPrefillLayer, layerIndex);

        afterQkvProjection(batchPrefillLayer, layerIndex);

        if (useFp16KVCache()) {
            if (splitHalfRope()) {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedFP16Paged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        ropeTheta(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            } else {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        TransformerPagedKvBatchPrefillKernels
                                ::batchedRopeWithKVCachePackedFP16Paged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        weights.freq_cis_realFlat.asFloatArray(),
                        weights.freq_cis_imagFlat.asFloatArray(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            }

            batchPrefillLayer.task(
                    "batch_attention",
                    packedHalf2Attention
                            ? TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionFP16OutKVFP16PackedTilePaged
                            : TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionFP16OutKVFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        } else {
            if (splitHalfRope()) {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedPaged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        ropeTheta(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            } else {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePackedPaged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        weights.freq_cis_realFlat.asFloatArray(),
                        weights.freq_cis_imagFlat.asFloatArray(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            }

            // Register-partitioned P·V accumulation + direct FP16 emission
            // (replaces batchedFlashAttention + attnCast).
            batchPrefillLayer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionFP16OutPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }

        batchPrefillLayer.task(
                "woProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.attnOutFP16,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                state.workspace.woOut,
                paddedBatch,
                dim,
                qDim);
        afterAttentionOutputProjection(batchPrefillLayer, layerIndex);

        feedForward(batchPrefillLayer, layerIndex);

        // Persist wrapXBatch for the next layer, and KV cache so the decode
        // layers can consume it via the activation graph pass-through.
        batchPrefillLayer.persistOnDevice(state.workspace.wrapXBatch, keyCache, valueCache);

        return batchPrefillLayer;
    }

    // @formatter:on

    /** The per-layer weights this layer's tasks read, uploaded once. */
    protected Object[] layerWeights(int layerIndex) {
        return new Object[] {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex].asHalfFloatArray(),
            weights.wkLayered[layerIndex].asHalfFloatArray(),
            weights.wvLayered[layerIndex].asHalfFloatArray(),
            weights.woLayered[layerIndex].asHalfFloatArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].asHalfFloatArray(),
            weights.w2Layered[layerIndex].asHalfFloatArray(),
            weights.w3Layered[layerIndex].asHalfFloatArray(),
            weights.freq_cis_realFlat.asFloatArray(),
            weights.freq_cis_imagFlat.asFloatArray()
        };
    }

    /** The QKV projection: packed {@code [q|k|v]} rows into {@code qkvResultBatch}. */
    protected void qkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int qDim = config.numberOfHeads() * config.headSize();
        int kvDim = config.kvDim();
        batchPrefillLayer.task(
                "qkvProj",
                TransformerBatchPrefillKernels::gemmMMAQKV,
                context,
                state.workspace.wrapXbFP16Batch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                state.workspace.qkvResultBatch,
                paddedBatch,
                qDim,
                kvDim,
                dim);
    }

    /**
     * The gate and up projections: packed {@code [gate|up]} rows into {@code gateUpResultBatch}.
     */
    protected void gateUpProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        batchPrefillLayer.task(
                "gateUpProj",
                TransformerBatchPrefillKernels::gemmMMAGateUp,
                context,
                state.workspace.normedXFFNFP16,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                state.workspace.gateUpResultBatch,
                paddedBatch,
                hidDim,
                dim);
    }

    /** The down projection's weights. */
    protected TornadoTensor downWeight(int layerIndex) {
        return weights.w2Layered[layerIndex];
    }

    /**
     * The feed-forward block: the attention output added to the stream, RMS norm, gate and up,
     * SwiGLU, down, and the residual add. A family with another feed-forward replaces it whole.
     */
    protected void feedForward(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        // ── FFN Block ──────────────────────────────────────────────────────────
        // x += woOut is fused into the FFN RMS reduction (drops the woResid task).
        batchPrefillLayer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        batchPrefillLayer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        // W1 and W3 in ONE tensor-core launch → packed [gate|up] rows.
        gateUpProjection(batchPrefillLayer, layerIndex);

        batchPrefillLayer
                .task(
                        "swiglu",
                        TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                        context,
                        state.workspace.wrapHbFP16Batch,
                        state.workspace.gateUpResultBatch,
                        hidDim)
                .task(
                        "w2Proj",
                        TransformerBatchPrefillKernels::gemmMMA,
                        context,
                        state.workspace.wrapHbFP16Batch,
                        downWeight(layerIndex).asHalfFloatArray(),
                        state.workspace.w2Out,
                        batchSize,
                        dim,
                        hidDim);
        afterDownProjection(batchPrefillLayer, layerIndex);
        batchPrefillLayer.task(
                "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out);
    }

    /** The grids of {@link #feedForward}'s tasks in layer {@code layerIndex}, under its prefix. */
    protected void addFeedForwardGrids(GridScheduler scheduler, String p, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        scheduler.addWorkerGrid(
                p + "batch_ffn_rms",
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE));
        scheduler.addWorkerGrid(
                p + "batch_ffn_rms_apply", WorkerGridFactory.genericWorker(batchSize * dim, 256));
        scheduler.addWorkerGrid(p + "gateUpProj", mmaGrid(paddedBatch, 2 * hidDim));
        scheduler.addWorkerGrid(p + "swiglu", elementwiseGrid(batchSize * hidDim));
        scheduler.addWorkerGrid(p + "w2Proj", mmaGrid(paddedBatch, dim));
        scheduler.addWorkerGrid(p + "w2Resid", elementwiseGrid(batchSize * dim));
    }

    /** Per-layer weights a family adds to the shared projection set, uploaded with it. */
    protected Object[] extraLayerWeights(int layerIndex) {
        return new Object[0];
    }

    /** Tasks a family runs between the fused QKV GEMM and RoPE, over the packed rows. */
    protected void afterQkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {}

    /** Tasks a family runs at the start of a layer, before the attention RMS norm. */
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {}

    /** Tasks a family runs on {@code woOut} before it is added to the stream. */
    protected void afterAttentionOutputProjection(TaskGraph batchPrefillLayer, int layerIndex) {}

    /** Tasks a family runs on {@code w2Out} before it is added to the stream. */
    protected void afterDownProjection(TaskGraph batchPrefillLayer, int layerIndex) {}

    /**
     * The grids of the family's own tasks in layer {@code layerIndex}, under its prefix {@code p}.
     */
    protected void addFamilyTaskGrids(GridScheduler scheduler, String p, int layerIndex) {}

    /**
     * Whether RoPE pairs element {@code i} with {@code i + headSize/2} (NeoX), computed from {@link
     * #ropeTheta()}, instead of Llama's adjacent pairs over the precomputed tables.
     */
    protected boolean splitHalfRope() {
        return false;
    }

    /** The RoPE base for {@link #splitHalfRope()}. */
    protected float ropeTheta() {
        throw new UnsupportedOperationException("adjacent-pair RoPE reads the precomputed tables");
    }

    // gemmMMA family: 256 threads/block (1D within block), grid over M- and N-blocks.
    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128; // BM
        int nBlocks = N / 128; // BN
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1); // groupIdx∈[0,mBlocks), groupIdy∈[0,nBlocks)
        return g;
    }

    static WorkerGrid elementwiseGrid(int n) { // n must be a multiple of 256
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int qDim = config.numberOfHeads() * config.headSize();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();

        // Parallel RMS reductions: one 256-thread workgroup per batch token
        WorkerGrid rmsWorker =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);

        // RMS apply: B*dim threads, local=256 (dim is always a multiple of 256 for LLaMA)
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        // RoPE+KV cache: B*(dim/2) threads, local=512
        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention (flash): B*nHeads workgroups × min(headSize,128) threads.
        // The kernel requires headSize <= 2*localSize.
        int attnLocal = Math.min(headSz, 128);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        // MMA grids
        WorkerGrid mmaQkvWorker = mmaGrid(paddedBatch, qDim + 2 * kvDim); // fused QKV
        WorkerGrid mmaDimWorker = mmaGrid(paddedBatch, dim); // woProj, w2Proj

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "qkvProj", mmaQkvWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            addFamilyTaskGrids(scheduler, p, i);
            scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            scheduler.addWorkerGrid(p + "woProj", mmaDimWorker);
            addFeedForwardGrids(scheduler, p, i);
        }
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }
}
