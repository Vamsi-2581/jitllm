package org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill;

import org.beehive.jitllm.backend.tornado.MlxPrefillSupport;
import org.beehive.jitllm.backend.tornado.kernels.MlxPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupFP16Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.mlx.Mlx;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Batched-prefill transformer-layer TaskGraphs for the Qwen3 FP16 unified batched prefill-decode
 * plan.
 *
 * <p>Mirrors {@link LlamaFP16LayersBatchPrefill} but adapts to Qwen3's GQA layout and
 * Qwen3-specific kernels (fused Q/K RMSNorm, RoPE theta = 1 000 000). Avoids any calls to {@code
 * Qwen3Configuration.headSize()}, {@code kvDim()}, or {@code kvMul()} which throw.
 */
public class Qwen3FP16LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

    static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int nHeadKv;
    private final int nEmbdHeadK;
    private final int nEmbdHeadV;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    /** Whether the projections are MLX GEMMs; see {@link MlxPrefillSupport}. */
    private final boolean mlx;

    // FP16 operands and products of the MLX GEMMs, shared by every layer (one element when unused).
    private final HalfFloatArray qkvHalf; // B × (qDim + 2·kvDim)
    private final HalfFloatArray attnHalf; // B × qDim
    private final HalfFloatArray outHalf; // B × dim
    private final HalfFloatArray gateUpHalf; // B × 2·hiddenDim
    private final HalfFloatArray hbHalf; // B × hiddenDim

    // [q|k|v] and [gate|up] stacked per layer, built once.
    private final HalfFloatArray[] wqkvCat;
    private final HalfFloatArray[] w13Cat;

    /** Whether every chunk's attention is MLX's; see {@link MlxPrefillSupport#attention}. */
    private final boolean sdpa;

    /** Positions the gathered keys and values hold: the context, plus one padded chunk. */
    private final int sdpaPositions;

    // The MLX attention's operands and output, [head][position][headDim] in FP16.
    private final HalfFloatArray sdpaQ;
    private final HalfFloatArray sdpaK;
    private final HalfFloatArray sdpaV;
    private final HalfFloatArray sdpaOut;

    public Qwen3FP16LayersBatchPrefill(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue();
        this.nEmbdHead = nEmbdHeadV;
        this.qDim = nEmbdHeadK * config.numberOfHeads();
        this.kvDim = nEmbdHeadV * nHeadKv;
        this.gqa = config.numberOfHeads() / nHeadKv;
        this.mlx = mlxProjections(state.executionPolicy());
        int rows = mlx ? batchSize : 0;
        this.qkvHalf = new HalfFloatArray(Math.max(1, rows * (qDim + 2 * kvDim)));
        this.attnHalf = new HalfFloatArray(Math.max(1, rows * qDim));
        this.outHalf = new HalfFloatArray(Math.max(1, rows * config.dim()));
        this.gateUpHalf = new HalfFloatArray(Math.max(1, rows * 2 * config.hiddenDim()));
        this.hbHalf = new HalfFloatArray(Math.max(1, rows * config.hiddenDim()));
        this.wqkvCat = new HalfFloatArray[mlx ? config.numberOfLayers() : 0];
        this.w13Cat = new HalfFloatArray[wqkvCat.length];
        for (int i = 0; i < wqkvCat.length; i++) {
            wqkvCat[i] =
                    HalfFloatArray.concat(
                            weights.wqLayered[i].asHalfFloatArray(),
                            weights.wkLayered[i].asHalfFloatArray(),
                            weights.wvLayered[i].asHalfFloatArray());
            w13Cat[i] =
                    HalfFloatArray.concat(
                            weights.w1Layered[i].asHalfFloatArray(),
                            weights.w3Layered[i].asHalfFloatArray());
        }
        this.sdpa = MlxPrefillSupport.attention(state.executionPolicy());
        int sdpaRows = sdpa ? batchSize : 0;
        this.sdpaPositions = sdpa ? config.contextLength() + batchSize : 0;
        this.sdpaQ = new HalfFloatArray(Math.max(1, sdpaRows * qDim));
        this.sdpaK = new HalfFloatArray(Math.max(1, sdpaPositions * kvDim));
        this.sdpaV = new HalfFloatArray(Math.max(1, sdpaPositions * kvDim));
        this.sdpaOut = new HalfFloatArray(Math.max(1, sdpaRows * qDim));
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

        TaskGraph layer = new TaskGraph(graphName);
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache());
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
            if (mlx) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        qkvHalf,
                        attnHalf,
                        outHalf,
                        gateUpHalf,
                        hbHalf);
            }
            if (sdpa) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, sdpaQ, sdpaK, sdpaV, sdpaOut);
            }
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache(),
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
            if (mlx) {
                layer.consumeFromDevice(pred, qkvHalf, attnHalf, outHalf, gateUpHalf, hbHalf);
            }
            if (sdpa) {
                layer.consumeFromDevice(pred, sdpaQ, sdpaK, sdpaV, sdpaOut);
            }
        }

        if (mlx) {
            addMlxLayer(layer, layerIndex);
            return layer;
        }

        // Per-layer weights: upload once
        Object[] layerWeights = {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex].asHalfFloatArray(),
            weights.wkLayered[layerIndex].asHalfFloatArray(),
            weights.wvLayered[layerIndex].asHalfFloatArray(),
            weights.woLayered[layerIndex].asHalfFloatArray(),
            weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
            weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].asHalfFloatArray(),
            weights.w2Layered[layerIndex].asHalfFloatArray(),
            weights.w3Layered[layerIndex].asHalfFloatArray()
        };
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, layerWeights);

        // ── Attention Block ────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps());

        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        layer.task(
                "batch_qkv",
                tiledQkv()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVFP16
                                : TransformerBatchPrefillKernels::batchedGemmQKVFP16)
                        : Qwen3Kernels::batchedFusedQKVMatmulFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                dim,
                qDim,
                kvDim,
                tiledQkv() ? batchSize : LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNorm,
                context,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(),
                nHeadKv,
                nEmbdHead,
                qDim,
                kvDim,
                config.rmsNormEps());

        if (sdpa) {
            addMlxAttention(layer, layerIndex, false);
        } else {
            addRopeAndAttention(layer, layerIndex);
        }

        // Output projection: n=qDim (input), d=dim (output)
        layer.task(
                "batch_attn_out",
                tiledGemm()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels
                                        ::batchedGemmFP16WithResidual
                                : TransformerBatchPrefillKernels::batchedGemmFP16WithResidual)
                        : TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                qDim,
                dim,
                tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);

        // ── FFN Block ──────────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps());

        layer.task(
                "batch_ffn_gate_up",
                tiledGemm()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels
                                        ::batchedGemmRmsNormFFNGateUpFP16
                                : TransformerBatchPrefillKernels::batchedGemmRmsNormFFNGateUpFP16)
                        : TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUp,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapHbBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                dim,
                hidDim,
                tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_ffn_down",
                tiledGemm()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels
                                        ::batchedGemmFP16WithResidual
                                : TransformerBatchPrefillKernels::batchedGemmFP16WithResidual)
                        : TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                context,
                state.workspace.wrapHbBatch,
                state.workspace.wrapXBatch,
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                hidDim,
                dim,
                tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);

        layer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());
        if (sdpa) {
            layer.persistOnDevice(sdpaQ, sdpaK, sdpaV, sdpaOut);
        }

        return layer;
    }

    /**
     * The layer with its four projections as MLX FP16 GEMMs: QKV and gate/up over weights stacked
     * on the host, then the output and down projections over the model's own. Normalization, RoPE,
     * attention and the residual connections stay JIT kernels.
     */
    private void addMlxLayer(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        int qkvDim = qDim + 2 * kvDim;
        HalfFloatArray wqkv = wqkvCat[layerIndex];
        HalfFloatArray w13 = w13Cat[layerIndex];
        HalfFloatArray wo = weights.woLayered[layerIndex].asHalfFloatArray();
        HalfFloatArray w2 = weights.w2Layered[layerIndex].asHalfFloatArray();
        Object[] layerWeights = {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
            weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            wqkv,
            wo,
            w13,
            w2
        };
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, layerWeights);

        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps());
        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);
        layer.libraryTask(
                "batch_qkv",
                Mlx::matmulTransposed,
                state.workspace.wrapXbFP16Batch,
                wqkv,
                qkvHalf,
                batchSize,
                dim,
                qkvDim);
        layer.task(
                "batch_qkv_split",
                MlxPrefillKernels::splitQkv,
                context,
                qkvHalf,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                qDim,
                kvDim,
                batchSize * qkvDim);
        layer.task(
                "batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNorm,
                context,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(),
                nHeadKv,
                nEmbdHead,
                qDim,
                kvDim,
                config.rmsNormEps());
        if (sdpa) {
            addMlxAttention(layer, layerIndex, true);
        } else {
            addRopeAndAttention(layer, layerIndex);
            layer.task(
                    "batch_attn_half",
                    MlxPrefillKernels::toHalf,
                    context,
                    state.workspace.wrapXbBatch,
                    attnHalf,
                    batchSize * qDim);
        }
        layer.libraryTask(
                "batch_attn_out",
                Mlx::matmulTransposed,
                attnHalf,
                wo,
                outHalf,
                batchSize,
                qDim,
                dim);
        layer.task(
                "batch_attn_residual",
                MlxPrefillKernels::addResidual,
                context,
                state.workspace.wrapXBatch,
                outHalf,
                1.0f,
                batchSize * dim);

        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps());
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        layer.libraryTask(
                "batch_ffn_gate_up",
                Mlx::matmulTransposed,
                state.workspace.wrapXbFP16Batch,
                w13,
                gateUpHalf,
                batchSize,
                dim,
                2 * hidDim);
        layer.task(
                "batch_ffn_swiglu",
                MlxPrefillKernels::swiglu,
                context,
                gateUpHalf,
                hbHalf,
                hidDim,
                batchSize * hidDim);
        layer.libraryTask(
                "batch_ffn_down",
                Mlx::matmulTransposed,
                hbHalf,
                w2,
                outHalf,
                batchSize,
                hidDim,
                dim);
        layer.task(
                "batch_ffn_residual",
                MlxPrefillKernels::addResidual,
                context,
                state.workspace.wrapXBatch,
                outHalf,
                1.0f,
                batchSize * dim);

        layer.persistOnDevice(
                state.workspace.wrapXBatch,
                keyCache(),
                valueCache(),
                qkvHalf,
                attnHalf,
                outHalf,
                gateUpHalf,
                hbHalf);
        if (sdpa) {
            layer.persistOnDevice(sdpaQ, sdpaK, sdpaV, sdpaOut);
        }
    }

    /**
     * RoPE with the key/value cache write, then MLX's fused causal attention of the chunk over its
     * whole context: Q packed {@code [head][token][headDim]} in FP16, and this layer's keys and
     * values of every position so far gathered from the paged cache. MLX reads the context length
     * from the start position when the task runs. Its output is unpacked into the output
     * projection's operand: FP16 for the MLX projection, FP32 {@code xb} for the JIT one.
     */
    private void addMlxAttention(TaskGraph layer, int layerIndex, boolean halfOutput) {
        if (useFp16KVCache()) {
            layer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3FP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        } else {
            layer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }
        int heads = config.numberOfHeads();
        layer.task(
                "batch_sdpa_pack_q",
                MlxPrefillKernels::packHeads,
                context,
                state.workspace.wrapQBatch,
                sdpaQ,
                heads,
                batchSize,
                nEmbdHead,
                batchSize * qDim);
        if (useFp16KVCache()) {
            layer.task(
                    "batch_sdpa_gather_kv",
                    MlxPrefillKernels::gatherKvFP16,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    sdpaK,
                    sdpaV,
                    state.workspace.wrapBlockTable,
                    layerIndex,
                    kvDim,
                    nEmbdHead,
                    sdpaPositions,
                    batchSize,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    sdpaPositions * kvDim);
        } else {
            layer.task(
                    "batch_sdpa_gather_kv",
                    MlxPrefillKernels::gatherKvFP32,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    sdpaK,
                    sdpaV,
                    state.workspace.wrapBlockTable,
                    layerIndex,
                    kvDim,
                    nEmbdHead,
                    sdpaPositions,
                    batchSize,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    sdpaPositions * kvDim);
        }
        layer.libraryTask(
                "batch_attention",
                Mlx::scaledDotProductAttentionCached,
                sdpaQ,
                sdpaK,
                sdpaV,
                sdpaOut,
                state.workspace.batchStartPosHolder,
                1,
                heads,
                nHeadKv,
                batchSize,
                sdpaPositions,
                nEmbdHead,
                (float) (1.0 / Math.sqrt(nEmbdHead)),
                true);
        if (halfOutput) {
            layer.task(
                    "batch_sdpa_unpack",
                    MlxPrefillKernels::unpackHeads,
                    context,
                    sdpaOut,
                    attnHalf,
                    heads,
                    batchSize,
                    nEmbdHead,
                    batchSize * qDim);
        } else {
            layer.task(
                    "batch_sdpa_unpack",
                    MlxPrefillKernels::unpackHeadsToFloat,
                    context,
                    sdpaOut,
                    state.workspace.wrapXbBatch,
                    heads,
                    batchSize,
                    nEmbdHead,
                    batchSize * qDim);
        }
    }

    /** RoPE with the key/value cache write, then attention over the cache into {@code xb}. */
    private void addRopeAndAttention(TaskGraph layer, int layerIndex) {
        if (useFp16KVCache()) {
            layer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3FP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        } else {
            layer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }

        // Reuses batchedFlashAttention: passes qDim as the 'dim' stride parameter.
        // Valid because qDim == dim for all standard Qwen3 models (nEmbdHeadK = dim/nHeads).
        if (useFp16KVCache()) {
            layer.task(
                    "batch_attention",
                    simdgroupAttention()
                            ? (nEmbdHead == 64
                                    ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                            ::batchedFlashAttentionKVFP16PagedHead64
                                    : nEmbdHead == 96
                                            ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                                    ::batchedFlashAttentionKVFP16PagedHead96
                                            : TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                                    ::batchedFlashAttentionKVFP16PagedHead128)
                            : TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionKVFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        } else {
            layer.task(
                    "batch_attention",
                    simdgroupAttention()
                            ? TransformerPagedKvBatchPrefillSimdgroupKernels
                                    ::batchedFlashAttentionPagedHead128
                            : laneAttention()
                                    ? TransformerPagedKvBatchPrefillKernels
                                            ::batchedFlashAttentionPagedLaneHead128
                                    : TransformerPagedKvBatchPrefillKernels
                                            ::batchedFlashAttentionPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }
    }

    // @formatter:on

    /**
     * Whether this family's projections are MLX GEMMs for a session running {@code policy}. The
     * decode layers ask the same question to learn which weights these graphs stop uploading.
     */
    public static boolean mlxProjections(ExecutionPolicy policy) {
        return MlxPrefillSupport.projections(policy);
    }

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        int qkvRows = qDim + 2 * kvDim;
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * qkvRows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (config.numberOfHeads() + nHeadKv) * nEmbdHead, nEmbdHead);

        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        int optLocal =
                laneAttention()
                        ? 32 * LaneAttentionPolicy.PREFILL_WARPS_PER_GROUP
                        : findOptimalLocalSize(nEmbdHead);
        int attnTiles =
                (batchSize + TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE - 1)
                        / TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE;
        WorkerGrid attnWorker =
                simdgroupAttention()
                        ? WorkerGridFactory.genericWorker(
                                attnTiles
                                        * config.numberOfHeads()
                                        * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                                TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS)
                        : WorkerGridFactory.genericWorker(
                                batchSize * config.numberOfHeads() * optLocal, optLocal);

        // Wo: B*dim output rows (n=qDim, d=dim)
        WorkerGrid matVecDimWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        int gemmLocal = simdgroupGemm() ? TransformerBatchPrefillSimdgroupKernels.THREADS : 256;
        int hidTile = simdgroupGemm() ? 32 : 64;
        WorkerGrid gemmDimWorker =
                WorkerGridFactory.genericWorker(
                        ((dim + 63) / 64) * ((batchSize + 63) / 64) * gemmLocal, gemmLocal);
        WorkerGrid gemmHidWorker =
                WorkerGridFactory.genericWorker(
                        ((hidDim + hidTile - 1) / hidTile) * ((batchSize + 63) / 64) * gemmLocal,
                        gemmLocal);
        WorkerGrid gemmQkvWorker =
                WorkerGridFactory.genericWorker(
                        ((qDim + 2 * kvDim) / 64) * ((batchSize + 63) / 64) * gemmLocal, gemmLocal);
        WorkerGrid matVecHidWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            if (mlx) {
                scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
                scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
                if (sdpa) {
                    scheduler.addWorkerGrid(p + "batch_sdpa_pack_q", elementwise(batchSize * qDim));
                    scheduler.addWorkerGrid(
                            p + "batch_sdpa_gather_kv", elementwise(sdpaPositions * kvDim));
                    scheduler.addWorkerGrid(p + "batch_sdpa_unpack", elementwise(batchSize * qDim));
                } else {
                    scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
                    scheduler.addWorkerGrid(p + "batch_attn_half", elementwise(batchSize * qDim));
                }
                scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
                scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", rmsApplyWorker);
                scheduler.addWorkerGrid(p + "batch_qkv_split", elementwise(batchSize * qkvRows));
                scheduler.addWorkerGrid(p + "batch_attn_residual", elementwise(batchSize * dim));
                scheduler.addWorkerGrid(p + "batch_ffn_swiglu", elementwise(batchSize * hidDim));
                scheduler.addWorkerGrid(p + "batch_ffn_residual", elementwise(batchSize * dim));
                continue;
            }
            scheduler.addWorkerGrid(p + "batch_qkv", tiledQkv() ? gemmQkvWorker : qkvWorker);
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (sdpa) {
                scheduler.addWorkerGrid(p + "batch_sdpa_pack_q", elementwise(batchSize * qDim));
                scheduler.addWorkerGrid(
                        p + "batch_sdpa_gather_kv", elementwise(sdpaPositions * kvDim));
                scheduler.addWorkerGrid(p + "batch_sdpa_unpack", elementwise(batchSize * qDim));
            } else {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
            scheduler.addWorkerGrid(
                    p + "batch_attn_out", tiledGemm() ? gemmDimWorker : matVecDimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(
                    p + "batch_ffn_gate_up", tiledGemm() ? gemmHidWorker : matVecHidWorker);
            scheduler.addWorkerGrid(
                    p + "batch_ffn_down", tiledGemm() ? gemmDimWorker : matVecDimWorker);
        }
    }

    /** One thread per element, rounded up to whole 256-thread groups. */
    private static WorkerGrid elementwise(int total) {
        return WorkerGridFactory.genericWorker((total + 255) / 256 * 256, 256);
    }

    private static int findOptimalLocalSize(int size) {
        int optimal = Math.min(size, 64);
        if (size % optimal != 0) {
            for (int s = 64; s >= 1; s--) {
                if (size % s == 0) {
                    optimal = s;
                    break;
                }
            }
        }
        return optimal;
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

    /**
     * Whether the cache is half precision: the representation the state holds, which the decode
     * layers after this prefill bind as well.
     */
    private boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache();
    }

    /** The key cache every graph of this family binds: FP16 when the state holds one. */
    private Object keyCache() {
        return useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** The value cache, following {@link #keyCache()}. */
    private Object valueCache() {
        return useFp16KVCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    /** The lane-cooperative kernel over the FP32 cache; see {@link LaneAttentionPolicy}. */
    private boolean laneAttention() {
        return !useFp16KVCache()
                && LaneAttentionPolicy.laneCooperativeAttentionFp32Cache(nEmbdHead);
    }

    /**
     * Attention on SIMD-group matrices; see {@link BatchPrefillGemmPolicy}. Over the FP32 cache
     * where the lane-cooperative kernel would otherwise run; over the FP16 cache for every head
     * width the FP16 SIMD-group kernels are written for.
     */
    private boolean simdgroupAttention() {
        return useFp16KVCache()
                ? BatchPrefillGemmPolicy.simdgroupAttention(nEmbdHead)
                : laneAttention() && BatchPrefillGemmPolicy.simdgroupAttention();
    }

    /** The tiled QKV kernel needs every 64-row tile to lie in one of Q, K and V. */
    private boolean tiledQkv() {
        return tiledGemm() && qDim % 64 == 0 && kvDim % 64 == 0;
    }

    /** The tiled projection kernels; see {@link BatchPrefillGemmPolicy}. */
    private static boolean tiledGemm() {
        return BatchPrefillGemmPolicy.tiled();
    }

    /**
     * The tiled projections on SIMD-group matrices; see {@link BatchPrefillGemmPolicy}. Their
     * double-buffered k-loop needs every contraction length to be a multiple of 64.
     */
    private boolean simdgroupGemm() {
        return BatchPrefillGemmPolicy.simdgroup()
                && config.dim() % 64 == 0
                && config.hiddenDim() % 64 == 0
                && qDim % 64 == 0;
    }
}
