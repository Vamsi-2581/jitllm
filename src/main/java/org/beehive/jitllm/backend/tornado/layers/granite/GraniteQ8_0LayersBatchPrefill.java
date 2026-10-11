package org.beehive.jitllm.backend.tornado.layers.granite;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.GraniteBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

// @formatter:off
/**
 * Granite batched-prefill transformer-layer TaskGraphs (Q8_0 weights), on the Metal SIMD-group
 * kernels only ({@link org.beehive.jitllm.backend.tornado.BatchPrefillSupport} refuses the family
 * elsewhere). Llama's layer with Granite's multipliers: the embedding multiplier is applied to the
 * batch by the first layer, the attention multiplier replaces {@code 1 / sqrt(headSize)}, and the
 * residual multiplier scales both projections that are added to the stream. RoPE rotates
 * interleaved pairs with the frequency computed from {@code ropeTheta}.
 */
// @formatter:on
public class GraniteQ8_0LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

    private final GraniteState state;
    private final GraniteTornadoWeights weights;
    private final GraniteConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public GraniteQ8_0LayersBatchPrefill(
            GraniteState state,
            GraniteTornadoWeights weights,
            GraniteConfiguration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache);
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
            // EVERY_EXECUTION: a lease change rewrites the table, and a stale index is still valid.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            layer.task(
                    "batch_embedding_scale",
                    GraniteBatchPrefillKernels::batchedScale,
                    context,
                    state.workspace.wrapXBatch,
                    config.embeddingScale(),
                    batchSize * dim);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());

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
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);
        layer.task(
                "batch_qkv",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                dim,
                dim,
                kvDim,
                batchSize);
        layer.task(
                "batch_rope_kv",
                GraniteBatchPrefillKernels::batchedRopeWithKVCacheGranitePaged,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                config.ropeTheta(),
                kvDim,
                config.headSize(),
                layerIndex,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride,
                dim);
        addAttention(layer, layerIndex);
        layer.task(
                "batch_attn_out",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithScaledResidual,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.woLayered[layerIndex].asByteArray(),
                dim,
                dim,
                batchSize,
                config.residualScale());
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
                TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpQ8,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapHbBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                dim,
                hidDim,
                batchSize);
        layer.task(
                "batch_ffn_down",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithScaledResidual,
                context,
                state.workspace.wrapHbBatch,
                state.workspace.wrapXBatch,
                weights.w2Layered[layerIndex].asByteArray(),
                hidDim,
                dim,
                batchSize,
                config.residualScale());
        layer.persistOnDevice(
                state.workspace.wrapXBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache);
        return layer;
    }

    private void addAttention(TaskGraph layer, int layerIndex) {
        int h = config.headSize();
        if (h == 64) {
            layer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillSimdgroupKernels
                            ::batchedFlashAttentionPagedScaledHead64,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim(),
                    config.attentionScale());
        } else if (h == 96) {
            layer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillSimdgroupKernels
                            ::batchedFlashAttentionPagedScaledHead96,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim(),
                    config.attentionScale());
        } else {
            layer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillSimdgroupKernels
                            ::batchedFlashAttentionPagedScaledHead128,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim(),
                    config.attentionScale());
        }
    }

    /** Registers the batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int local = TransformerBatchPrefillSimdgroupKernels.THREADS;
        int tokenTiles = (batchSize + 63) / 64;
        WorkerGrid scaleWorker =
                WorkerGridFactory.genericWorker(((batchSize * dim + 255) / 256) * 256, 256);
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        ((dim + 2 * kvDim) / 64) * tokenTiles * local, local);
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) {
            ropeLocal--;
        }
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(
                        ((batchSize + TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE - 1)
                                        / TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE)
                                * config.numberOfHeads()
                                * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                        TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS);
        WorkerGrid dimWorker =
                WorkerGridFactory.genericWorker(((dim + 63) / 64) * tokenTiles * local, local);
        WorkerGrid hidWorker =
                WorkerGridFactory.genericWorker(((hidDim + 31) / 32) * tokenTiles * local, local);
        scheduler.addWorkerGrid("batchPrefillLayer_0.batch_embedding_scale", scaleWorker);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "batch_qkv", qkvWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            scheduler.addWorkerGrid(p + "batch_attn_out", dimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up", hidWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_down", dimWorker);
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
