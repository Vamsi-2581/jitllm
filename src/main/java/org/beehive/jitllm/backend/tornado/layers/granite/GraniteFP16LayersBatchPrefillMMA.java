package org.beehive.jitllm.backend.tornado.layers.granite;

import org.beehive.jitllm.backend.tornado.kernels.GraniteBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Granite's tensor-core batched prefill: {@link LlamaFP16LayersBatchPrefillMMA} with Granite's
 * multipliers. The first layer scales the batch by the embedding multiplier; Q is scaled by {@code
 * attentionScale * sqrt(headSize)} after the QKV GEMM, which the attention kernel's {@code 1 /
 * sqrt(headSize)} turns into the attention multiplier; and both projections added to the stream are
 * scaled by the residual multiplier. RoPE reads the precomputed tables, built from Granite's own
 * base.
 */
public class GraniteFP16LayersBatchPrefillMMA extends LlamaFP16LayersBatchPrefillMMA {

    public GraniteFP16LayersBatchPrefillMMA(
            GraniteState state,
            GraniteTornadoWeights weights,
            GraniteConfiguration config,
            int batchSize) {
        super(state, weights, config, batchSize);
    }

    private GraniteConfiguration granite() {
        return (GraniteConfiguration) config;
    }

    @Override
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {
        if (layerIndex == 0) {
            batchPrefillLayer.task(
                    "batch_embedding_scale",
                    GraniteBatchPrefillKernels::batchedScale,
                    context,
                    state.workspace.wrapXBatch,
                    granite().embeddingScale(),
                    batchSize * config.dim());
        }
    }

    @Override
    protected void afterQkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "batch_q_scale",
                GraniteBatchPrefillKernels::batchedPackedScaleQ,
                context,
                state.workspace.qkvResultBatch,
                granite().attentionScale() * (float) Math.sqrt(config.headSize()),
                config.dim(),
                config.kvDim(),
                batchSize);
    }

    @Override
    protected void afterAttentionOutputProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "batch_wo_scale",
                GraniteBatchPrefillKernels::batchedScale,
                context,
                state.workspace.woOut,
                granite().residualScale(),
                batchSize * config.dim());
    }

    @Override
    protected void afterDownProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "batch_w2_scale",
                GraniteBatchPrefillKernels::batchedScale,
                context,
                state.workspace.w2Out,
                granite().residualScale(),
                batchSize * config.dim());
    }

    @Override
    protected void addFamilyTaskGrids(GridScheduler scheduler, String p, int layerIndex) {
        var dimWorker = WorkerGridFactory.genericWorker(roundUp(batchSize * config.dim()), 256);
        if (layerIndex == 0) {
            scheduler.addWorkerGrid(p + "batch_embedding_scale", dimWorker);
        }
        scheduler.addWorkerGrid(p + "batch_q_scale", dimWorker);
        scheduler.addWorkerGrid(p + "batch_wo_scale", dimWorker);
        scheduler.addWorkerGrid(p + "batch_w2_scale", dimWorker);
    }

    private static int roundUp(int n) {
        return ((n + 255) / 256) * 256;
    }
}
