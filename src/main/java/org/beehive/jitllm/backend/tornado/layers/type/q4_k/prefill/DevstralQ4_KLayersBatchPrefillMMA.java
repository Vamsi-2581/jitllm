package org.beehive.jitllm.backend.tornado.layers.type.q4_k.prefill;

import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * Devstral's Q4_K batched prefill on the tensor cores: {@link LlamaFP16LayersBatchPrefillMMA}, each
 * projection's Q4_K weight decoded into one FP16 scratch before its FP16 GEMM.
 *
 * <p>There is no Q4_K tensor-core GEMM; decoding a projection at a time keeps the weights in the
 * file's 4.5 bits per weight between chunks, at the cost of one scratch the size of the widest
 * projection pair (gate and up) and a decode pass per projection per chunk. Q, K and V are decoded
 * end to end, and gate and up likewise, so one plain GEMM writes the packed rows the layers read.
 */
// @formatter:on
public class DevstralQ4_KLayersBatchPrefillMMA extends LlamaFP16LayersBatchPrefillMMA {

    // Created on first use: the base builds the graphs in its constructor, before this class's
    // fields are assigned, so this field takes no initializer.
    private HalfFloatArray scratch;

    public DevstralQ4_KLayersBatchPrefillMMA(
            State state, LlamaTornadoWeights weights, DevstralConfiguration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    private int qDim() {
        return config.numberOfHeads() * config.headSize();
    }

    private HalfFloatArray scratch() {
        if (scratch == null) {
            int dim = config.dim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim();
            long widest =
                    Math.max(
                            Math.max((long) (qDim() + 2 * kvDim) * dim, 2L * hidden * dim),
                            Math.max((long) dim * qDim(), (long) dim * hidden));
            scratch = new HalfFloatArray(Math.toIntExact(widest));
        }
        return scratch;
    }

    @Override
    protected Object[] layerWeights(int layerIndex) {
        return new Object[] {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex].asByteArray(),
            weights.wkLayered[layerIndex].asByteArray(),
            weights.wvLayered[layerIndex].asByteArray(),
            weights.woLayered[layerIndex].asByteArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].asByteArray(),
            weights.w2Layered[layerIndex].asByteArray(),
            weights.w3Layered[layerIndex].asByteArray(),
            weights.freq_cis_realFlat.asFloatArray(),
            weights.freq_cis_imagFlat.asFloatArray()
        };
    }

    @Override
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {
        if (layerIndex == 0) {
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch());
        } else {
            batchPrefillLayer.consumeFromDevice("batchPrefillLayer_" + (layerIndex - 1), scratch());
        }
    }

    /**
     * Decodes {@code w} into the scratch: Q4_K, or Q6_K where the file gave a tensor more bits (a
     * Q4_K_M file holds {@code attn_v} and {@code ffn_down} as Q6_K in half its layers).
     */
    private void decode(TaskGraph layer, String task, TornadoTensor w, int destOffset, int n) {
        switch (w.dataType()) {
            case Q4_K ->
                    layer.task(
                            task,
                            TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16,
                            context,
                            w.asByteArray(),
                            scratch(),
                            destOffset,
                            n);
            case Q6_K ->
                    layer.task(
                            task,
                            TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16,
                            context,
                            w.asByteArray(),
                            scratch(),
                            destOffset,
                            n);
            default ->
                    throw new UnsupportedOperationException(
                            "the devstral batched prefill has no decoder for "
                                    + w.dataType()
                                    + " ("
                                    + task
                                    + "); the projections must be Q4_K or Q6_K");
        }
    }

    @Override
    protected void qkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        decode(batchPrefillLayer, "qDecode", weights.wqLayered[layerIndex], 0, qDim() * dim);
        decode(
                batchPrefillLayer,
                "kDecode",
                weights.wkLayered[layerIndex],
                qDim() * dim,
                kvDim * dim);
        decode(
                batchPrefillLayer,
                "vDecode",
                weights.wvLayered[layerIndex],
                (qDim() + kvDim) * dim,
                kvDim * dim);
        batchPrefillLayer.task(
                "qkvProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapXbFP16Batch,
                scratch(),
                state.workspace.qkvResultBatch,
                paddedBatch,
                qDim() + 2 * kvDim,
                dim);
    }

    @Override
    protected void attentionOutputProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        decode(batchPrefillLayer, "woDecode", weights.woLayered[layerIndex], 0, dim * qDim());
        batchPrefillLayer.task(
                "woProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.attnOutFP16,
                scratch(),
                state.workspace.woOut,
                paddedBatch,
                dim,
                qDim());
    }

    @Override
    protected void gateUpProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidden = config.hiddenDim();
        decode(batchPrefillLayer, "gateDecode", weights.w1Layered[layerIndex], 0, hidden * dim);
        decode(
                batchPrefillLayer,
                "upDecode",
                weights.w3Layered[layerIndex],
                hidden * dim,
                hidden * dim);
        batchPrefillLayer.task(
                "gateUpProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.normedXFFNFP16,
                scratch(),
                state.workspace.gateUpResultBatch,
                paddedBatch,
                2 * hidden,
                dim);
    }

    @Override
    protected void downProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidden = config.hiddenDim();
        decode(batchPrefillLayer, "downDecode", weights.w2Layered[layerIndex], 0, dim * hidden);
        batchPrefillLayer.task(
                "w2Proj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapHbFP16Batch,
                scratch(),
                state.workspace.w2Out,
                batchSize,
                dim,
                hidden);
    }

    @Override
    protected void addFamilyTaskGrids(GridScheduler scheduler, String p, int layerIndex) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidden = config.hiddenDim();
        scheduler.addWorkerGrid(p + "qDecode", elementwise(qDim() * dim));
        scheduler.addWorkerGrid(p + "kDecode", elementwise(kvDim * dim));
        scheduler.addWorkerGrid(p + "vDecode", elementwise(kvDim * dim));
        scheduler.addWorkerGrid(p + "woDecode", elementwise(dim * qDim()));
        scheduler.addWorkerGrid(p + "gateDecode", elementwise(hidden * dim));
        scheduler.addWorkerGrid(p + "upDecode", elementwise(hidden * dim));
        scheduler.addWorkerGrid(p + "downDecode", elementwise(dim * hidden));
    }

    private static uk.ac.manchester.tornado.api.WorkerGrid elementwise(int n) {
        return WorkerGridFactory.genericWorker(((n + 255) / 256) * 256, 256);
    }
}
