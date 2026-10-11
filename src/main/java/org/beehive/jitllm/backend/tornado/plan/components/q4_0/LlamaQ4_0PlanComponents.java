package org.beehive.jitllm.backend.tornado.plan.components.q4_0;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.llama.LlamaQ4_0FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.llama.LlamaQ4_0LayersBatchPrefill;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.LlamaQ4_0FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;

/**
 * Llama's plan when its per-layer weights are retained as Q4_0.
 *
 * <p>Mixed by construction, because the file is. {@code llama-quantize ... Q4_0} writes every
 * {@code blk.*} weight as Q4_0 but leaves {@code token_embd} — which is also the output projection
 * when the two are tied — as Q6_K, which has no kernel here and is still materialized as Q8_0. So
 * the transformer layers read Q4_0 and the logits layer is the existing Q8_0 one, each reading what
 * its own tensors actually hold.
 *
 * <p><b>Single-token only.</b> Q4_0 has no prefill or batched kernels, and saying so here rather
 * than implementing {@code PrefillDecodeForwardPlanComponents} is what makes the registry refuse
 * those modes by name instead of failing on a cast.
 */
public class LlamaQ4_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final SchedulerType schedulerType;

    public LlamaQ4_0PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (LlamaConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new LlamaQ4_0FFNLayers("layers", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    /** Sequential prefill/decode has no Q4_0 layers. */
    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        throw new UnsupportedOperationException(
                "PREFILL_DECODE not yet supported for Llama + Q4_0");
    }

    /** Batched prefill: the host decodes the Q4_0 embeddings into the FP32 batch carrier. */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(state, config, lastBatchLayerId, true, true);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        throw new UnsupportedOperationException(
                "PREFILL_DECODE not yet supported for Llama + Q4_0");
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new LlamaQ4_0FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new LlamaQ4_0LayersBatchPrefill(state, weights, config, batchSize);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsQ8_0LayerDecode(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
