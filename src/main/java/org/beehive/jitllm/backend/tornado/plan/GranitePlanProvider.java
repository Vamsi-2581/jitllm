package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.GraniteFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.GraniteQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/** Granite's plan components. */
public final class GranitePlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("granite");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.STANDARD_AND_BATCH;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        GraniteState typed = PlanStates.expect(GraniteState.class, state, ID);
        return weights == DataType.F16
                ? new GraniteFP16PlanComponents(typed, model)
                : new GraniteQ8_0PlanComponents(typed, model);
    }

    /** Single-token and batched-prefill plans, on the NVIDIA-class decode layers. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.singleTokenAndBatchedNvidiaLayers(c);
    }

    /**
     * The batched prefill runs the Metal SIMD-group kernels or the CUDA tensor cores, for its
     * scaled residual and attention; there is no scalar path.
     */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return BackendId.METAL.equals(c.backend()) || c.tensorCores()
                ? Optional.empty()
                : Optional.of(
                        "the granite batched prefill runs on the Metal SIMD-group kernels or the"
                                + " CUDA tensor cores, for its scaled residual and attention");
    }

    /** The batched prefill's projections through cuBLAS: Q8_0 weights decoded to FP16 first; F16 weights' output and down projections. */
    @Override
    public Set<DataType> nativeLibraryWeights() {
        return Set.of(DataType.Q8_0, DataType.F16);
    }
}
