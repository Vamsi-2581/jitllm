package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

// @formatter:off
/**
 * Where a family's batched prefill ({@code --batch-prefill-size N}) cannot be built, whatever the
 * key/value cache precision.
 *
 * <p>Refused by name before any plan is built, rather than left to fail inside TornadoVM's compiler
 * or its sketcher halfway through the first generation. Two families have such holes today, both
 * off the tensor-core path:
 *
 * <ul>
 *   <li><b>gemma4</b>: the batched projections and attention are tensor-core kernels, with no
 *       scalar twin. Without tensor cores the plan fails at "MMA instructions only supported for
 *       the CUDA backend".
 *   <li><b>qwen35</b> on OpenCL: the scalar batched projections ({@code matrixVectorTiledBatchQ4_0}
 *       and its Q4_1/Q5_K siblings) do not compile on TornadoVM's OpenCL backend. A guard the
 *       compiler can prove true after the kernel's early return becomes a {@code LogicConstantNode}
 *       once a constant-trip loop is unrolled, and the OpenCL LIR builder has no rule for it
 *       ({@code OCLNodeLIRBuilder.emitLogicNode}). That is a compiler defect, reproduced by a
 *       ten-line kernel, and it is refused here until TornadoVM fixes it rather than worked around
 *       in kernels CUDA shares.
 * </ul>
 *
 * <p>Single-token decode and sequential prefill/decode run on both backends.
 */
// @formatter:on
public final class BatchPrefillSupport {

    private BatchPrefillSupport() {}

    /** Why this combination's batched prefill cannot be built, or empty when it can. */
    public static Optional<String> unsupported(Combination c) {
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE || BackendId.CPU.equals(c.backend())) {
            return Optional.empty();
        }
        return switch (c.architecture()) {
            case "gemma4" ->
                    c.tensorCores()
                                    || (BackendId.METAL.equals(c.backend())
                                            && c.weights()
                                                    == org.beehive.jitllm.runtime.tensor.DataType
                                                            .Q8_0)
                            ? Optional.empty()
                            : Optional.of(
                                    "the gemma4 batched prefill is written for tensor cores,"
                                            + " and for Q8_0 weights on the Metal SIMD-group"
                                            + " kernels, neither of which this combination has");
            case "llama" ->
                    c.weights() == org.beehive.jitllm.runtime.tensor.DataType.Q4_0
                                    && !BackendId.METAL.equals(c.backend())
                            ? Optional.of(
                                    "the llama Q4_0 batched prefill runs the Metal SIMD-group"
                                            + " kernels only")
                            : Optional.empty();
            case "granite" ->
                    BackendId.METAL.equals(c.backend())
                            ? Optional.empty()
                            : Optional.of(
                                    "the granite batched prefill runs the Metal SIMD-group kernels"
                                            + " only, for its scaled residual and attention");
            case "phi3" ->
                    BackendId.METAL.equals(c.backend())
                            ? Optional.empty()
                            : Optional.of(
                                    "the phi3 batched prefill runs the Metal SIMD-group kernels"
                                            + " only, for its fused QKV and gate/up weights");
            case "qwen35" ->
                    BackendId.OPENCL.equals(c.backend())
                            ? Optional.of(
                                    "the qwen35 batched prefill projections do not compile on the"
                                            + " OpenCL backend (TornadoVM OpenCL code generation"
                                            + " fails with 'logic node (LogicConstantNode)')")
                            : Optional.empty();
            default -> Optional.empty();
        };
    }

    /** Why {@code model}'s batched prefill cannot be built on the current device, if it cannot. */
    public static Optional<String> unsupportedOnCurrentDevice(Model model) {
        return unsupported(
                new Combination(
                        model.architectureId().toString(),
                        model.weights().dataType(),
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                                .backend(),
                        false,
                        TensorCoreSupport.isTensorCoreCapableBackend(),
                        Fp16KeyValueSupport.nvidiaDevice()));
    }

    /**
     * Refuses a batched prefill this configuration cannot build.
     *
     * @throws UnsupportedOperationException naming the combination, the reason, and the way out
     */
    public static void require(Model model, ExecutionPolicy policy, boolean gpu) {
        if (!gpu) {
            return;
        }
        Combination combination = Fp16KeyValueSupport.resolve(model, policy, gpu);
        unsupported(combination)
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(refusal(combination, reason));
                        });
    }

    /** The refusal: the combination, why, and that it holds for either cache. */
    static String refusal(Combination combination, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                "batched prefill is not supported for "
                        + combination
                        + ": "
                        + reason
                        + ". This holds for either key/value cache precision. Drop"
                        + " --batch-prefill-size (the single-token plan runs this model here),"
                        + " or leave prefill batching off in the execution policy");
    }

    /**
     * The families whose batched prefill runs the Metal SIMD-group kernels; see {@link
     * #defaultFor}.
     */
    static final java.util.Set<String> METAL_BATCHED_FAMILIES =
            java.util.Set.of(
                    "qwen3",
                    "llama",
                    "qwen2",
                    "deepseek-r1-distill-qwen",
                    "mistral",
                    "phi3",
                    "granite",
                    "qwen35",
                    "gemma4");

    /** The batched-prefill chunk Metal runs by default; see {@link #defaultFor}. */
    public static final int METAL_DEFAULT_PREFILL_BATCH = 256;

    // @formatter:off
    /**
     * The policy a model runs when its caller did not choose one: batched prefill on Metal for the
     * families whose batched kernels are tuned there, the given policy otherwise.
     *
     * <p>The families in {@link #METAL_BATCHED_FAMILIES}, in F16 and Q8_0, prefill an order of
     * magnitude faster in batches of {@value #METAL_DEFAULT_PREFILL_BATCH} than one token at a time
     * on an Apple GPU (M4 Pro, pp512: Qwen3-0.6B F16 151 to 4171 tok/s; Llama-3.2-1B F16 40 to
     * 2372, Q8_0 45 to 2075; Qwen3.5-0.8B Q4_0 60 to 2057; Gemma-4-E2B Q8_0 68 to 1067); greedy
     * text is identical. Q4_0 is batched by default for llama and qwen35 only. Other families keep
     * single-token prefill on Metal until their batched kernels are tuned there. Only a
     * single-token policy is changed, so an explicit choice is never overridden.
     */
    // @formatter:on
    public static ExecutionPolicy defaultFor(Model model, ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.SINGLE_TOKEN
                || !org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                        .backend()
                        .equals(BackendId.METAL)
                || !METAL_BATCHED_FAMILIES.contains(model.architectureId().toString())) {
            return policy;
        }
        var type = model.weights().dataType();
        boolean q4Batched =
                type == org.beehive.jitllm.runtime.tensor.DataType.Q4_0
                        && java.util.Set.of("llama", "qwen35")
                                .contains(model.architectureId().toString());
        if (type != org.beehive.jitllm.runtime.tensor.DataType.F16
                && type != org.beehive.jitllm.runtime.tensor.DataType.Q8_0
                && !q4Batched) {
            return policy;
        }
        if (unsupportedOnCurrentDevice(model).isPresent()) {
            return policy;
        }
        return ExecutionPolicy.from(policy)
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(METAL_DEFAULT_PREFILL_BATCH)
                .build();
    }
}
