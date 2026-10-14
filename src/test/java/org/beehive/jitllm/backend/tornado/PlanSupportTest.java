package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * The admission every GPU plan passes first, as a pure function of the family, the policy and the
 * devices. The key/value cache precision is not an input: a family that cannot run on a device is
 * refused with an FP32 cache as with an FP16 one, and in single-token decode as in either prefill.
 */
public class PlanSupportTest {

    private static final ExecutionPolicy SINGLE_TOKEN = ExecutionPolicy.builder().build();

    private static final ExecutionPolicy SEQUENTIAL =
            ExecutionPolicy.builder()
                    .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(1)
                    .build();

    private static final ExecutionPolicy BATCHED =
            ExecutionPolicy.builder()
                    .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(128)
                    .build();

    private static final Map<String, ExecutionPolicy> MODES =
            Map.of("single-token", SINGLE_TOKEN, "sequential", SEQUENTIAL, "batched", BATCHED);

    private static Optional<String> check(
            String architecture, ExecutionPolicy policy, boolean split, BackendId... devices) {
        return PlanSupport.unsupported(architecture, policy, List.of(devices), split);
    }

    @Test
    public void deepseek2RunsOnCudaInEveryMode() {
        MODES.forEach(
                (name, policy) ->
                        assertEquals(
                                name,
                                Optional.empty(),
                                check("deepseek2", policy, false, BackendId.CUDA)));
    }

    @Test
    public void deepseek2IsRefusedOffCudaInEveryMode() {
        for (BackendId backend : List.of(BackendId.OPENCL, BackendId.METAL)) {
            MODES.forEach(
                    (name, policy) -> {
                        Optional<String> reason = check("deepseek2", policy, false, backend);
                        assertTrue(name + " on " + backend, reason.isPresent());
                        assertTrue(reason.get(), reason.get().contains("CUDA"));
                    });
        }
    }

    @Test
    public void theCpuIsNeverRefused() {
        MODES.forEach(
                (name, policy) ->
                        assertEquals(
                                name,
                                Optional.empty(),
                                check("deepseek2", policy, false, BackendId.CPU)));
    }

    @Test
    public void aFamilyWithoutADeviceRestrictionRunsEverywhere() {
        for (BackendId backend : List.of(BackendId.CUDA, BackendId.OPENCL, BackendId.METAL)) {
            MODES.forEach(
                    (name, policy) ->
                            assertEquals(
                                    name + " on " + backend,
                                    Optional.empty(),
                                    check("llama", policy, false, backend)));
        }
    }

    @Test
    public void aSplitRefusesSequentialPrefill() {
        Optional<String> reason = check("llama", SEQUENTIAL, true, BackendId.CUDA);
        assertTrue(reason.isPresent());
        assertTrue(reason.get(), reason.get().contains("--devices"));
        assertTrue(reason.get(), reason.get().contains("--batch-prefill-size"));
        // Without a split the same policy runs.
        assertEquals(Optional.empty(), check("llama", SEQUENTIAL, false, BackendId.CUDA));
    }

    @Test
    public void aSplitRunsSingleTokenAndBatchedPrefill() {
        for (ExecutionPolicy policy : List.of(SINGLE_TOKEN, BATCHED)) {
            assertEquals(
                    Optional.empty(), check("llama", policy, true, BackendId.CUDA, BackendId.CUDA));
        }
    }

    @Test
    public void everyStageDeviceIsChecked() {
        assertTrue(check("deepseek2", BATCHED, true, BackendId.CUDA, BackendId.OPENCL).isPresent());
        assertEquals(
                Optional.empty(),
                check("deepseek2", BATCHED, true, BackendId.CUDA, BackendId.CUDA));
    }

    private static ExecutionPolicy batched(int width, boolean nativeLibraries) {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(width)
                .nativeLibraries(nativeLibraries)
                .build();
    }

    @Test
    public void unpackedWeightsServeEveryPolicy() {
        for (ExecutionPolicy policy :
                List.of(SINGLE_TOKEN, batched(64, true), batched(96, false))) {
            assertEquals(Optional.empty(), PlanSupport.packedUnsupported("gemma4", false, policy));
        }
    }

    @Test
    public void packedWeightsNeedATiledPrefillWidth() {
        for (String architecture : List.of("llama", "qwen35", "gemma4")) {
            for (ExecutionPolicy policy :
                    List.of(SINGLE_TOKEN, SEQUENTIAL, BATCHED, batched(256, false))) {
                assertEquals(
                        architecture,
                        Optional.empty(),
                        PlanSupport.packedUnsupported(architecture, true, policy));
            }
            Optional<String> reason =
                    PlanSupport.packedUnsupported(architecture, true, batched(64, false));
            assertTrue(architecture, reason.isPresent());
            assertTrue(reason.get(), reason.get().contains("multiple of 128"));
        }
    }

    @Test
    public void packedGemma4WeightsRefuseNativeProjections() {
        Optional<String> reason = PlanSupport.packedUnsupported("gemma4", true, batched(128, true));
        assertTrue(reason.isPresent());
        assertTrue(reason.get(), reason.get().contains("cuBLAS"));
        assertTrue(reason.get(), reason.get().contains("ModelOptions.executionPolicy"));
        // Llama's native prefill takes its int8 GEMM over packed tiles.
        assertEquals(
                Optional.empty(), PlanSupport.packedUnsupported("llama", true, batched(128, true)));
    }
}
