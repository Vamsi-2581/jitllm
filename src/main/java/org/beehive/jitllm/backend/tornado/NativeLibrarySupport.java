package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

/**
 * Where vendor native libraries ({@code --with-native-libraries}) have an implementation.
 *
 * <p>Experimental and off by default: the JIT kernels run every supported configuration, and they
 * are what is measured across models and sizes. On CUDA the native path replaces batched-prefill
 * projections with cuBLAS GEMMs over stacked weight copies (and the first chunk's attention with
 * cuDNN where it is usable), which costs a second copy of those projections on the device — for
 * Qwen3-8B F16, 8.6 GB beside 13.2 GB of weights. On Metal it runs the batched prefill's attention
 * and projections on Apple MLX ({@link MlxPrefill}).
 *
 * <p>Which families and weights have a path on which backend is each provider's answer ({@code
 * TornadoPlanProvider.nativeLibraryWeights}); nothing here lists them.
 *
 * <p>Asking for it anywhere it is not implemented is refused, never ignored: a request that quietly
 * runs the JIT kernels would read, in a benchmark, as a measurement of the libraries.
 */
public final class NativeLibrarySupport {

    private NativeLibrarySupport() {}

    /** Why native libraries cannot be used here, or empty when they can. */
    public static Optional<String> unsupported(Combination c) {
        if (BackendId.CPU.equals(c.backend())) {
            return Optional.of("the CPU path has no native-library implementation");
        }
        if (!BackendId.CUDA.equals(c.backend()) && !BackendId.METAL.equals(c.backend())) {
            return Optional.of("native libraries are implemented on CUDA and Metal only");
        }
        boolean implemented =
                TornadoPlanRegistry.provider(c.architecture())
                        .map(
                                provider ->
                                        provider.nativeLibraryWeights(c.backend())
                                                .contains(c.weights()))
                        .orElse(false);
        if (!implemented) {
            return Optional.of(
                    "no native-library path is implemented for "
                            + c.architecture()
                            + " / "
                            + c.weights()
                            + " on "
                            + c.backend()
                            + " yet ("
                            + TornadoPlanRegistry.nativeLibraryPaths(c.backend())
                            + " only)");
        }
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE) {
            return Optional.of(
                    "native libraries replace batched-prefill kernels only; add"
                            + " --with-prefill-decode --batch-prefill-size N");
        }
        if (BackendId.METAL.equals(c.backend())) {
            return Optional.empty();
        }
        if (!c.tensorCores()) {
            return Optional.of("the native path needs a tensor-core device");
        }
        return Optional.empty();
    }

    /**
     * Refuses a native-library request the resolved configuration does not implement.
     *
     * @throws UnsupportedOperationException naming the combination, the reason, and the way out
     */
    public static void require(Model model, ExecutionPolicy policy, boolean gpu) {
        if (!policy.nativeLibraries()) {
            return;
        }
        Combination combination = Fp16KeyValueSupport.resolve(model, policy, gpu);
        Optional<String> reason = unsupported(combination);
        if (reason.isEmpty()) {
            if (BackendId.METAL.equals(combination.backend())) {
                if (!MlxPrefill.available()) {
                    reason = Optional.of("MLX's mlx.metallib could not be loaded in this process");
                }
            } else if (!NativePrefillSupport.cublasAvailable()) {
                reason = Optional.of("cuBLAS could not be loaded in this process");
            }
        }
        reason.ifPresent(
                r -> {
                    throw new UnsupportedOperationException(refusal(combination, r));
                });
    }

    /** The refusal: the combination, why, and how to run with the JIT kernels instead. */
    static String refusal(Combination combination, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                "native libraries (--with-native-libraries, experimental) are not implemented for "
                        + combination
                        + ": "
                        + reason
                        + ". Drop --with-native-libraries (the JIT kernels run this"
                        + " configuration), or nativeLibraries(false) in the execution policy");
    }
}
