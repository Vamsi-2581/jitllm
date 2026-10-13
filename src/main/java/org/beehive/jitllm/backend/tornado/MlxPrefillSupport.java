package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

import uk.ac.manchester.tornado.mlx.provider.MlxLibraryProvider;

import java.util.function.BooleanSupplier;

/**
 * Whether the batched prefill pass runs its projections on Apple MLX, the Metal counterpart of
 * {@link NativePrefillSupport}'s cuBLAS path.
 *
 * <p>Under native libraries ({@code --with-native-libraries}) on the Metal backend, the Qwen3 F16
 * batched-prefill projections (QKV, output, gate/up and down) become MLX FP16 GEMMs ({@code
 * Mlx::matmulTransposed}) launched by TornadoVM on its own command queue and buffers. Q, K and V,
 * and gate and up, are multiplied as one GEMM each over weights stacked on the host, which costs a
 * second copy of those projections on the device.
 *
 * <p>Needs TornadoVM's {@code tornado.mlx} module and MLX's {@code mlx.metallib} ({@code brew
 * install mlx}) on an M4-class GPU or newer. Resolved lazily: nothing here loads the Objective-C
 * runtime or the kernel library unless a Metal plan asks.
 */
public final class MlxPrefillSupport {

    private MlxPrefillSupport() {}

    /** Whether native libraries resolve to MLX for a session running {@code policy}. */
    public static boolean enabled(ExecutionPolicy policy) {
        return policy.nativeLibraries() && metal() && Probe.MLX;
    }

    /**
     * Whether the batch-prefill projections run on MLX. {@code -Djitllm.mlx.projections=false}
     * keeps the JIT projections.
     */
    public static boolean projections(ExecutionPolicy policy) {
        return enabled(policy)
                && !"false".equalsIgnoreCase(System.getProperty("jitllm.mlx.projections"));
    }

    /**
     * Whether batched-prefill attention runs as MLX's fused causal attention over the chunk's
     * context, gathered from the paged cache. {@code -Djitllm.mlx.attention=false} keeps the JIT
     * paged attention.
     */
    public static boolean attention(ExecutionPolicy policy) {
        return enabled(policy)
                && !"false".equalsIgnoreCase(System.getProperty("jitllm.mlx.attention"));
    }

    /** Whether MLX's kernel library can be reached from this process at all. */
    public static boolean available() {
        return Probe.MLX;
    }

    private static boolean metal() {
        return BackendId.METAL.equals(TornadoDevices.current().backend());
    }

    /** The library probe, in its own holder so that asking about the backend loads nothing. */
    private static final class Probe {

        private static final boolean MLX = probe(MlxLibraryProvider::isAvailable);

        private static boolean probe(BooleanSupplier available) {
            try {
                return available.getAsBoolean();
            } catch (RuntimeException | LinkageError e) {
                return false;
            }
        }
    }
}
