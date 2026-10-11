package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Whether decode attention runs the lane-cooperative kernel instead of the per-key one.
 *
 * <p>Support and preference are kept apart here for the same reason they are in {@link
 * Fp16GemvReductionPolicy}. The support half is {@link DeviceCapability#SHUFFLE_REDUCED_FP16_GEMV}:
 * the kernel reduces with {@code simdShuffleDown} and broadcasts with {@code simdBroadcastFirst},
 * so a backend that miscompiles the warp shuffle would compute a wrong answer, silently. OpenCL is
 * such a backend.
 *
 * <p>The shape half is not a preference at all but a hard precondition. There is one kernel per
 * head width: the 128-wide one assigns lane {@code L} the head dimensions {@code 2L}, {@code 2L+1},
 * {@code 64+2L}, {@code 64+2L+1}, the 96-wide one {@code 2L}, {@code 2L+1} and {@code 64+L}, the
 * 64-wide one {@code 2L} and {@code 2L+1}, each a correct partition of exactly that width across
 * exactly 32 lanes. Any other width is not slower, it is wrong, so the check is an equality and not
 * a heuristic. This class is the one place that maps a head width to its kernel and its grid.
 *
 * <h2>Scope</h2>
 *
 * <p>Reached by split-KV decode attention over an FP16 paged cache in the Llama (Q4_0, Q8_0, FP16)
 * Qwen3 (Q8_0, FP16), Qwen2 (Q8_0, FP16), Granite (Q8_0, through the scaled kernels) and Phi-3
 * (Q8_0, 96-wide heads) families. The FP32 key/value cache, the packed-half2 variant and the
 * non-split per-head kernel keep the kernels they had, as does the Qwen3.5 family (256-wide heads),
 * which has its own single-warp kernel.
 *
 * <p>Measured on one device, an RTX 5070 Ti (sm_120). The decision is a backend-level default
 * rather than a per-device measurement, and what it rests on is structural rather than incidental:
 * the kernel it replaces spends 34052 bytes of shared memory per block on a private 128-float
 * accumulator per thread, which caps it at six resident warps per SM of a possible 48, and reaches
 * every one of those floats through a bank all 32 lanes of the warp share.
 */
// @formatter:on
public final class LaneAttentionPolicy {

    // @formatter:off
    /**
     * Warps per (head, split), which is how many independent key streams a block keeps in flight.
     *
     * <p>Not a free parameter; it was screened. One warp per block was measured first and is 1.65x
     * faster than the kernel it replaces at depth zero and 2.3x <b>slower</b> at depth 2048: with
     * 16 heads and 8 splits that is 128 warps on a 70-SM device, under two warps per SM, and every
     * warp's loop carries a dependent chain of load, reduce, score, load, accumulate, with nothing
     * to hide the KV-cache latency behind. Four, eight, sixteen and thirty-two were then measured,
     * and tg128 at depth 2048 reads 286.4, 351.0, <b>391.0</b> and 367.8 tok/s. Thirty-two
     * regresses because a 512-thread block at 34 registers stops fitting three to an SM.
     *
     * <p>Sixteen warps take the grid to 2048 warps and give each block sixteen independent key
     * streams, for 8320 bytes of shared memory in the end-of-block fold against the replaced
     * kernel's 34052 bytes in its inner loop.
     */
    // @formatter:on
    public static final int WARPS_PER_GROUP = 16;

    /**
     * Warps per (head, split) for the 64-wide kernel. Half the work per key of a 128-wide head, so
     * fewer warps keep the group busy: four measured fastest of two, four and eight on an A10
     * (Llama-3.2-1B, tg128 at depths 0 and 680).
     */
    public static final int HEAD64_WARPS_PER_GROUP = 4;

    /**
     * Warps per (head, split) for the 96-wide kernel: eight measured fastest of four, eight and
     * sixteen on an A10 (Phi-3-mini, tg128 at depths 0 and 680).
     */
    public static final int HEAD96_WARPS_PER_GROUP = 8;

    /** The signature of a lane-cooperative split-KV kernel that also takes the score scale. */
    public interface ScaledKernel
            extends TornadoFunctions.Task16<
                    KernelContext,
                    FloatArray,
                    HalfFloatArray,
                    HalfFloatArray,
                    FloatArray,
                    Integer,
                    Integer,
                    Integer,
                    Integer,
                    IntArray,
                    Integer,
                    IntArray,
                    Integer,
                    Integer,
                    Integer,
                    Float> {}

    /** The signature every lane-cooperative split-KV kernel shares. */
    public interface Kernel
            extends TornadoFunctions.Task15<
                    KernelContext,
                    FloatArray,
                    HalfFloatArray,
                    HalfFloatArray,
                    FloatArray,
                    Integer,
                    Integer,
                    Integer,
                    Integer,
                    IntArray,
                    Integer,
                    IntArray,
                    Integer,
                    Integer,
                    Integer> {}

    private LaneAttentionPolicy() {}

    /**
     * Whether a lane-cooperative kernel may be used for this head width on the active device.
     *
     * @param headSize the model's head width; anything but 64, 96 or 128 is refused
     */
    public static boolean laneCooperativeAttention(int headSize) {
        return (headSize == 64 || headSize == 96 || headSize == 128)
                && TornadoDevices.target()
                        .capabilities()
                        .supports(DeviceCapability.SHUFFLE_REDUCED_FP16_GEMV);
    }

    /** The kernel for {@code headSize}, which {@link #laneCooperativeAttention} admitted. */
    public static Kernel kernel(int headSize) {
        return switch (headSize) {
            case 64 ->
                    TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVFP16PagedLaneHead64;
            case 96 ->
                    TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVFP16PagedLaneHead96;
            case 128 ->
                    TransformerPagedKvKernels
                            ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead128;
            default ->
                    throw new IllegalArgumentException(
                            "no lane-cooperative attention kernel for head width " + headSize);
        };
    }

    /**
     * Whether a lane-cooperative kernel that takes its score scale exists for this head width: for
     * a family whose attention multiplier is not {@code 1/sqrt(headSize)}.
     */
    public static boolean scaledLaneCooperativeAttention(int headSize) {
        return (headSize == 64 || headSize == 128) && laneCooperativeAttention(headSize);
    }

    /**
     * The scaled kernel for {@code headSize}, which {@link #scaledLaneCooperativeAttention}
     * admitted.
     */
    public static ScaledKernel scaledKernel(int headSize) {
        return switch (headSize) {
            case 64 ->
                    TransformerPagedKvKernels
                            ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead64Scaled;
            case 128 ->
                    TransformerPagedKvKernels
                            ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead128Scaled;
            default ->
                    throw new IllegalArgumentException(
                            "no scaled lane-cooperative attention kernel for head width "
                                    + headSize);
        };
    }

    /** The grid for {@code groups} (head, split) pairs of a {@code headSize}-wide head. */
    public static WorkerGrid worker(int groups, int headSize) {
        int warps =
                switch (headSize) {
                    case 64 -> HEAD64_WARPS_PER_GROUP;
                    case 96 -> HEAD96_WARPS_PER_GROUP;
                    default -> WARPS_PER_GROUP;
                };
        return WorkerGridFactory.createLaneAttentionWorker(groups, warps);
    }
}
