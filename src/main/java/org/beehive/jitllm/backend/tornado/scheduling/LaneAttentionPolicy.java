package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.DeviceCapability;

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
 * <p>The shape half is not a preference at all but a hard precondition. The kernel assigns lane
 * {@code L} the head dimensions {@code 2L}, {@code 2L+1}, {@code 64+2L}, {@code 64+2L+1}, which is
 * a correct partition of exactly a 128-wide head across exactly 32 lanes. Anything else is not
 * slower, it is wrong, so the check is an equality and not a heuristic.
 *
 * <h2>Scope</h2>
 *
 * <p>Reached by the Qwen3 FP16 family's split-KV decode attention over an FP16 paged cache. Every
 * other family, every other quantisation, the FP32 key/value cache, the packed-half2 variant and
 * the non-split per-head kernel keep the kernel they had. Head widths other than 128 keep it too,
 * which is what leaves Llama-3.2-1B (64) and the Qwen3.5 family (256) exactly as they were.
 *
 * <p>Measured on one device, an RTX 5070 Ti (sm_120). The decision is a backend-level default
 * rather than a per-device measurement, and what it rests on is structural rather than incidental:
 * the kernel it replaces spends 34052 bytes of shared memory per block on a private 128-float
 * accumulator per thread, which caps it at six resident warps per SM of a possible 48, and reaches
 * every one of those floats through a bank all 32 lanes of the warp share.
 */
// @formatter:on
public final class LaneAttentionPolicy {

    /** The only head width the lane-cooperative kernel is written for. */
    public static final int SUPPORTED_HEAD_SIZE = 128;

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

    // @formatter:off
    /**
     * Warps per (query token, head) in the batched-prefill lane-cooperative kernel. Prefill
     * launches one workgroup per token and head, so there is no shortage of workgroups to hide
     * latency with, and fewer warps per workgroup keep more of them resident. Measured on an M4
     * Pro, Qwen3-0.6B F16, batch 256 (pp512 / pp2048 tok/s): 1 warp 1671 / 681, 2 1947 / 1024, 4
     * 1961 / 1166, 8 1937 / 999, 16 1822 / 937.
     */
    // @formatter:on
    public static final int PREFILL_WARPS_PER_GROUP = 4;

    private LaneAttentionPolicy() {}

    /**
     * Whether the lane-cooperative kernel may be used for this head width on the active device:
     * CUDA, and Metal, where it is the FP16 cache's split-KV decode attention.
     *
     * @param headSize the model's head width; anything but {@link #SUPPORTED_HEAD_SIZE} is refused
     */
    public static boolean laneCooperativeAttention(int headSize) {
        var capabilities = TornadoDevices.current().capabilities();
        return headSize == SUPPORTED_HEAD_SIZE
                && (capabilities.supports(DeviceCapability.SHUFFLE_REDUCED_FP16_GEMV)
                        // Metal: the FP32-cache twin already runs there on the same shuffles, and
                        // the 64-thread per-key FP16 kernel needs 34052 bytes of threadgroup
                        // memory, past Metal's 32 KB.
                        || capabilities.supports(DeviceCapability.SUBGROUP_SHUFFLE_32));
    }

    // @formatter:off
    /**
     * Whether split-KV decode attention over the FP32 cache runs the lane-cooperative kernel.
     *
     * <p>Granted where {@link DeviceCapability#SUBGROUP_SHUFFLE_32} holds (Metal), whose FP32 cache
     * otherwise runs {@code processHeadsFlashAttentionSplitKVPaged32}: one thread per key, each
     * reading a whole key row and folding a 128-wide accumulator through threadgroup memory for
     * every key. Same 128-wide precondition as {@link #laneCooperativeAttention}. CUDA's FP32 cache
     * keeps its kernel; it was not measured there.
     */
    // @formatter:on
    public static boolean laneCooperativeAttentionFp32Cache(int headSize) {
        return headSize == SUPPORTED_HEAD_SIZE
                && TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.SUBGROUP_SHUFFLE_32);
    }
}
