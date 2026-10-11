package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Batched-prefill kernels specific to the Granite family. */
public final class GraniteBatchPrefillKernels {

    private GraniteBatchPrefillKernels() {}

    /** {@code x[i] *= scale} for the first {@code n} elements: Granite's embedding multiplier. */
    public static void batchedScale(KernelContext context, FloatArray x, float scale, int n) {
        int i = context.globalIdx;
        if (i < n) {
            x.set(i, x.get(i) * scale);
        }
    }

    /**
     * {@code GranitePagedKvKernels.ropeRotationWithCacheCopyPaged} for a batch: interleaved pairs
     * {@code (i, i + 1)}, the frequency computed from {@code ropeTheta} in the kernel, the rotated
     * K and the V of every active token written to the paged cache. One thread per (token, pair);
     * padding tokens past {@code batchStartPosHolder[1]} do nothing.
     */
    public static void batchedRopeWithKVCacheGranitePaged(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int headSize,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        int globalIdx = context.globalIdx;
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int i = (globalIdx % halfDim) * 2;
        if (batchIdx >= batchStartPosHolder.get(1)) {
            return;
        }
        int pos = batchStartPosHolder.get(0) + batchIdx;
        int qOffset = batchIdx * dim;
        int kOffset = batchIdx * kvDim;
        int headDim = i % headSize;
        float freq = 1.0f / TornadoMath.pow(ropeTheta, headDim / (float) headSize);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        float v0q = wrapQBatch.get(qOffset + i);
        float v1q = wrapQBatch.get(qOffset + i + 1);
        wrapQBatch.set(qOffset + i, v0q * fcr - v1q * fci);
        wrapQBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

        if (i + 1 < kvDim) {
            float v0k = wrapKBatch.get(kOffset + i);
            float v1k = wrapKBatch.get(kOffset + i + 1);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;
            wrapKBatch.set(kOffset + i, rotK0);
            wrapKBatch.set(kOffset + i + 1, rotK1);
            int cacheOff =
                    KvBlockAddress.offset(
                            blockTable,
                            batchStartPosHolder.get(2),
                            pos,
                            KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                            kvDim,
                            blockCfg,
                            blockStride);
            wrapKeyCache.set(cacheOff + i, rotK0);
            wrapKeyCache.set(cacheOff + i + 1, rotK1);
            wrapValueCache.set(cacheOff + i, wrapVBatch.get(kOffset + i));
            wrapValueCache.set(cacheOff + i + 1, wrapVBatch.get(kOffset + i + 1));
        }
    }
}
