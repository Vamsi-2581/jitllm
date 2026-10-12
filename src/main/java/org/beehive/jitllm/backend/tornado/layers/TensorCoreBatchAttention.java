package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.kernels.BatchMmaKernels;
import org.beehive.jitllm.backend.tornado.kernels.TensorCoreAttentionKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.State;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid3D;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * A batched prefill's attention as FP16 tensor-core GEMMs over the paged FP16 cache, for the layer
 * graphs of the tensor-core prefill paths.
 *
 * <p>Per layer: the layer's cache up to the chunk's last position gathered per key/value head
 * ({@code keys[g][t][d]}, {@code valuesT[g][d][t]}), the queries in FP16, then per pass of a few
 * query heads the scores ({@code Q . K^T}), the causal softmax, and the weighted values ({@code P .
 * V}), each head against its key/value head; finally the output in FP16 in the {@code attnOutFP16}
 * layout the output projection reads. Positions past the chunk's last one are neither scored nor
 * read.
 *
 * <p>The GEMM tiles columns 128 at a time. A head 128 wide (or a multiple) writes its values
 * straight into the output; a narrower one, 64 wide, is computed 128 wide into a scratch with a
 * 128-column slot per head and narrowed on the way to FP16. The extra columns read the next
 * key/value head's rows, so the transposed values carry one head of zero padding.
 */
// @formatter:on
public final class TensorCoreBatchAttention {

    /** The query heads per pass whose FP32 scores fit in this many bytes. */
    private static final long SCORES_BUDGET = 64L << 20;

    private static final int TILE = 128;

    private final State state;
    private final int qDim;
    private final int kvDim;
    private final int headSize;
    private final int heads;
    private final int kvMul;
    private final int rows;
    private final int paddedPositions;
    private final int headsPerPass;
    private final int wide;
    private final HalfFloatArray queries;
    private final HalfFloatArray keys;
    private final HalfFloatArray valuesT;
    private final FloatArray scores;
    private final HalfFloatArray probs;
    private final FloatArray attn;

    /**
     * Whether a layer of this shape takes the tensor-core attention: an FP16 cache, a padded batch
     * of whole 128-row tiles, a head a multiple of 64 wide, and tensor cores.
     */
    public static boolean applies(boolean fp16Cache, int paddedRows, int headSize) {
        return fp16Cache
                && paddedRows % TILE == 0
                && headSize % 64 == 0
                && TensorCoreSupport.isTensorCoreCapableBackend();
    }

    /**
     * @param rows the padded batch: the GEMMs' row count, a multiple of 128
     * @param contextLength the positions the cache holds
     */
    public TensorCoreBatchAttention(
            State state,
            int qDim,
            int kvDim,
            int headSize,
            int heads,
            int kvMul,
            int rows,
            int contextLength) {
        this.state = state;
        this.qDim = qDim;
        this.kvDim = kvDim;
        this.headSize = headSize;
        this.heads = heads;
        this.kvMul = kvMul;
        this.rows = rows;
        this.paddedPositions = (contextLength + TILE - 1) / TILE * TILE;
        this.headsPerPass = headsPerPass(heads, rows, paddedPositions);
        this.wide = (headSize + TILE - 1) / TILE * TILE;
        int scoreRows = rows * headsPerPass * paddedPositions;
        this.queries = TornadoWorkspaces.halfFloats(rows * qDim);
        this.keys = TornadoWorkspaces.halfFloats(kvDim * paddedPositions);
        int valuesPad = wide == headSize ? 0 : (wide - headSize) * paddedPositions;
        this.valuesT = TornadoWorkspaces.halfFloats(kvDim * paddedPositions + valuesPad);
        this.scores = TornadoWorkspaces.floats(scoreRows);
        this.probs = TornadoWorkspaces.halfFloats(scoreRows);
        this.attn = TornadoWorkspaces.floats(rows * heads * wide);
    }

    /** The most query heads per pass whose scores fit the budget, a divisor of all of them. */
    private static int headsPerPass(int heads, int rows, int positions) {
        for (int h = heads; h >= 1; h--) {
            if (heads % h == 0 && (long) rows * h * positions * 4 <= SCORES_BUDGET) {
                return h;
            }
        }
        return 1;
    }

    /** The device buffers the tasks use, for the layer graphs' transfers. */
    public Object[] buffers() {
        return new Object[] {queries, keys, valuesT, scores, probs, attn};
    }

    // @formatter:off
    /**
     * Appends the attention of cache layer {@code kvLayer} to {@code layer}, reading the queries
     * from the packed {@code qkvResultBatch} rows and writing {@code attnOutFP16}.
     */
    // @formatter:on
    public void appendTasks(TaskGraph layer, KernelContext context, int kvLayer) {
        int qkvStride = qDim + 2 * kvDim;
        layer.task(
                "attn_gather",
                TensorCoreAttentionKernels::gatherPagedKeyValues,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.wrapKeyCacheFP16,
                state.workspace.wrapValueCacheFP16,
                keys,
                valuesT,
                kvLayer,
                kvDim,
                headSize,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride,
                paddedPositions);
        layer.task(
                "attn_q16",
                TensorCoreAttentionKernels::queriesToHalf,
                context,
                state.workspace.qkvResultBatch,
                queries,
                qDim,
                qkvStride,
                rows * qDim);
        int hp = headsPerPass;
        boolean narrow = wide != headSize;
        for (int pass = 0; pass * hp < heads; pass++) {
            int z0 = pass * hp;
            layer.task(
                    "attn_scores_" + pass,
                    BatchMmaKernels::gemmMMAStrided,
                    context,
                    queries,
                    keys,
                    scores,
                    rows,
                    paddedPositions,
                    headSize,
                    qDim,
                    headSize,
                    hp * paddedPositions,
                    headSize,
                    paddedPositions * headSize,
                    paddedPositions,
                    state.workspace.batchStartPosHolder,
                    1,
                    0,
                    z0,
                    z0,
                    kvMul,
                    0);
            layer.task(
                    "attn_softmax_" + pass,
                    BatchMmaKernels::causalSoftmax,
                    context,
                    state.workspace.batchStartPosHolder,
                    scores,
                    probs,
                    hp,
                    paddedPositions,
                    (float) (1.0 / Math.sqrt(headSize)));
            layer.task(
                    "attn_values_" + pass,
                    BatchMmaKernels::gemmMMAStrided,
                    context,
                    probs,
                    valuesT,
                    attn,
                    rows,
                    wide,
                    paddedPositions,
                    hp * paddedPositions,
                    paddedPositions,
                    narrow ? heads * wide : qDim,
                    paddedPositions,
                    headSize * paddedPositions,
                    wide,
                    state.workspace.batchStartPosHolder,
                    0,
                    1,
                    0,
                    z0,
                    kvMul,
                    z0);
        }
        if (narrow) {
            layer.task(
                    "attn_out16",
                    BatchMmaKernels::narrowHeadsToHalf,
                    context,
                    attn,
                    state.workspace.attnOutFP16,
                    heads,
                    headSize,
                    wide,
                    rows * qDim);
        } else {
            layer.task(
                    "attn_out16",
                    BatchMmaKernels::toHalf,
                    context,
                    attn,
                    state.workspace.attnOutFP16,
                    rows * qDim);
        }
    }

    /** Registers the grids of {@link #appendTasks} for the layer graph named {@code graph}. */
    public void addGrids(GridScheduler scheduler, String graph) {
        String p = graph + ".";
        scheduler.addWorkerGrid(p + "attn_gather", elementwise(paddedPositions * kvDim));
        scheduler.addWorkerGrid(p + "attn_q16", elementwise(rows * qDim));
        int hp = headsPerPass;
        for (int pass = 0; pass * hp < heads; pass++) {
            scheduler.addWorkerGrid(p + "attn_scores_" + pass, mmaGrid(rows, paddedPositions, hp));
            WorkerGrid1D softmax = new WorkerGrid1D(rows * hp * BatchMmaKernels.GROUP);
            softmax.setLocalWork(BatchMmaKernels.GROUP, 1, 1);
            scheduler.addWorkerGrid(p + "attn_softmax_" + pass, softmax);
            scheduler.addWorkerGrid(p + "attn_values_" + pass, mmaGrid(rows, wide, hp));
        }
        scheduler.addWorkerGrid(p + "attn_out16", elementwise(rows * qDim));
    }

    private static WorkerGrid mmaGrid(int m, int n, int batches) {
        WorkerGrid3D grid = new WorkerGrid3D((m / TILE) * 256, n / TILE, batches);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    private static WorkerGrid elementwise(int n) {
        return WorkerGridFactory.genericWorker((n + 255) / 256 * 256, 256);
    }
}
