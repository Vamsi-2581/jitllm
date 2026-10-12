package org.beehive.jitllm.backend.tornado.layers;

import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The Q8_0 projections of a tensor-core batched prefill as FP16 GEMMs: each weight decoded into one
 * shared FP16 scratch, then {@link TransformerBatchPrefillKernels#gemmMMA} over it.
 *
 * <p>The Q8_0 GEMM decodes every weight again for every 128-row tile of the batch, inside its inner
 * loop. Decoding once per chunk costs one pass over the weights and runs the GEMM at the FP16 rate.
 * The scratch is the size of the widest projection the caller multiplies.
 *
 * <p>The GEMM is {@link SplitKGemm}'s, which splits the depth of a projection too narrow to fill
 * the device.
 *
 * <p>When the decoded weights of the whole model fit in {@link #RESIDENT_BYTES}, each projection
 * keeps its own FP16 copy instead, decoded on the first chunk only: a small model's decode costs a
 * sixth of its prefill, and its FP16 weights are a few gigabytes.
 */
// @formatter:on
public final class Q8DecodedProjections {

    /** The FP16 weight bytes up to which every projection keeps a resident decoded copy. */
    public static final long RESIDENT_BYTES = 6L << 30;

    /** Whether a model whose layers hold {@code weights} weights keeps resident FP16 copies. */
    public static boolean resident(long weights) {
        return 2 * weights <= RESIDENT_BYTES;
    }

    private final KernelContext context;
    private final HalfFloatArray scratch;
    private final SplitKGemm gemm;
    private final Map<String, Integer> decodeTasks = new LinkedHashMap<>();
    private final java.util.List<String> markTasks = new java.util.ArrayList<>();
    private final boolean resident;
    private final IntArray decoded;
    private int nextFlag;

    /**
     * @param widest the most weights one projection call multiplies, all its tensors together
     * @param rows the padded batch, the GEMMs' row count
     * @param widestSplit the widest output a split projection writes
     */
    public Q8DecodedProjections(
            KernelContext context,
            long widest,
            int rows,
            int widestSplit,
            boolean resident,
            int projectionCalls) {
        this.context = context;
        this.resident = resident;
        this.decoded = new IntArray(Math.max(1, projectionCalls));
        this.scratch = new HalfFloatArray(resident ? 1 : Math.toIntExact(widest));
        this.gemm = new SplitKGemm(context, rows, widestSplit);
    }

    /** The buffers, which every layer graph transfers or consumes with its intermediates. */
    public Object[] buffers() {
        return new Object[] {scratch, gemm.partial(), decoded};
    }

    // @formatter:off
    /**
     * {@code c[m][N] = a[m][k] . [W_0 | W_1 | ...]^T} as task {@code task}: each Q8_0 weight
     * ({@code [N_i][k]}, row-major) decoded into the scratch after the one before it, then one FP16
     * GEMM over their rows, so the output columns are the weights' rows in order. The GEMM's grid is
     * the caller's, under {@code task}; {@link #addGrids} registers the decode tasks'.
     */
    // @formatter:on
    public void project(
            TaskGraph layer,
            String task,
            HalfFloatArray a,
            FloatArray c,
            int m,
            int k,
            TornadoTensor... ws) {
        HalfFloatArray b = scratch;
        int flag = 0;
        if (resident) {
            long total = 0;
            for (TornadoTensor w : ws) {
                total += w.asByteArray().getSize() / 34 * 32;
            }
            b = new HalfFloatArray(Math.toIntExact(total));
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, b);
            flag = nextFlag++;
        }
        int offset = 0;
        for (int i = 0; i < ws.length; i++) {
            if (ws[i].isPacked()) {
                throw PackedWeights.noPackedKernel("Q8_0 batch-prefill task '" + task + "'");
            }
            // 34 bytes to 32 weights.
            int n = ws[i].asByteArray().getSize() / 34 * 32;
            String decode = task + "Decode" + i;
            if (resident) {
                layer.task(
                        decode,
                        TransformerBatchPrefillKernels::dequantizeQ8_0ToFP16Once,
                        context,
                        ws[i].asByteArray(),
                        b,
                        offset,
                        n,
                        decoded,
                        flag);
            } else {
                layer.task(
                        decode,
                        TransformerBatchPrefillKernels::dequantizeQ8_0ToFP16,
                        context,
                        ws[i].asByteArray(),
                        b,
                        offset,
                        n);
            }
            decodeTasks.put(
                    layer.getTaskGraphName() + "." + decode,
                    resident ? Math.min(n, TransformerBatchPrefillKernels.DECODE_ONCE_LANES) : n);
            offset += n;
        }
        if (resident) {
            String mark = task + "Decoded";
            layer.task(mark, TransformerBatchPrefillKernels::markDecoded, context, decoded, flag);
            markTasks.add(layer.getTaskGraphName() + "." + mark);
        }
        gemm.gemm(layer, task, a, b, c, m, offset / k, k);
    }

    /**
     * Registers the grids of the decode tasks and the split projections {@link #project} added,
     * replacing the caller's grid for a split projection. Call after the caller's grids.
     */
    public void addGrids(GridScheduler scheduler) {
        decodeTasks.forEach(
                (name, n) ->
                        scheduler.addWorkerGrid(
                                name, WorkerGridFactory.genericWorker((n + 255) / 256 * 256, 256)));
        markTasks.forEach(name -> scheduler.addWorkerGrid(name, WorkerGridFactory.genericWorker(1, 1)));
        gemm.addGrids(scheduler);
    }
}
