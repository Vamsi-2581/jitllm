package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.mlx.Mlx;

import java.util.Random;

/**
 * The MLX down projection of a batched prefill over a row whose SwiGLU output passes FP16's 65504,
 * as Granite 4.0's first token does (134340 into the down GEMM, 102097 out of it): with {@code
 * swigluRowScale} the residual matches an FP32 reference and every row stays finite, and the
 * unscaled {@code swiglu} / {@code addResidual} pair turns the same row non-finite.
 */
public class MlxPrefillDownProjectionAccelTest {

    private static final int ROWS = 4;
    private static final int HIDDEN = 4096;
    private static final int DIM = 256;
    private static final float RESIDUAL_SCALE = 0.25f;

    @Test
    public void anOverflowingRowIsScaledIntoHalfRangeAndBack() throws Exception {
        assumeTrue("environment absent: MLX on Metal", MlxPrefill.available());
        Random rng = new Random(7L);
        HalfFloatArray gateUp = new HalfFloatArray(ROWS * 2 * HIDDEN);
        for (int b = 0; b < ROWS; b++) {
            for (int i = 0; i < HIDDEN; i++) {
                float gate = (rng.nextFloat() * 2.0f - 1.0f) * 4.0f;
                float up = (rng.nextFloat() * 2.0f - 1.0f) * 4.0f;
                if (b == 0 && i % 512 == 0) {
                    gate = 300.0f;
                    up = 450.0f;
                }
                gateUp.set(b * 2 * HIDDEN + i, new HalfFloat(gate));
                gateUp.set(b * 2 * HIDDEN + HIDDEN + i, new HalfFloat(up));
            }
        }
        HalfFloatArray w = new HalfFloatArray(DIM * HIDDEN);
        for (int j = 0; j < DIM; j++) {
            for (int i = 0; i < HIDDEN; i++) {
                float v = (rng.nextFloat() * 2.0f - 1.0f) * 0.02f;
                w.set(j * HIDDEN + i, new HalfFloat(i % 512 == 0 ? 0.25f : v));
            }
        }
        FloatArray x = new FloatArray(ROWS * DIM);
        FloatArray xControl = new FloatArray(ROWS * DIM);
        for (int i = 0; i < ROWS * DIM; i++) {
            float v = (rng.nextFloat() * 2.0f - 1.0f) * 10.0f;
            x.set(i, v);
            xControl.set(i, v);
        }
        float[] expected = reference(gateUp, w, x);

        FloatArray rowScale = new FloatArray(ROWS);
        rowScale.init(Float.NaN);
        HalfFloatArray hb = new HalfFloatArray(ROWS * HIDDEN);
        HalfFloatArray out = new HalfFloatArray(ROWS * DIM);
        HalfFloatArray hbControl = new HalfFloatArray(ROWS * HIDDEN);
        HalfFloatArray outControl = new HalfFloatArray(ROWS * DIM);
        KernelContext context = new KernelContext();
        TaskGraph g =
                new TaskGraph("down")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                context,
                                gateUp,
                                w,
                                x,
                                xControl,
                                rowScale,
                                hb,
                                out,
                                hbControl,
                                outControl)
                        .task(
                                "row_scale",
                                MlxPrefillKernels::swigluRowScale,
                                context,
                                gateUp,
                                rowScale,
                                HIDDEN,
                                ROWS,
                                256)
                        .task(
                                "swiglu",
                                MlxPrefillKernels::swigluScaled,
                                context,
                                gateUp,
                                hb,
                                rowScale,
                                HIDDEN,
                                ROWS * HIDDEN)
                        .libraryTask("gemm", Mlx::matmulTransposed, hb, w, out, ROWS, HIDDEN, DIM)
                        .task(
                                "residual",
                                MlxPrefillKernels::addRowScaledResidual,
                                context,
                                x,
                                out,
                                rowScale,
                                RESIDUAL_SCALE,
                                DIM,
                                ROWS * DIM)
                        .task(
                                "swiglu_control",
                                MlxPrefillKernels::swiglu,
                                context,
                                gateUp,
                                hbControl,
                                HIDDEN,
                                ROWS * HIDDEN)
                        .libraryTask(
                                "gemm_control",
                                Mlx::matmulTransposed,
                                hbControl,
                                w,
                                outControl,
                                ROWS,
                                HIDDEN,
                                DIM)
                        .task(
                                "residual_control",
                                MlxPrefillKernels::addResidual,
                                context,
                                xControl,
                                outControl,
                                RESIDUAL_SCALE,
                                ROWS * DIM)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, x, xControl, rowScale);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("down.row_scale", grid(ROWS * 256, 256));
        s.addWorkerGrid("down.swiglu", grid(ROWS * HIDDEN, 256));
        s.addWorkerGrid("down.residual", grid(ROWS * DIM, 256));
        s.addWorkerGrid("down.swiglu_control", grid(ROWS * HIDDEN, 256));
        s.addWorkerGrid("down.residual_control", grid(ROWS * DIM, 256));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }

        assertTrue("row 0 scaled: " + rowScale.get(0), rowScale.get(0) < 1.0f);
        for (int b = 1; b < ROWS; b++) {
            assertEquals("row " + b + " left alone", 1.0f, rowScale.get(b), 0.0f);
        }
        boolean controlOverflowed = false;
        for (int j = 0; j < DIM; j++) {
            controlOverflowed |= !Float.isFinite(xControl.get(j));
        }
        assertTrue(
                "the unscaled control stayed finite: the case no longer overflows",
                controlOverflowed);
        for (int b = 0; b < ROWS; b++) {
            float peak = 0.0f;
            for (int j = 0; j < DIM; j++) {
                peak = Math.max(peak, Math.abs(expected[b * DIM + j]));
            }
            for (int j = 0; j < DIM; j++) {
                int id = b * DIM + j;
                assertFalse("row " + b + " col " + j + " non-finite", !Float.isFinite(x.get(id)));
                assertEquals(
                        "row " + b + " col " + j, expected[id], x.get(id), 4e-3f * peak + 1e-2f);
            }
        }
    }

    /** {@code x + RESIDUAL_SCALE * W · SwiGLU(gate, up)} in FP32 from the same FP16 inputs. */
    private static float[] reference(HalfFloatArray gateUp, HalfFloatArray w, FloatArray x) {
        float[] y = new float[ROWS * DIM];
        float[] h = new float[HIDDEN];
        for (int b = 0; b < ROWS; b++) {
            for (int i = 0; i < HIDDEN; i++) {
                float gate = gateUp.get(b * 2 * HIDDEN + i).getFloat32();
                float up = gateUp.get(b * 2 * HIDDEN + HIDDEN + i).getFloat32();
                h[i] = gate / (1.0f + (float) Math.exp(-gate)) * up;
            }
            for (int j = 0; j < DIM; j++) {
                double sum = 0.0;
                for (int i = 0; i < HIDDEN; i++) {
                    sum += (double) w.get(j * HIDDEN + i).getFloat32() * h[i];
                }
                y[b * DIM + j] = x.get(b * DIM + j) + RESIDUAL_SCALE * (float) sum;
            }
        }
        return y;
    }

    private static WorkerGrid grid(int global, int local) {
        WorkerGrid w = new WorkerGrid1D(global);
        w.setLocalWork(local, 1, 1);
        return w;
    }
}
