package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvQwen2Q8AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.QWEN2_5_0_5B_Q8_0, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.checkBatchedOnMetal(Fixture.QWEN2_5_0_5B_Q8_0);
    }

    @Test
    public void singleToken1_5B() throws Exception {
        GpuFp16KvPrecision.check(
                "qwen2.5-1.5b-instruct-q8_0.gguf", GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void batchedPrefill1_5B() throws Exception {
        GpuFp16KvPrecision.checkBatchedOnMetal("qwen2.5-1.5b-instruct-q8_0.gguf");
    }
}
