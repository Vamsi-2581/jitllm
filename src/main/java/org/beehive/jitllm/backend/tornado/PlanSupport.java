package org.beehive.jitllm.backend.tornado;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

/**
 * Where a family's plan cannot run at all, whatever the key/value cache precision: on a device its
 * kernels are not written for, or split across devices in a mode a split has no plan for.
 *
 * <p>Asked before the cache and batched-prefill checks, which assume the plan can run, and before
 * any device buffer exists. Each family's plan provider names its own devices ({@code
 * TornadoPlanProvider.deviceUnsupported}); this class names no family.
 */
public final class PlanSupport {

    private PlanSupport() {}

    /**
     * Why {@code architecture} cannot run under {@code policy} on {@code devices} — one device, or
     * each stage's of a split — or empty when it can.
     */
    public static Optional<String> unsupported(
            String architecture, ExecutionPolicy policy, List<BackendId> devices, boolean split) {
        if (split) {
            Optional<String> mode = splitUnsupported(policy);
            if (mode.isPresent()) {
                return mode;
            }
        }
        for (BackendId backend : devices) {
            Optional<String> device = deviceUnsupported(architecture, backend);
            if (device.isPresent()) {
                return device;
            }
        }
        return Optional.empty();
    }

    /** Why a model split across devices cannot run {@code policy}, if it cannot. */
    static Optional<String> splitUnsupported(ExecutionPolicy policy) {
        return policy.phaseStrategy() == ExecutionPolicy.PhaseStrategy.PREFILL_DECODE
                        && policy.prefillBatchSize() <= 1
                ? Optional.of(
                        "a model split across devices (--devices) runs single-token or with"
                                + " batched prefill (--batch-prefill-size N, N > 1), not"
                                + " sequential prefill/decode; choose one of them or drop"
                                + " --devices")
                : Optional.empty();
    }

    /** Why {@code architecture}'s plans cannot run on a {@code backend} device, if they cannot. */
    static Optional<String> deviceUnsupported(String architecture, BackendId backend) {
        if (BackendId.CPU.equals(backend)) {
            return Optional.empty();
        }
        return TornadoPlanRegistry.provider(architecture)
                .flatMap(provider -> provider.deviceUnsupported(backend))
                .map(reason -> "on " + backend + ", " + reason);
    }

    /**
     * Refuses a plan {@code model} cannot run under {@code policy}: on the device it would build
     * for, on each stage's device of a split, or in a mode a split cannot run.
     *
     * @throws UnsupportedOperationException naming the model and the reason
     */
    public static void require(Model model, ExecutionPolicy policy, boolean gpu) {
        if (!gpu) {
            return;
        }
        Optional<DeviceSplit> split = DeviceSplit.fromSystemProperties();
        List<BackendId> devices = new ArrayList<>();
        if (split.isPresent()) {
            for (TornadoDevice device : TornadoVMMasterPlanPipeline.stageDevices(split.get())) {
                devices.add(TornadoDevices.of(device).backend());
            }
        } else {
            devices.add(TornadoDevices.target().backend());
        }
        String architecture = model.architectureId().toString();
        unsupported(architecture, policy, devices, split.isPresent())
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(
                                    refusal(architecture, model, reason));
                        });
    }

    static String refusal(String architecture, Model model, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                architecture + " / " + model.weights().dataType() + " cannot run: " + reason);
    }
}
