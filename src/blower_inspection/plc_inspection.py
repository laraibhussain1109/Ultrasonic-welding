"""PLC event protocol and fail-closed indexed inspection state machine."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .phase_locked import (ANGLES, Defect, FitmentObservation, FitmentResult, InspectionState,
                           PhaseLockedConfig, PhaseLockedStructuralInspector, PhaseResult, Verdict,
                           analyze_fitment, register_to_golden, representations)


@dataclass(frozen=True)
class PLCEvent:
    name: str
    angle: int | None = None
    motor_stopped: bool = False
    timestamp: float = 0.0

    @classmethod
    def parse(cls, message: str, timestamp: float | None = None) -> "PLCEvent":
        text = message.strip().upper()
        if text in {"HOME", "POSITION_0", "PHASE:0"}:
            return cls("HOME", 0, True, timestamp or time.time())
        for angle in ANGLES:
            if text in {f"POSITION_{angle}", f"PHASE:{angle}", f"PHASE_{angle}", str(angle)}:
                return cls("POSITION", angle, True, timestamp or time.time())
        if text in {"PART_PRESENT", "FITMENT_START", "FITMENT_COMPLETE", "MOTOR_MOVING"}:
            return cls(text, None, text != "MOTOR_MOVING", timestamp or time.time())
        raise ValueError(f"Unknown PLC event: {message!r}")


class PhaseInspectionController:
    """Own one part session; only accepts frames during a stopped phase burst."""
    def __init__(self, inspector: PhaseLockedStructuralInspector, config: PhaseLockedConfig | None = None,
                 log_root: str | Path = "data/results/phase_locked") -> None:
        self.inspector, self.config, self.log_root = inspector, config or inspector.config, Path(log_root)
        self.reset()

    def reset(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.state = InspectionState.IDLE
        self.expected_index = 0
        self.fitment_observations: list[FitmentObservation] = []
        self.fitment_result: FitmentResult | None = None
        self.home: np.ndarray | None = None
        self.current_angle: int | None = None
        self.phase_started_at = 0.0
        self.burst: list[np.ndarray] = []
        self.phase_results: dict[int, PhaseResult] = {}
        self.events: list[dict] = []
        self.final_verdict: Verdict | None = None
        self.reason_codes: list[str] = []

    def event(self, event: PLCEvent) -> None:
        self.events.append({"name": event.name, "angle": event.angle, "timestamp": event.timestamp or time.time()})
        if event.name == "PART_PRESENT" and self.state == InspectionState.IDLE:
            self.state = InspectionState.PART_PRESENT
        elif event.name == "FITMENT_START" and self.state == InspectionState.PART_PRESENT:
            self.state = InspectionState.FITMENT_ROTATION
        elif event.name == "FITMENT_COMPLETE" and self.state == InspectionState.FITMENT_ROTATION:
            self.state = InspectionState.FITMENT_ANALYSIS
            self.fitment_result = analyze_fitment(self.fitment_observations, self.config)
            if self.fitment_result.verdict != Verdict.PASS:
                self._finish(self.fitment_result.verdict, self.fitment_result.reason)
            else:
                self.state = InspectionState.FITMENT_PASS
        elif event.name == "HOME" and self.state == InspectionState.FITMENT_PASS:
            self.state = InspectionState.WAIT_PHASE
        elif event.name == "MOTOR_MOVING":
            if self.state in {InspectionState.SETTLING, InspectionState.CAPTURING_BURST}:
                self.burst.clear()
            if self.state not in {InspectionState.IDLE, InspectionState.FITMENT_ROTATION}:
                self.state = InspectionState.WAIT_PHASE
        elif event.name == "POSITION":
            if self.state != InspectionState.WAIT_PHASE or not event.motor_stopped:
                self._invalid("PHASE_RECEIVED_IN_WRONG_STATE")
                return
            expected = ANGLES[self.expected_index] if self.expected_index < len(ANGLES) else None
            if event.angle != expected or event.angle in self.phase_results:
                self._invalid("PLC_PHASE_ORDER_ERROR")
                return
            self.current_angle, self.phase_started_at, self.burst = event.angle, time.monotonic(), []
            self.state = InspectionState.SETTLING

    def observe_fitment(self, observation: FitmentObservation) -> None:
        if self.state != InspectionState.FITMENT_ROTATION:
            raise RuntimeError("Fitment observations are accepted only during FITMENT_ROTATION")
        self.fitment_observations.append(observation)

    def lock_home(self, image: np.ndarray) -> None:
        if self.state != InspectionState.WAIT_PHASE:
            raise RuntimeError("HOME can only be locked after fitment passes")
        self.home = image.copy()

    def submit_frame(self, image: np.ndarray, now: float | None = None) -> PhaseResult | None:
        if self.state not in {InspectionState.SETTLING, InspectionState.CAPTURING_BURST}:
            raise RuntimeError("Moving/asynchronous frame rejected: no stationary phase is active")
        now = time.monotonic() if now is None else now
        if (now - self.phase_started_at) * 1000 < self.config.settle_delay_ms:
            return None
        self.state = InspectionState.CAPTURING_BURST
        self.burst.append(image.copy())
        if len(self.burst) < self.config.burst_frame_count:
            return None
        self.state = InspectionState.ANALYZING_PHASE
        result = self.inspector.inspect_phase(int(self.current_angle), self.burst)
        self.phase_results[result.angle] = result
        self.expected_index += 1
        if result.verdict == Verdict.VIEW_INVALID:
            self._invalid(result.reason)
        elif self.expected_index == len(ANGLES):
            self.state = InspectionState.FINAL_HOME_CHECK
        else:
            self.state = InspectionState.WAIT_PHASE
        return result

    def check_closure(self, final_image: np.ndarray) -> Verdict:
        if self.state != InspectionState.FINAL_HOME_CHECK or self.home is None:
            self._invalid("HOME_CLOSURE_NOT_READY")
            return Verdict.INSPECTION_INVALID
        home_rep, final_rep = representations(self.home)["gradient"], representations(final_image)["gradient"]
        if home_rep.shape != final_rep.shape:
            self._invalid("HOME_ROI_SHAPE_CHANGED")
            return Verdict.INSPECTION_INVALID
        shift, response = cv2.phaseCorrelate(home_rep, final_rep)
        ratio = max(abs(shift[0]) / home_rep.shape[1], abs(shift[1]) / home_rep.shape[0])
        if response < self.config.closure_min_correlation or ratio > self.config.closure_max_translation_ratio:
            self._invalid("POSITION_DRIFT")
            return Verdict.INSPECTION_INVALID
        self.state = InspectionState.FINALIZING
        if set(self.phase_results) != set(ANGLES):
            self._invalid("MISSING_PHASES")
        elif any(result.verdict == Verdict.FAIL for result in self.phase_results.values()):
            self._finish(Verdict.FAIL, "CONFIRMED_STRUCTURAL_DEFECT")
        else:
            self._finish(Verdict.PASS, "ALL_PHASES_STRUCTURALLY_NORMAL")
        return self.final_verdict or Verdict.INSPECTION_INVALID

    def _invalid(self, reason: str) -> None:
        self._finish(Verdict.INSPECTION_INVALID, reason)

    def _finish(self, verdict: Verdict, reason: str) -> None:
        self.final_verdict, self.reason_codes = verdict, [reason]
        self.state = InspectionState.PASS if verdict == Verdict.PASS else (InspectionState.FAIL if verdict in {Verdict.FAIL, Verdict.FITMENT_FAIL} else InspectionState.INVALID)
        self.save_log()

    def save_log(self) -> Path:
        self.log_root.mkdir(parents=True, exist_ok=True)
        path = self.log_root / f"{self.session_id}.json"
        payload = {"session_id": self.session_id, "timestamp": time.time(), "state": self.state.value,
                   "final_result": self.final_verdict.value if self.final_verdict else None,
                   "reason_codes": self.reason_codes, "fitment": asdict(self.fitment_result) if self.fitment_result else None,
                   "events": self.events, "phases": {str(k): {"angle": v.angle, "verdict": v.verdict.value,
                    "reason": v.reason, "captured_frames": v.captured_frames, "qualified_frames": v.qualified_frames,
                    "registrations": v.registrations, "defects": [d.to_dict() for d in v.defects], "timings_ms": v.timings_ms}
                    for k, v in self.phase_results.items()}}
        path.write_text(json.dumps(payload, indent=2, default=lambda value: value.value), encoding="utf-8")
        return path
