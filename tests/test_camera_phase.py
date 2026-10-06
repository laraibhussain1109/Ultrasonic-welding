from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.camera_phase import CameraPhaseSequencer


def surface(offset: int = 0) -> np.ndarray:
    image = np.zeros((100, 240, 3), np.uint8)
    for x in range(-20, 260, 20):
        cv2.line(image, (x + offset, 0), (x + offset, 99), (220, 220, 220), 5)
    return image


def feed(sequencer, frames, start=0.0, step=.05):
    updates = []
    now = start
    for frame in frames:
        updates.append(sequencer.offer(frame, now=now))
        now += step
    return updates, now


def test_camera_only_sequence_uses_motion_order_not_surface_identity():
    sequencer = CameraPhaseSequencer(settle_delay_ms=0, burst_frame_count=3,
                                     moving_confirmation_frames=2,
                                     stationary_confirmation_frames=2,
                                     lock_initial_home=False)
    # Initial continuous revolution followed by a confirmed HOME stop.
    moving = [surface(offset) for offset in (0, 6, 12, 18, 24)]
    updates, now = feed(sequencer, moving + [surface(24)] * 3)
    assert sequencer.home_frame is not None
    assert updates[-1].state == "HOME LOCKED — WAITING FOR 60°"

    emitted = []
    for expected in (60, 120, 180, 240, 300, 360):
        # The visible surface is deliberately unrelated to the numeric angle;
        # only the next moving-to-stopped transition advances the phase.
        random_offset = (expected * 7) % 19
        updates, now = feed(
            sequencer,
            [surface(random_offset), surface(random_offset + 7), surface(random_offset + 14)]
            + [surface(random_offset + 14)] * 5,
            start=now,
        )
        bursts = [update for update in updates if update.burst]
        assert len(bursts) == 1
        emitted.append(bursts[0].angle)
    assert emitted == [60, 120, 180, 240, 300, 360]


def test_camera_sequence_does_not_inspect_while_moving():
    sequencer = CameraPhaseSequencer(settle_delay_ms=0, burst_frame_count=2,
                                     moving_confirmation_frames=2,
                                     stationary_confirmation_frames=2,
                                     lock_initial_home=False)
    updates, _ = feed(sequencer, [surface(offset) for offset in range(0, 60, 5)])
    assert all(not update.burst for update in updates)
    assert sequencer.state == "FITMENT ROTATION"


def test_stationary_sensor_noise_does_not_hold_machine_in_motion():
    sequencer = CameraPhaseSequencer(settle_delay_ms=0, burst_frame_count=2,
                                     moving_confirmation_frames=2,
                                     stationary_confirmation_frames=2,
                                     motion_threshold=2.5,
                                     motion_flow_threshold=.35,
                                     lock_initial_home=False)
    rng = np.random.default_rng(7)
    base = surface(0).astype(np.int16)
    # Establish real motion and then stop at HOME.
    frames = [surface(offset) for offset in (0, 7, 14, 21)]
    for _ in range(5):
        noise = rng.integers(-2, 3, base.shape, dtype=np.int16)
        frames.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    updates, _ = feed(sequencer, frames)
    assert sequencer.home_frame is not None
    assert updates[-1].state == "HOME LOCKED — WAITING FOR 60°"
    assert updates[-1].pixel_motion < sequencer.motion_threshold


def test_settling_frames_can_be_flushed_without_losing_phase():
    sequencer = CameraPhaseSequencer(settle_delay_ms=0, burst_frame_count=2,
                                     moving_confirmation_frames=2,
                                     stationary_confirmation_frames=2,
                                     lock_initial_home=False)
    updates, now = feed(sequencer, [surface(x) for x in (0, 7, 14)] + [surface(14)] * 3)
    assert sequencer.home_frame is not None
    updates, now = feed(sequencer, [surface(x) for x in (2, 9, 16)] + [surface(16)] * 2, start=now)
    assert updates[-1].angle == 60
    assert updates[-1].state.startswith("CAPTURING")
    sequencer.discard_partial_burst()
    updates, _ = feed(sequencer, [surface(16)] * 2, start=now)
    emitted = [update for update in updates if update.burst]
    assert emitted and emitted[0].angle == 60


def test_camera_station_uses_armed_frame_as_home_and_captures_all_six_stops():
    sequencer = CameraPhaseSequencer(settle_delay_ms=0, burst_frame_count=2,
                                     moving_confirmation_frames=2,
                                     stationary_confirmation_frames=2)
    first = sequencer.offer(surface(3), now=0.0)
    assert first.state == "HOME LOCKED — WAITING FOR 60°"
    assert first.home_frame is not None

    emitted = []
    now = .05
    for index in range(6):
        base = 3 + index * 4
        updates, now = feed(
            sequencer,
            [surface(base + 5), surface(base + 10), surface(base + 15)]
            + [surface(base + 15)] * 4,
            start=now,
        )
        emitted.extend(update.angle for update in updates if update.burst)
    assert emitted == [60, 120, 180, 240, 300, 360]
