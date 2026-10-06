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
                                     stationary_confirmation_frames=2)
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
                                     stationary_confirmation_frames=2)
    updates, _ = feed(sequencer, [surface(offset) for offset in range(0, 60, 5)])
    assert all(not update.burst for update in updates)
    assert sequencer.state == "FITMENT ROTATION"
