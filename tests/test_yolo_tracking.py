from blower_inspection.yolo_tracking import RotatingPartInspector, TrackedPart


def test_rotating_views_produce_one_failed_final_verdict():
    inspector = RotatingPartInspector(lost_timeout_s=1.0)
    track = TrackedPart(7, (10, 20, 100, 50), 0.9)

    assert inspector.observe_tracks([track], now=10.0) == []
    inspector.record_inspection(7, is_pass=True, anomaly_score=0.2)
    inspector.observe_tracks([track], now=10.4)
    inspector.record_inspection(7, is_pass=False, anomaly_score=0.8)

    assert inspector.observe_tracks([], now=11.3) == []
    completed = inspector.observe_tracks([], now=11.5)

    assert len(completed) == 1
    assert completed[0].track_id == 7
    assert completed[0].status == "FAIL"
    assert completed[0].frames_inspected == 2
    assert completed[0].worst_score == 0.8


def test_uninspected_track_is_not_counted():
    inspector = RotatingPartInspector(lost_timeout_s=0.5)
    inspector.observe_tracks([TrackedPart(1, (0, 0, 10, 10), 0.9)], now=0.0)

    assert inspector.observe_tracks([], now=1.0) == []
