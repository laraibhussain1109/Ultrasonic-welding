from datetime import datetime

from blower_inspection.daily_stats import DailyStatistics, operating_day


def test_operating_day_changes_at_seven_am():
    assert operating_day(datetime(2026, 7, 19, 6, 59)) == "2026-07-18"
    assert operating_day(datetime(2026, 7, 19, 7, 0)) == "2026-07-19"


def test_daily_statistics_persists_final_verdicts(tmp_path):
    stats = DailyStatistics(tmp_path / "daily.json")
    now = datetime(2026, 7, 19, 8, 0)

    assert stats.record("PASS", now) == {"inspected": 1, "passed": 1, "failed": 0}
    assert stats.record("FAIL", now) == {"inspected": 2, "passed": 1, "failed": 1}
    assert DailyStatistics(tmp_path / "daily.json").counts(now) == {"inspected": 2, "passed": 1, "failed": 1}


def test_memory_only_statistics_never_write_disk(tmp_path):
    path = tmp_path / "daily.json"
    stats = DailyStatistics(path, memory_only=True)
    now = datetime(2026, 7, 19, 8, 0)

    assert stats.record("FAIL", now) == {"inspected": 1, "passed": 0, "failed": 1}
    assert stats.counts(now)["inspected"] == 1
    assert not path.exists()
    stats.clear()
    assert stats.counts(now) == {"inspected": 0, "passed": 0, "failed": 0}
