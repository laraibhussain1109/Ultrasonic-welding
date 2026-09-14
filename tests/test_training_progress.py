from blower_inspection.training_progress import TrainingProgress
from blower_inspection.tao_training import _run_with_live_progress


def test_progress_formats_count_percent_elapsed_and_eta():
    progress = TrainingProgress("Distillation epoch", 2, 8, 10.0)

    assert progress.eta_seconds == 30.0
    assert progress.format() == (
        "Distillation epoch: 2/8 (25.0%) | elapsed 0:00:10 | ETA 0:00:30"
    )


def test_progress_reports_calculating_before_first_unit():
    assert "ETA calculating" in TrainingProgress("TAO train", 0, 50, 0.0).format()


def test_tao_output_is_streamed_and_zero_based_epochs_are_reported(monkeypatch, capsys):
    class Process:
        stdout = iter(["setup\n", "Epoch 0: loss=1\n", "Epoch 1: loss=.5\n"])

        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr("blower_inspection.tao_training.subprocess.Popen", lambda *args, **kwargs: Process())
    updates = []

    _run_with_live_progress(["docker"], "train", 2, updates.append)

    assert capsys.readouterr().out == "setup\nEpoch 0: loss=1\nEpoch 1: loss=.5\n"
    assert [(item.completed, item.total) for item in updates] == [(0, 2), (1, 2), (2, 2), (2, 2)]
