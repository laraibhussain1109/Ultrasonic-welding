from pathlib import Path


def test_tao_probe_checks_gpu_and_anomaly_training_inventory():
    source = Path("scripts/tao_probe.ps1").read_text(encoding="utf-8")

    assert "nvidia-smi" in source
    assert "tao --help" in source
    assert "pip list" in source
    assert "VISUAL_ANOMALY_CANDIDATE" in source
    assert "NO VISUAL-ANOMALY TRAINING TASK FOUND" in source
    assert "autograd/anomaly_mode.py" in source
    assert "tao_probe.txt" in source
