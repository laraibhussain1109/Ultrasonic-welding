from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cv2", exc_type=ImportError)
pytest.importorskip("numpy", exc_type=ImportError)

import blower_inspection.supersimplenet as supersimplenet
from blower_inspection.supersimplenet import SuperSimpleNetInspector


def test_existing_exported_torch_model_is_used_without_export(tmp_path, monkeypatch):
    model_path = tmp_path / "supersimplenet.pt"
    model_path.write_bytes(b"exported model")
    monkeypatch.setattr(supersimplenet, "_require", lambda _name: pytest.fail("unexpected anomalib import"))

    resolved = SuperSimpleNetInspector()._ensure_inference_model(model_path)

    assert resolved == model_path


def test_legacy_lightning_checkpoint_is_exported_for_inference(tmp_path, monkeypatch):
    configured_path = tmp_path / "supersimplenet.ckpt"
    configured_path.write_bytes(b"lightning checkpoint")
    calls = []

    class FakeEngine:
        def export(self, **kwargs):
            calls.append(kwargs)
            exported = kwargs["export_root"] / "weights" / "torch" / "model.pt"
            exported.parent.mkdir(parents=True)
            exported.write_bytes(b"torch inference model")
            return exported

    modules = {
        "anomalib.deploy": SimpleNamespace(ExportType=SimpleNamespace(TORCH="torch")),
        "anomalib.models": SimpleNamespace(Supersimplenet=lambda: "model"),
        "anomalib.engine": SimpleNamespace(Engine=FakeEngine),
    }
    monkeypatch.setattr(supersimplenet, "_require", modules.__getitem__)

    resolved = SuperSimpleNetInspector()._ensure_inference_model(configured_path)

    assert resolved == tmp_path / "supersimplenet.pt"
    assert resolved.read_bytes() == b"torch inference model"
    assert calls[0]["ckpt_path"] == configured_path
    assert calls[0]["export_type"] == "torch"


def test_missing_model_explains_that_training_must_create_pt_export(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"exported \.pt model"):
        SuperSimpleNetInspector()._ensure_inference_model(tmp_path / "supersimplenet.pt")
