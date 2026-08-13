import pytest

torch = pytest.importorskip("torch")

from blower_inspection.distillation import discrepancy_map, feature_matching_loss


class TinyFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.Identity()
        self.layer1 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.layer2 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.layer3 = torch.nn.Conv2d(4, 4, 3, padding=1)


def test_identical_student_teacher_have_zero_discrepancy():
    teacher = TinyFeatureModel().eval()
    student = TinyFeatureModel().eval()
    student.load_state_dict(teacher.state_dict())
    tensor = torch.randn(1, 3, 20, 20)

    result = discrepancy_map(torch, teacher, student, tensor, output_size=20)

    assert result.shape == (1, 20, 20)
    assert float(result.max()) < 1e-6


def test_feature_matching_loss_increases_for_different_features():
    normal = [torch.ones(1, 2, 4, 4)]
    matching = feature_matching_loss(torch, normal, [torch.ones(1, 2, 4, 4)])
    different = feature_matching_loss(
        torch, normal, [torch.cat((torch.ones(1, 1, 4, 4), -torch.ones(1, 1, 4, 4)), dim=1)]
    )

    assert float(matching) == pytest.approx(0.0)
    assert float(different) > float(matching)
