"""STFPM-style student/teacher anomaly detection for normal-only training."""

from __future__ import annotations

from typing import Any, Iterable


def resnet_features(model: Any, tensor: Any) -> list[Any]:
    """Return the three spatial feature levels used by STFPM."""
    value = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
    level1 = model.layer1(value)
    level2 = model.layer2(level1)
    level3 = model.layer3(level2)
    return [level1, level2, level3]


def build_student_teacher(torch: Any, models: Any, device: Any) -> tuple[Any, Any]:
    """Create a frozen ImageNet teacher and trainable randomly initialized student."""
    teacher = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    student = models.resnet18(weights=None).to(device).train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, student


def feature_matching_loss(torch: Any, teacher_features: Iterable[Any], student_features: Iterable[Any]) -> Any:
    losses = []
    for teacher, student in zip(teacher_features, student_features):
        teacher = torch.nn.functional.normalize(teacher, dim=1)
        student = torch.nn.functional.normalize(student, dim=1)
        losses.append((teacher - student).pow(2).mean())
    return torch.stack(losses).sum()


def discrepancy_map(torch: Any, teacher: Any, student: Any, tensor: Any, output_size: int) -> Any:
    """Fuse normalized teacher/student discrepancies into one spatial map."""
    with torch.inference_mode():
        teacher_features = resnet_features(teacher, tensor)
        student_features = resnet_features(student, tensor)
        maps = []
        for teacher_level, student_level in zip(teacher_features, student_features):
            teacher_level = torch.nn.functional.normalize(teacher_level, dim=1)
            student_level = torch.nn.functional.normalize(student_level, dim=1)
            difference = (teacher_level - student_level).pow(2).sum(dim=1, keepdim=True)
            maps.append(
                torch.nn.functional.interpolate(
                    difference, size=(output_size, output_size), mode="bilinear", align_corners=False
                )
            )
        return torch.stack(maps).sum(dim=0).squeeze(1)
