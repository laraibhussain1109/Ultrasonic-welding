import pytest

torch = pytest.importorskip("torch")

from blower_inspection.vit_tokens import spatial_patch_tokens


class Model:
    num_prefix_tokens = 1


def test_tensor_dinov2_output_drops_cls_token_before_grid_reshape():
    output = torch.arange(4 * 1370 * 3, dtype=torch.float32).reshape(4, 1370, 3)

    tokens = spatial_patch_tokens(output, Model(), (37, 37))

    assert tokens.shape == (4, 1369, 3)
    assert torch.equal(tokens[:, 0], output[:, 1])


def test_mapping_dinov2_patch_tokens_are_not_trimmed_twice():
    patch_tokens = torch.zeros((4, 1369, 384))

    tokens = spatial_patch_tokens({"x_norm_patchtokens": patch_tokens}, Model(), (37, 37))

    assert tokens is patch_tokens


def test_incompatible_patch_count_has_actionable_error():
    output = torch.zeros((1, 1000, 8))

    with pytest.raises(RuntimeError, match="token count does not match"):
        spatial_patch_tokens(output, Model(), (37, 37))
