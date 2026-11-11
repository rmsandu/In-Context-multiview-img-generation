import numpy as np
import pytest

from analysis.lora_anatomy import classify, inner, module_stats


def test_module_stats_match_explicit_svd():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(16, 300)), rng.normal(size=(200, 16))
    dense = b @ a
    sv = np.linalg.svd(dense, compute_uv=False)
    stats = module_stats(a, b)
    assert stats["norm"] == pytest.approx(np.linalg.norm(dense))
    assert stats["sigma_max"] == pytest.approx(sv[0])
    assert 1.0 < stats["eff_rank"] <= 16.0


def test_rank_one_update_has_effective_rank_one():
    rng = np.random.default_rng(1)
    a = np.zeros((16, 50))
    b = np.zeros((40, 16))
    a[0], b[:, 0] = rng.normal(size=50), rng.normal(size=40)
    stats = module_stats(a, b)
    assert stats["eff_rank"] == pytest.approx(1.0)
    assert stats["top1_energy"] == pytest.approx(1.0)


def test_inner_product_matches_dense():
    rng = np.random.default_rng(2)
    a1, b1 = rng.normal(size=(16, 30)), rng.normal(size=(20, 16))
    a2, b2 = rng.normal(size=(16, 30)), rng.normal(size=(20, 16))
    assert inner(a1, b1, a2, b2) == pytest.approx(np.sum((b1 @ a1) * (b2 @ a2)))


@pytest.mark.parametrize(
    ("module", "stream", "role"),
    [
        ("transformer_blocks.3.attn.add_q_proj", "double", "text attention"),
        ("transformer_blocks.3.attn.to_add_out", "double", "text attention"),
        ("transformer_blocks.3.attn.to_q", "double", "image attention"),
        ("transformer_blocks.3.norm1_context.linear", "double", "modulation (adaLN)"),
        ("transformer_blocks.3.ff_context.net.2", "double", "text MLP"),
        ("single_transformer_blocks.12.proj_mlp", "single", "joint MLP"),
        ("single_transformer_blocks.12.attn.to_k", "single", "joint attention"),
        ("single_transformer_blocks.12.proj_out", "single", "joint attn+MLP out"),
    ],
)
def test_classify_roles(module, stream, role):
    info = classify(module)
    assert info["stream"] == stream and info["role"] == role


def test_subspace_overlap_bounds():
    from analysis.lora_anatomy import subspace_overlap

    rng = np.random.default_rng(3)
    m = rng.normal(size=(400, 8))
    assert subspace_overlap(m, m @ rng.normal(size=(8, 8))) == pytest.approx(1.0)
    e = np.eye(400)
    assert subspace_overlap(e[:, :8], e[:, 8:16]) == pytest.approx(0.0)
    random = subspace_overlap(m, rng.normal(size=(400, 8)))
    assert random < 0.1  # about 8 / 400 for unrelated subspaces
