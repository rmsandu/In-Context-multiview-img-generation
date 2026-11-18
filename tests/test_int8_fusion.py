import pytest


def test_fusion_stats_large_update_survives_small_update_does_not():
    torch = pytest.importorskip("torch")
    from analysis.int8_fusion import fusion_stats

    gen = torch.Generator().manual_seed(0)
    w = torch.randn(64, 256, generator=gen)
    step = w.abs().amax(dim=1, keepdim=True) / 127
    big = fusion_stats(w, 20 * step * torch.randn(64, 256, generator=gen))
    tiny = fusion_stats(w, 0.01 * step * torch.randn(64, 256, generator=gen))
    assert big["cos"] > 0.95 and big["error_over_update"] < 0.1
    assert tiny["changed"] < 0.05 and tiny["error_over_update"] > 10
