import torch

from suanet.model import SpeckleVarianceAttention, SUANet


def test_suanet_scratch_forward_shape():
    model = SUANet(encoder_name="scratch", pretrained=False, num_classes=2)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 2)


def test_all_ablation_combinations_forward():
    inputs = torch.randn(1, 3, 64, 64)
    for use_dla in (False, True):
        for use_sva in (False, True):
            for use_mgp in (False, True):
                model = SUANet(
                    encoder_name="scratch",
                    pretrained=False,
                    use_dla=use_dla,
                    use_sva=use_sva,
                    use_mgp=use_mgp,
                )
                assert model(inputs).shape == (1, 2)


def test_sva_variance_map_is_finite_and_bounded():
    module = SpeckleVarianceAttention(channels=8, kernel=7)
    variance = module.local_variance(torch.randn(2, 8, 16, 16))
    assert torch.isfinite(variance).all()
    assert variance.min() >= 0
    assert variance.max() <= 1.00001
