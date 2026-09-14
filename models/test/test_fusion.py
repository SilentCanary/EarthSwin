import torch

from fusion import ScaleFusion


def test_scale_fusion():

    print("Starting Scale Fusion test...\n")

    pyramid_features = torch.randn(
        4,
        3,
        512
    )

    print(
        "Input shape:",
        pyramid_features.shape
    )

    fusion = ScaleFusion(
        feature_dim=512,
        num_scales=3
    )

    output = fusion(
        pyramid_features
    )

    print(
        "Output shape:",
        output.shape
    )

    assert output.shape == (
        4,
        512
    ), f"Wrong output shape: {output.shape}"

    assert not torch.isnan(
        output
    ).any(), "Output contains NaNs"

    assert not torch.isinf(
        output
    ).any(), "Output contains infinities"

    print(
        "NaNs:",
        torch.isnan(output).sum().item()
    )

    print(
        "Infs:",
        torch.isinf(output).sum().item()
    )

    print("\nSCALE FUSION TEST PASSED")


if __name__ == "__main__":
    test_scale_fusion()