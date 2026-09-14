import torch

from attention import CrossModalAttention


def test_cross_modal_attention():

    print("Starting Cross-Modal Attention test...\n")

    # Fake CNN features
    cnn_features = torch.randn(
        1,
        256
    )

    # Fake Swin features
    swin_features = torch.randn(
        1,
        768
    )

    print(
        "CNN input shape:",
        cnn_features.shape
    )

    print(
        "Swin input shape:",
        swin_features.shape
    )

    # Create attention module
    attention = CrossModalAttention(
        cnn_dim=256,
        swin_dim=768,
        attention_dim=512,
        num_heads=8
    )

    # Forward pass
    output = attention(
        cnn_features,
        swin_features
    )

    print(
        "Attention output shape:",
        output.shape
    )

    # Check output shape
    assert output.shape == (
        1,
        512
    ), f"Wrong output shape: {output.shape}"

    # Check NaNs
    assert not torch.isnan(
        output
    ).any(), "Output contains NaNs"

    # Check infinities
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

    print("\nCROSS-MODAL ATTENTION TEST PASSED")


if __name__ == "__main__":
    test_cross_modal_attention()