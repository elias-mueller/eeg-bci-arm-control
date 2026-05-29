import pytest

torch = pytest.importorskip("torch")
from eeg_bci_pipeline.training.eegnet import EEGNet  # noqa: E402


def test_forward_produces_correct_output_shape():
    model = EEGNet(n_channels=4, n_samples=128, n_classes=2, kernel_length=32)
    x = torch.randn(3, 1, 4, 128)

    out = model(x)

    assert out.shape == (3, 2)


def test_forward_works_with_different_dimensions():
    model = EEGNet(n_channels=22, n_samples=768, n_classes=4, kernel_length=125)
    x = torch.randn(2, 1, 22, 768)

    out = model(x)

    assert out.shape == (2, 4)


def test_forward_with_custom_hyperparameters():
    model = EEGNet(
        n_channels=8,
        n_samples=256,
        n_classes=3,
        f1=16,
        d=4,
        kernel_length=64,
        dropout_rate=0.25,
    )
    x = torch.randn(5, 1, 8, 256)

    out = model(x)

    assert out.shape == (5, 3)


def test_forward_produces_finite_logits():
    model = EEGNet(n_channels=4, n_samples=128, n_classes=2, kernel_length=32)
    x = torch.randn(4, 1, 4, 128)

    out = model(x)

    assert torch.isfinite(out).all()


def test_rejects_too_few_samples():
    with pytest.raises(ValueError, match="n_samples"):
        EEGNet(n_channels=4, n_samples=16, n_classes=2)
