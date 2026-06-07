"""Shared pytest configuration for the eeg_bci_pipeline test suite."""

from __future__ import annotations


def pytest_configure(config: object) -> None:
    # The EEGNet tests train tiny models (a few channels, short windows, batch
    # size 4). On a many-core host PyTorch's default intra-op parallelism uses
    # one thread per core, and for ops this small the thread-sync overhead dwarfs
    # the compute, making these tests 100x+ slower and wildly load-dependent
    # (seconds vs minutes for the same test). Pinning to a single thread for the
    # test session makes the tiny trainings fast and their timing deterministic.
    # Only the test process is affected; runtime/benchmark code never imports this
    # conftest, so real training keeps full parallelism.
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
