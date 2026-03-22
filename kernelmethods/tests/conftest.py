import numpy as np
import pytest


@pytest.fixture
def rng():
    """Per-test deterministic RNG to avoid order-dependent global state."""

    return np.random.default_rng(42)


def make_numeric_sample(rng, num_samples, sample_dim):
    return rng.random((num_samples, sample_dim))


def make_labels(rng, num_samples, choices=(1, 2)):
    return rng.choice(np.asarray(choices), size=num_samples)
