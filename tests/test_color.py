import numpy as np

from chromarecover.color import linear_to_srgb, srgb_to_linear


def test_srgb_linear_round_trip() -> None:
    values = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    rgb = np.stack([values, values[::-1], np.full_like(values, 0.37)], axis=-1)
    reconstructed = linear_to_srgb(srgb_to_linear(rgb))
    assert np.max(np.abs(reconstructed - rgb)) < 2e-6


def test_srgb_reference_points() -> None:
    result = srgb_to_linear(np.array([0.0, 0.04045, 1.0], dtype=np.float32))
    assert np.allclose(result, [0.0, 0.0031308, 1.0], atol=2e-6)

