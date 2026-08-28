import cv2
import numpy as np

from chromarecover.primitives import extract_primitives, group_primitives


def test_contour_primitives_and_graph_reject_distant_fragment() -> None:
    glyphs = np.zeros((180, 280), dtype=np.uint8)
    cv2.putText(glyphs, "820", (28, 128), cv2.FONT_HERSHEY_DUPLEX, 2.6, 1, 8, cv2.LINE_AA)
    noisy = glyphs.copy()
    cv2.circle(noisy, (250, 22), 7, 1, -1)

    _labels, primitives = extract_primitives(noisy.astype(bool))
    grouped = group_primitives(noisy.astype(bool))

    assert len(primitives) >= 4
    assert grouped is not None
    assert not grouped[22, 250]
    assert np.count_nonzero(grouped & glyphs.astype(bool)) / np.count_nonzero(glyphs) > 0.95
