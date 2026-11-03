import numpy as np
import pytest
from PIL import Image

from digits_starter.images import export_digit, image_features


def test_image_feature_shape_and_range(tmp_path):
    path = tmp_path / "white.png"
    Image.new("L", (32, 32), color=255).save(path)
    features = image_features(path)
    assert features.shape == (1, 64)
    np.testing.assert_allclose(features, 16)
    np.testing.assert_allclose(image_features(path, invert=True), 0)


def test_dataset_export_roundtrip(split, tmp_path):
    image = split.images[split.test_indices[0]]
    path = tmp_path / "digit.png"
    export_digit(image, path)
    np.testing.assert_allclose(image_features(path).reshape(8, 8), image, atol=16 / 255)
    with pytest.raises(FileExistsError):
        export_digit(image, path)

