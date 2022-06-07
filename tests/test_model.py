import torch

import pytest

from yolov5_simple import YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x


# Disabled testing larger models until I find a way to speed this up
# @pytest.mark.parametrize('module', [YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x])
@pytest.mark.parametrize('module', [YOLOv5n, YOLOv5s])
@pytest.mark.parametrize('in_channels', [3, 10])
@pytest.mark.parametrize('size', [(640, 640)])
@pytest.mark.parametrize('n_classes', [80, 5])
def test_yolov5_random_data(module, in_channels, size, n_classes):
    batch_size = 12

    model = module(in_channels, n_classes)
    x = torch.rand(batch_size, in_channels, *size)

    model.eval()
    with torch.no_grad():
        out = model(x)

    assert len(out) == model.detect.anchors.shape[0]
    assert tuple(out[0].shape) == (batch_size, 3, 80, 80, n_classes + 5)
    assert tuple(out[1].shape) == (batch_size, 3, 40, 40, n_classes + 5)
    assert tuple(out[2].shape) == (batch_size, 3, 20, 20, n_classes + 5)

