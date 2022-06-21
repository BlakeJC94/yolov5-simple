from pathlib import Path
from time import time
from datetime import timedelta

import numpy as np
import torch
import humanize

from yolov5_simple import YOLOv5x
# from yolov5.utils.general import check_yaml
from yolov5.models.yolo import Model


def time_trials(n_repeats):

    def profiler(func):

        def wrapper(*args, **kwargs):
            results = np.zeros(n_repeats)

            for i in range(n_repeats):
                print(f"Starting trial {i} for {func.__name__}..")
                t1 = time()
                func(*args, **kwargs)
                duration = time() - t1
                results[i] = duration

            out = timedelta(seconds=results.mean())
            out = humanize.precisedelta(out, minimum_unit='milliseconds')
            print(f"Finished! Mean runtime was {out}")
            return

        return wrapper

    return profiler


@time_trials(n_repeats=3)
def benchmark_model(model, x):
    _ = model(x)


def get_original_model(cfg='yolov5s.yaml'):
    cfg = Path(__file__).parent / f'yolov5/models/{cfg}'
    # cfg = check_yaml(cfg)  # check YAML
    model = Model(cfg)
    return model

if __name__ == '__main__':
    batch_size = 12
    in_channels = 3
    size = (640, 640)
    n_classes = 80
    device = 'cpu'

    x = torch.rand(batch_size, in_channels, *size).to(device)

    print('BENCHMARK SIMPLE')
    model_simple = YOLOv5x(in_channels, n_classes)
    model_simple.to(device)
    model_simple.eval()
    with torch.no_grad():
        benchmark_model(model_simple, x)

    print('BENCHMARK ORIGINAL')
    model_original = get_original_model('yolov5x.yaml')
    model_original.to(device).eval()
    with torch.no_grad():
        benchmark_model(model_original, x)

    # NOTES:
    # - CUDA support doesn't work out of the box on 3070Ti :(
    # - Running both models on the CPU gives a similar results though:
    #    BENCHMARK SIMPLE
    #    Starting trial 0 for benchmark_model..
    #    Starting trial 1 for benchmark_model..
    #    Starting trial 2 for benchmark_model..
    #    Finished! Mean runtime was 6 seconds and 213.81 milliseconds
    #    BENCHMARK ORIGINAL
    #    Starting trial 0 for benchmark_model..
    #    Starting trial 1 for benchmark_model..
    #    Starting trial 2 for benchmark_model..
    #    Finished! Mean runtime was 5 seconds and 935.47 milliseconds
