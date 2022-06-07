from time import time, sleep
from datetime import timedelta

import numpy as np
import torch
import humanize

from yolov5_simple import YOLOv5x


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



if __name__ == '__main__':
    batch_size = 12
    in_channels = 3
    size = (640, 640)
    n_classes = 80

    model_simple = YOLOv5x(in_channels, n_classes)
    model_simple.eval()

    x = torch.rand(batch_size, in_channels, *size)

    with torch.no_grad():
        benchmark_model(model_simple, x)
