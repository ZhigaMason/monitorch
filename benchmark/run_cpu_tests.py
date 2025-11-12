import pandas as pd
import tqdm
import torch.nn as nn
from benchmark_utils import benchmark_monitorch_lens
from monitorch.lens import (
    LossMetrics,
    OutputActivation,
    ParameterGradientGeometry,
    ParameterGradientActivation,
    OutputNorm,
    ParameterNorm
)


if __name__ == "__main__":

    loss_fn = nn.CrossEntropyLoss()

    KWARGS = [
        dict(
            lens_list=[],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[LossMetrics(loss_fn=loss_fn)],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=loss_fn
        ),
        dict(
            lens_list=[OutputActivation()],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[ParameterGradientGeometry()],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[ParameterGradientActivation()],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[OutputNorm()],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[ParameterNorm()],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[OutputActivation(inplace=False)],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[ParameterGradientGeometry(inplace=False)],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[ParameterGradientActivation(inplace=False)],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[OutputNorm(inplace=False)],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
        dict(
            lens_list=[ParameterNorm(inplace=False)],
            inspector_kwargs=dict(visualizer='print'),
            loss_fn=nn.CrossEntropyLoss()
        ),
    ]

    results = []

    for kwargs in tqdm.tqdm(KWARGS):
        res = benchmark_monitorch_lens(**kwargs)
        results.append(kwargs | res)

    df = pd.DataFrame(results)
    df.to_csv("benchmark/results/.csv")
