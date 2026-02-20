import multiprocessing as mp
import os
import time

import pandas as pd
import psutil
import torch
from torch import nn, optim
from torchvision.models import vit_b_16
from tqdm import tqdm, trange

from monitorch.inspector import PyTorchInspector
from monitorch.lens import LossMetrics, OutputActivation, OutputNorm, ParameterGradientActivation, ParameterGradientGeometry, ParameterNorm
from monitorch.visualizer import MatplotlibVisualizer


def benchmark_monitorch_lens(
    lens_list : list,
    loss_fn,
    dataset = None,
    inspector_kwargs : dict = {},
    dev:str="cpu",
    num_classes:int=200,
    batch_size:int=32,
    num_batches:int=20,
    num_epochs:int=10,
    image_size:int=224,
    learning_rate:float=1e-4,
):
    process = psutil.Process(os.getpid())

    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    model = vit_b_16(weights=None, num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    if dataset is None:
        inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)
    else:
        data_iter = iter(dataset)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    cpu_mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    start_time = time.perf_counter()
    # --- Monitorch ---
    inspector = None
    if lens_list:
        inspector = PyTorchInspector(lenses=lens_list, module=model, **inspector_kwargs)

    # --- Reset memory stats ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- Measure before training ---

    # --- Training loop ---
    for epoch in trange(num_epochs):
        for step in num_batches:
            optimizer.zero_grad()

            if dataset is None:
                x, y = inputs, targets
            else:
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataset)
                    x, y = next(data_iter)
                x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        if inspector:
            inspector.tick_epoch()

    if inspector and isinstance(inspector.visualizer, MatplotlibVisualizer):
        fig = inspector.visualizer.show_fig()
        fig.savefig('benchmark/plots/' + time.asctime())

    # --- After training ---
    wall_time = time.perf_counter() - start_time
    cpu_mem_after = process.memory_info().rss / (1024 ** 2)
    cpu_mem_used = cpu_mem_after - cpu_mem_before

    peak_gpu_mem = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0
    )

    result = {
        "lenses": [type(l).__name__ for l in lens_list] if lens_list else ["None"],
        "wall_time_s": round(wall_time, 3),
        "cpu_mem_used_MB": round(cpu_mem_used, 2),
        "peak_cpu_mem_MB": round(cpu_mem_after, 2),
        "peak_gpu_mem_MB": round(peak_gpu_mem, 2),
        "num_params_M": round(num_params / 1e6, 2),
    }
    torch.cuda.empty_cache()
    return result


def run_dev_benchmark(dev, num_epochs):

    loss_fns = [nn.CrossEntropyLoss() for _ in range(3)]

    # Fields:
    #   lens_list
    #   loss_fn
    #   inspector_kwargs (visualizer)
    #   comment
    #   dev
    #   num_epochs=5

    # Stats:
    #   nruns = 4 + 5 * 3 *2 =34
    #   exp_runtime = 34 * 15m =  510m = ~9h
    KWARGS = \
    [
        dict(
            lens_list = [],
            loss_fn=nn.CrossEntropyLoss(),
            inspector_kwargs={},
            comment="baseline",
            dev=dev,
            num_epochs=num_epochs
        ),
        dict(
            lens_list=[LossMetrics(loss_fn=loss_fns[0])],
            loss_fn=loss_fns[0],
            inspector_kwargs={'visualizer' : 'print'},
            comment="LossMetrics(); visualizer=print",
            dev=dev,
            num_epochs=num_epochs
        ),
        dict(
            lens_list=[LossMetrics(loss_fn=loss_fns[1])],
            loss_fn=loss_fns[1],
            inspector_kwargs={'visualizer' : 'matplotlib'},
            comment="LossMetrics(); visualizer=matplotlib",
            dev=dev,
            num_epochs=num_epochs
        ),
        dict(
            lens_list=[LossMetrics(loss_fn=loss_fns[2])],
            loss_fn=loss_fns[2],
            inspector_kwargs={'visualizer' : 'tensorboard'},
            comment="LossMetrics(); visualizer=tensorboard",
            dev=dev,
            num_epochs=num_epochs
        ),
    ] + [
        dict(
            lens_list=[
                ParameterGradientGeometry(parameters=('weight', 'bias'), **lens_kwargs),
                ParameterGradientGeometry(parameters=('in_proj_weight', 'in_proj_bias'), **lens_kwargs),
            ],
            loss_fn=nn.CrossEntropyLoss(),
            inspector_kwargs=inspector_kwargs,
            dev=dev,
            num_epochs=num_epochs,
            comment=f"ParameterGradientGeometry(parameters=(weight, bias), {lens_kwargs_comment}),ParameterGradientGeometry(parameters=(in_proj_weight, in_proj_bias), {lens_kwargs_comment}); {inspector_comment}"
        )
        for lens_kwargs, lens_kwargs_comment in [
                ({'inplace' : True},  'inplace=True'),
                ({'inplace' : False}, 'inplace=False')
        ]
        for inspector_kwargs, inspector_comment in [
                ({'visualizer' : 'print'},       "visualizer='print'"),
                ({'visualizer' : 'matplotlib'},  "visualizer='matplotlib'"),
                ({'visualizer' : 'tensorboard'}, "visualizer='tensorboard'"),
        ]
    ] + [
        dict(
            lens_list = [lens(**lens_kwargs)],
            loss_fn=nn.CrossEntropyLoss(),
            inspector_kwargs=inspector_kwargs,
            comment=f"{lens.__name__}({lens_kwargs_comment}); {inspector_comment}",
            dev=dev,
            num_epochs=num_epochs
        )
        for lens in [
                OutputActivation,
                ParameterGradientActivation,
                OutputNorm,
                ParameterNorm,
        ]
        for lens_kwargs, lens_kwargs_comment in [
                ({'inplace' : 'True'},  'inplace=True'),
                ({'inplace' : 'False'}, 'inplace=False')
        ]
        for inspector_kwargs, inspector_comment in [
                ({'visualizer' : 'print'},       "visualizer='print'"),
                ({'visualizer' : 'matplotlib'},  "visualizer='matplotlib'"),
                ({'visualizer' : 'tensorboard'}, "visualizer='tensorboard'"),
        ]
    ]

    results = []

    if not os.path.exists("benchmark/"):
        os.mkdir("benchmark/")

    if not os.path.exists("benchmark/plots/"):
        os.mkdir("benchmark/plots/")

    if not os.path.exists("benchmark/results/"):
        os.mkdir("benchmark/results/")

    ctx = mp.get_context("spawn")
    for i, kwargs in tqdm(enumerate(KWARGS[:2])):
        with ctx.Pool(1) as pool:
            comment = kwargs.pop('comment')
            res = pool.apply(benchmark_monitorch_lens, kwds=kwargs)
        results.append(kwargs | res | {'comment' : comment})
        df = pd.DataFrame(results)
        df.to_csv(f"benchmark/results/checkpoint_{i}.csv")

    df = pd.DataFrame(results)
    df.to_csv("benchmark/results/exhaustive_cpu.csv")
