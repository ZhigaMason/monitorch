import gc

import pandas as pd
import torch
from torch import nn, optim
from torch.autograd.profiler_util import EventList
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision.models import vit_b_16

from monitorch.inspector import PyTorchInspector
from monitorch.visualizer import MatplotlibVisualizer


def benchmark(
    loss_fn = nn.CrossEntropyLoss(),
    lens_list : list = [],
    dataset = None,
    inspector_kwargs : dict = {},
    dev:str="cuda",
    num_classes:int=200,
    batch_size:int=32,
    num_batches:int=5,
    num_epochs:int=5,
    image_size:int=224,
    learning_rate:float=1e-4
):

    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    model = vit_b_16(weights=None, num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    events = []

    with profile(
        activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        acc_events=True,
        record_shapes=False,
        with_stack=False,
    ) as prof, record_function('mt::initialization'):
        inspector = None
        if lens_list:
            inspector = PyTorchInspector(lenses=lens_list, module=model, **inspector_kwargs)
            torch.cuda.synchronize(device)

    events += [e for e in prof.events() if 'mt::' in e.key]

    gc.collect()

    for _ in range(num_epochs):
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            acc_events=True,
            record_shapes=False,
            with_stack=False,
            #schedule=my_schedule,
        ) as prof:
            with record_function('mt::training_epoch'):
                for step in range(num_batches):
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

                    with record_function('mt::training::forward_pass'):
                        out = model(x)
                        torch.cuda.synchronize(device)

                    with record_function('mt::training::loss_computation'):
                        loss = loss_fn(out, y)
                        torch.cuda.synchronize(device)

                    with record_function('mt::training::backward_pass'):
                        loss.backward()
                        torch.cuda.synchronize(device)

                    with record_function('mt::training::optimizer_steps'):
                        optimizer.step()
                        torch.cuda.synchronize(device)
            prof.step()

            if inspector:
                inspector.tick_epoch()
        es = prof.events()
        del prof
        gc.collect()
        events += [e for e in es if 'mt::' in e.key]

    with profile(
        activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        acc_events=True,
        record_shapes=False,
        with_stack=False,
    ) as prof, record_function('mt::matplotlib_visualization'):
        if inspector and isinstance(inspector.visualizer, MatplotlibVisualizer):
            fig = inspector.visualizer.show_fig()


    peak_gpu_mem = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0
    )
    torch.cuda.empty_cache()

    events += [e for e in prof.events() if 'mt::' in e.key]
    del prof
    gc.collect()
    events = EventList(events, profile_memory=True, use_device=True)

    events.sort(key=lambda x: x.key )

    res = pd.DataFrame(
        0.0,
        columns=['CPU TIME (ms)', 'GPU TIME (ms)', 'CPU MEMORY (MB)', 'GPU MEMORY (MB)', 'NCALLS'],
        index=list({e.key for e in events})
    )

    for e in events:
        cnt = e.count
        res.loc[e.key, 'CPU TIME (ms)'] += e.cpu_time / 1000
        res.loc[e.key, 'GPU TIME (ms)'] += e.device_time / 1000
        res.loc[e.key, 'CPU MEMORY (MB)'] += e.cpu_memory_usage / 1024**2
        res.loc[e.key, 'GPU MEMORY (MB)'] += e.device_memory_usage / 1024**2
        if e.cpu_time > 0: # count only CPU runs, as the ones in cuda are spawned from the same function call
            res.loc[e.key, 'NCALLS'] += cnt

    res = res.sort_index()
    return res, num_params, peak_gpu_mem
