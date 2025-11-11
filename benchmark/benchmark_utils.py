import time
import os
import psutil
import torch
import tqdm
from torch import nn, optim
from torchvision.models import vit_b_16
from monitorch.inspector import PyTorchInspector

def benchmark_monitorch_lens(
    lens_list : list,
    loss_fn,
    dataset = None,
    inspector_kwargs : dict = {},
    dev:str="cpu",
    num_classes:int=200,
    batch_size:int=32,
    num_batches:int=50,
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
    for step in tqdm.trange(num_batches):
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

    del model
    torch.cuda.empty_cache()
    return result
