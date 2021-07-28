import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler


def check(model, input_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with profiler.profile(wiht_stack=True, profile_memory=True) as prof:

        out = model(torch.randn(1, *input_size).to(device))
        print(prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cpu_time_total', row_limit=5))
