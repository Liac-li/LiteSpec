"""Synchronized stage diagnostics for a configured number of optimizer windows."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import time

import torch


def timed(timer, name):
    return timer.measure(name) if timer is not None else nullcontext()


class StepTiming:
    def __init__(self, steps, device):
        self.steps = int(steps)
        self.device = torch.device(device)
        self.active = False

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def begin(self, optimizer_step):
        self.active = optimizer_step < self.steps
        if self.active:
            self.synchronize()
            self.seconds = defaultdict(float)
            self.calls = defaultdict(int)
            self.workload = defaultdict(list)
            self.started = time.perf_counter()

    @contextmanager
    def measure(self, name):
        if not self.active:
            yield
            return
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] += time.perf_counter() - started
            self.calls[name] += 1

    def record(self, name, value):
        if self.active:
            self.workload[name].append(float(value))

    def finish(self):
        if self.active:
            self.synchronize()
            self.seconds["window"] = time.perf_counter() - self.started

    def snapshot(self):
        if not self.active:
            return None
        result = dict(seconds=dict(self.seconds), calls=dict(self.calls), workload={
            name: dict(mean=sum(values) / len(values), max=max(values))
            for name, values in self.workload.items()
        })
        if self.device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
        return result
