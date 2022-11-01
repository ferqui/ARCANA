import torch
import torch.nn as nn
from typing import Optional, Callable, Sequence
from dynapsetorch.model import LIF, AdexLIF, ADM
from dynapsetorch.surrogate import fast_sigmoid


class DelayChain(nn.Module):
    """Delay chain network"""

    def __init__(self, n_channels: int, n_pool: int, n_out: int):
        super(DelayChain, self).__init__()

        self.n_pool = n_pool
        self.n_channels = n_channels
        self.adm_encoder = ADM(n_channels, 1.0, 1.0, 0)
        self.pool_layer = nn.ModuleList()
        for _ in range(n_pool):
            pool = AdexLIF(n_channels * 2, [0, n_channels * 2, 0, 0])
            pool.weight_ampa.data *= torch.eye(n_channels * 2)
            self.pool_layer.append(pool)
        self.readout = AdexLIF(n_out, [0, n_pool * n_channels * 2, 0, 0])

    def reset(self):
        self.adm_encoder.reset()
        for layer in self.pool_layer:
            layer.reset()
        self.readout.reset()

    def forward(self, input):
        in_spikes, _, _ = self.adm_encoder(input)
        out_spikes = []
        for pool in self.pool_layer:
            if pool.state is None:
                pool.state = pool.init_state(in_spikes)
            s_o = pool(input_ampa=in_spikes)
            out_spikes.append(s_o)
            in_spikes = s_o
        pool_spikes = torch.stack(out_spikes, dim=1)
        if self.readout.state is None:
            self.readout.state = self.readout.init_state(pool_spikes)
        ro_spikes = self.readout(
            input_ampa=pool_spikes.view(-1, self.n_pool * self.n_channels * 2)
        )
        return ro_spikes, pool_spikes


class DelayChainLIF(nn.Module):
    """Delay chain network using LIF neurons"""

    def __init__(
        self,
        n_channels: int,
        n_pool: int,
        n_out: int,
        thr: float = 1.0,
        tau: float = 20.0,
        dt: float = 1.0,
        activation_fn: torch.autograd.Function = fast_sigmoid,
    ):
        super(DelayChainLIF, self).__init__()

        self.n_pool = n_pool
        self.n_channels = n_channels
        self.adm_encoder = ADM(n_channels, 1.0, 1.0, 0)
        self.pool_layer = nn.ModuleList()
        for _ in range(n_pool):
            pool = LIF(
                n_channels * 2,
                n_channels * 2,
                thr=thr,
                tau=tau,
                dt=dt,
                activation_fn=activation_fn,
            )
            pool.base_layer.weight.data = torch.eye(n_channels * 2) * 1
            self.pool_layer.append(pool)
        self.readout = LIF(
            n_pool * n_channels * 2,
            n_out,
            thr=thr,
            tau=tau,
            dt=dt,
            activation_fn=activation_fn,
        )

    def reset(self):
        self.adm_encoder.reset()
        for layer in self.pool_layer:
            layer.reset()
        self.readout.reset()

    def forward(self, input):
        in_spikes, _, _ = self.adm_encoder(input)
        out_spikes = []
        for pool in self.pool_layer:
            if pool.state is None:
                pool.init_state(in_spikes)
            s_o = pool(in_spikes)
            out_spikes.append(s_o)
            in_spikes = s_o
        pool_spikes = torch.stack(out_spikes, dim=1)
        if self.readout.state is None:
            self.readout.init_state(pool_spikes)
        ro_spikes = self.readout(
            pool_spikes.view(-1, self.n_pool * self.n_channels * 2)
        )
        return ro_spikes, pool_spikes


class EIBalancedNetwork(nn.Module):
    """EI-balanced network"""

    def __init__(
        self,
        n_in: int,
        n_class: int,
        ex_per_class: int = 1,
        activation_fn: torch.autograd.Function = fast_sigmoid,
    ):
        super(EIBalancedNetwork, self).__init__()

        self.n_in = n_in
        self.n_class = n_class
        self.ex_per_class = ex_per_class

        self.ex_layer = AdexLIF(
            ex_per_class * n_class, [n_in, 0, 0, n_class], activation_fn=activation_fn
        )
        self.in_layer = AdexLIF(
            n_class,
            [ex_per_class * n_class, 0, 0, n_class],
            activation_fn=activation_fn,
        )
        self.out_in = None

    def reset(self):
        self.ex_layer.reset()
        self.in_layer.reset()
        self.out_in = None

    def forward(self, input):
        if self.out_in is None:
            self.out_in = torch.zeros(input.shape[0], self.n_class, device=input.device)

        if self.ex_layer.state is None:
            self.ex_layer.state = self.ex_layer.init_state(input)
        out_ex = self.ex_layer(input_nmda=input, input_gaba_b=self.out_in)

        if self.out_in.state is None:
            self.out_in.state = self.out_in.init_state(input)
        self.out_in = self.in_layer(input_nmda=out_ex, input_gaba_b=self.out_in)

        return out_ex
