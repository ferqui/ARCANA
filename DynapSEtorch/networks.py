import torch
import torch.nn as nn
from DynapSEtorch.model import AdexLIF, ADM


class DelayChain(nn.Module):
    """Delay chain network"""

    def __init__(self, n_channels, n_pool, n_out):
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
        for layer in self.pool_layer:
            layer.reset()
        self.readout.reset()

    def forward(self, input):
        in_spikes = self.adm_encoder(input)
        out_spikes = []
        for pool in self.pool_layer:
            s_o = pool(input_ampa=in_spikes)
            out_spikes.append(s_o)
            in_spikes = s_o
        pool_spikes = torch.stack(out_spikes, dim=1)
        ro_spikes = self.readout(
            pool_spikes.view(-1, self.n_pool * self.n_channels * 2)
        )
        return ro_spikes, pool_spikes


class EIBalancedNetwork(nn.Module):
    """EI-balanced network"""

    def __init__(self, n_in, n_class, ex_per_class=1):
        super(EIBalancedNetwork, self).__init__()

        self.n_in = n_in
        self.n_class = n_class
        self.ex_per_class = ex_per_class

        self.ex_layer = AdexLIF(ex_per_class * n_class, [n_in, 0, 0, n_class])
        self.in_layer = AdexLIF(n_class, [ex_per_class * n_class, 0, 0, n_class])
        self.out_in = None

    def reset(self):
        self.ex_layer.reset()
        self.in_layer.reset()
        self.out_in = None

    def forward(self, input):
        if self.out_in is None:
            self.out_in = torch.zeros(input.shape[0], self.n_class, device=input.device)

        out_ex = self.ex_layer(input_nmda=input, input_gaba_b=self.out_in)
        self.out_in = self.in_layer(input_nmda=out_ex, input_gaba_b=self.out_in)

        return out_ex