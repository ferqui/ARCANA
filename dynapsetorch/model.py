import torch
import torch.nn as nn
from torch.nn import init
from typing import Callable, Optional, Sequence
from dynapsetorch.surrogate import fast_sigmoid, triangular, step

import numpy as np

from collections import namedtuple

class Round(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


round = Round.apply


class DPINeuron(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        # Neuron parameters
        Itau_mem: float = 1e-12,
        Igain_mem: float = 1e-12,
        Ith: float = 1e-12,
        Idc: float = 1e-12,
        refP: float = 0.0,
        # Positive feedback
        Ipfb_th: float = 1e-12,
        Ipfb_norm: float = 1e-12,
        ## AMPA parameters
        Itau_ampa: float = 1e-12,
        Igain_ampa: float = 1e-12,
        Iw_ampa: float = 1e-12,
        dt: float = 1e-3,
        surrogate_fn: Callable = fast_sigmoid,
        train_Itau_mem=False,
        train_Igain_mem=False,
        train_Idc=False,
        train_ampa=True,
        **kwargs
    ):
        super(DPINeuron, self).__init__(**kwargs)

        self.n_in = n_in
        self.n_out = n_out
        self.surrogate_fn = surrogate_fn
        self.MAX_FANIN = 64

        ## Constants
        self.dt = dt
        self.I0 = 5e-13  # Dark current
        self.Ut = 25e-3  # Thermal voltage
        self.kappa = (0.75 + 0.66) / 2  # Transistor slope factor
        self.Cmem = 3e-12  # Membrane capacitance
        self.Campa = 2e-12  # AMPA synapse capacitance
        self.Cshunt = 2e-12  # AMPA synapse capacitance

        ## Membrane parameters
        self.Itau_mem = Itau_mem
        self.Igain_mem = Igain_mem
        self.tau_mem = (
            (self.Ut / self.kappa) * self.Cmem
        ) / Itau_mem  # Soma time constant

        # Alpha and beta are trainable parameters that depends on leakage and gain current
        self.alpha = nn.Parameter(
            torch.tensor(Igain_mem / Itau_mem), requires_grad=train_Igain_mem
        )
        self.beta = nn.Parameter(
            torch.tensor(1 + self.I0 / Itau_mem), requires_grad=train_Itau_mem
        )

        ## Positive feedback current
        self.Ipfb_th = Ipfb_th
        self.Ipfb_norm = Ipfb_norm

        ## Other neuron parameters
        self.refP = refP
        self.Ith = Ith  # Firing threshold
        self.Idc = nn.Parameter(torch.tensor(Idc), requires_grad=train_Idc)  # Input DC

        ## AMPA
        self.train_ampa = train_ampa
        self.Itau_ampa = Itau_ampa
        self.Igain_ampa = Igain_ampa
        self.Iw_ampa = nn.Parameter(torch.tensor(Iw_ampa), train_ampa)
        self.Iw_ampa.register_hook(lambda grad: grad*1e-12)
        self.W_ampa = nn.Parameter(torch.empty(n_out, n_in), train_ampa)
        self.tau_ampa = (
            (self.Ut / self.kappa) * self.Campa
        ) / Itau_ampa  # AMPA time constant

        ## SHUNT
        # TODO: Change AMPA for shunt in parameters
        self.train_ampa = train_ampa
        self.Itau_shunt = Itau_ampa
        self.Igain_shunt = Igain_ampa
        self.Iw_shunt = nn.Parameter(torch.tensor(Iw_ampa), train_ampa)
        self.Iw_shunt.register_hook(lambda grad: grad*1e-12)
        self.W_shunt = nn.Parameter(torch.empty(n_out, n_in), train_ampa)
        self.tau_shunt = (
            (self.Ut / self.kappa) * self.Cshunt
        ) / Itau_ampa  # AMPA time constant 

        ## Weights initialization
        nn.init.constant_(self.W_ampa, 1.0)
        nn.init.constant_(self.W_shunt, 1.0)
        self.state = None

    def initialize(self, X):
        Imem = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        Iampa = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        Ishunt = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        refractory = torch.zeros(X.shape[0], self.n_out, device=X.device)

        return (Imem, Iampa, Ishunt, refractory)

    def I2V(self, I):
        return (self.Ut / self.kappa) * torch.log(I / self.I0)

    def printParameters(self, optimizer, args, kwargs):
        print("************************")
        print("****** Post hook *******")
        print("************************")
        for n, p in self.named_parameters():
            print(n, p)
        print('Itau_mem Parameter containing:')
        print(self.Itau_mem)
        print('Igain_mem Parameter containing:')
        print(self.Igain_mem)
        print('tau_mem Parameter containing:')
        print(self.tau_mem)
        print("")

    def UpdateParams(self, optimizer, args, kwargs):
        self.Itau_mem = self.I0 / (self.beta - 1)
        self.Igain_mem = self.alpha * self.Itau_mem
        self.tau_mem = (self.Ut / self.kappa) * self.Cmem / self.Itau_mem
        
        self.Iw_ampa.data = torch.clamp_min(self.Iw_ampa.data, self.I0)
        self.Iw_shunt.data = torch.clamp_min(self.Iw_shunt.data, self.I0)

        self.W_ampa.data = torch.clamp_min(self.W_ampa.data, 0.0)
        self.W_shunt.data = torch.clamp_min(self.W_shunt.data, 0.0)

        # fanin = self.W_ampa.sum(dim=1) + self.W_shunt.sum(dim=1)
        # self.W_ampa.data = torch.clamp_max(fanin.repeat(self.n_in,1).T, self.MAX_FANIN)*self.W_ampa.data/fanin.repeat(self.n_in,1).T
        # self.W_shunt.data = torch.clamp_max(fanin.repeat(self.n_in,1).T, self.MAX_FANIN)*self.W_shunt.data/fanin.repeat(self.n_in,1).T

    def forward(self, X, state=None):
        if state is None:
            state = self.initialize(X)

        (Imem, Iampa, Ishunt, refractory) = state
        Iahp = self.I0
        Inmda = self.I0
        ########### SYNAPSE ###########
        numSynAmpa = torch.nn.functional.linear(X, round(self.W_ampa))
        numSynShunt = torch.nn.functional.linear(X, round(self.W_shunt))
        if self.training and self.train_ampa:
            numSynAmpa.register_hook(lambda grad: grad*1e12)
            numSynShunt.register_hook(lambda grad: grad*1e12)


        dIampa = -Iampa / self.tau_ampa
        Iampa = Iampa + (
            self.Igain_ampa / self.Itau_ampa
        ) * self.Iw_ampa * numSynAmpa

        dIshunt = -Ishunt / self.tau_shunt
        Ishunt = Ishunt + (
            self.Igain_shunt / self.Itau_shunt
        ) * self.Iw_shunt * numSynShunt

        ########### SOMA ###########
        ## Input current
        Iin = self.Idc + Iampa + Inmda - Ishunt
        Iin = Iin * (refractory <= 0)
        Iin = torch.clamp_min(Iin, self.I0)

        ## Positive feedback
        Ifb = (
            self.I0 ** (1 / (self.kappa + 1))
            * Imem ** (self.kappa / (self.kappa + 1))
            / (1 + torch.exp(-self.Ipfb_norm * (Imem - self.Ipfb_th)))
        )
        f_imem = (Ifb / self.Itau_mem) * (Imem + self.Igain_mem)

        ## Soma derivative
        dImem = (
            self.alpha * (Iin - self.Itau_mem - Iahp)
            - self.beta * Imem
            + f_imem.detach()
        ) / (self.tau_mem * (1 + self.Igain_mem / Imem))

        ########### GRADIENT UPDATE ###########
        Imem = Imem + dImem * self.dt
        Imem = torch.clamp_min(Imem, self.I0)

        Iampa = Iampa + dIampa * self.dt
        Iampa = torch.clamp_min(Iampa, self.I0)

        Iampa = Iampa + dIshunt * self.dt
        Iampa = torch.clamp_min(Iampa, self.I0)

        ########### SPIKE ###########
        spike = self.surrogate_fn(Imem - self.Ith)
        Imem = (1.0 - spike) * Imem + spike * self.I0

        refractory = refractory - self.dt
        refractory = torch.clamp_min(refractory, 0.0)
        refractory = (1.0 - spike) * refractory + spike * self.refP

        ########### SAVE STATE ###########
        state = (Imem, Iampa, Ishunt, refractory)

        return spike, state


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    def train(neuron, optimizer, epochs):
        loss_hist = []
        for _ in range(epochs):
            outAcum = 0.0
            state = None
            for t in range(1000):
                out, state = neuron(torch.zeros(1, 1), state)
                outAcum += out

            loss = (outAcum.mean() - torch.tensor(5.0)) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
        return loss_hist

    @torch.no_grad()
    def test(neuron):
        state = None
        totalImem = []
        totalVmem = []
        for t in range(1000):
            out, state = neuron(torch.zeros(1, 1), state)
            (Imem, Iampa, _) = state
            totalImem.append(Imem.numpy().item())
            totalVmem.append(neuron.I2V(Imem).numpy().item())
        return totalImem, totalVmem

    neuron = DPINeuron(
        1,
        1,
        Itau_mem=4e-12,
        Igain_mem=20e-12,
        Ith=0.012,
        Idc=60e-12,
        refP=0.0,
        Ipfb_th=20e-12,
        Ipfb_norm=2e9,
        Itau_ampa=4e-12,
        Igain_ampa=20e-12,
        Iw_ampa=4e-12,
        dt=1e-3,
    )
    optimizer = torch.optim.Adam(neuron.parameters(), lr=1e-3)
    optimizer.register_step_post_hook(neuron.UpdateParams)
    optimizer.register_step_post_hook(neuron.printParameters)
    print(optimizer)

    sns.set_theme(style="whitegrid")
    params = {
        'text.usetex': True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica"
    }
    plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    Imem, Vmem = test(neuron)
    sns.lineplot(ax=axs[0], x=np.arange(len(Vmem)), y=Vmem, label="Before optimization")
    
    epochs = 100
    loss = train(neuron, optimizer, epochs)
    Imem, Vmem = test(neuron)

    sns.lineplot(ax=axs[0], x=np.arange(len(Vmem)), y=Vmem, label="After optimization")
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Voltage (mV)')
    axs[0].set_title('Membrane potential')

    sns.lineplot(ax=axs[1], x=np.arange(epochs), y=loss)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title(f'Loss over {epochs} epochs')

    fig.suptitle(r"\huge Neuron parameters optimization"
                 "\n"
                 r"\normalsize Optimizing leakage and gain currents of the neuron in order to fire 5 spikes in 1 second with a DC input of 60$pA$")

    plt.subplots_adjust(top=0.85, hspace=0.5) 
    plt.savefig('optimization.png', bbox_inches='tight', transparent=False, dpi=300)
    # plt.show()
