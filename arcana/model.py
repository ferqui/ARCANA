import torch
import torch.nn as nn
from torch.nn import init
from typing import Callable, Optional, Sequence
from arcana.surrogate import fast_sigmoid, triangular, step

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
        ## GABAa parameters
        Itau_shunt: float = 1e-12,
        Igain_shunt: float = 1e-12,
        Iw_shunt: float = 1e-12,
        dt: float = 1e-3,
        surrogate_fn: Callable = fast_sigmoid,
        train_Itau_mem=False,
        train_Igain_mem=False,
        train_Idc=False,
        train_ampa=True,
        train_shunt=True,
        **kwargs
    ):
        super(DPINeuron, self).__init__(**kwargs)

        self.n_in = n_in
        self.n_out = n_out
        self.surrogate_fn = surrogate_fn
        self.MAX_FANIN = 64

        ## Constants
        self.dt = dt
        self.I0 = 0.5e-13  # Dark current
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
        self.train_Igain_mem = train_Igain_mem
        self.alpha = nn.Parameter(
            torch.tensor(Igain_mem / Itau_mem), requires_grad=train_Igain_mem
        )
        self.train_Itau_mem = train_Itau_mem
        self.beta = nn.Parameter(
            torch.tensor(self.I0 / Itau_mem), requires_grad=train_Itau_mem
        )

        ## Positive feedback current
        self.Ipfb_th = Ipfb_th
        self.Ipfb_norm = Ipfb_norm

        ## Other neuron parameters
        self.refP = refP
        self.Ith = Ith  # Firing threshold
        self.Idc = nn.Parameter(torch.tensor(Idc), requires_grad=train_Idc)  # Input DC
        self.train_Idc = train_Idc
        if train_Idc:
            self.Idc.register_hook(lambda grad: grad*1e-12)

        ## AMPA
        self.train_ampa = train_ampa
        self.Itau_ampa = Itau_ampa
        self.Igain_ampa = Igain_ampa
        self.Iw_ampa = nn.Parameter(torch.tensor(Iw_ampa), requires_grad=train_ampa)
        if train_ampa:
            self.Iw_ampa.register_hook(lambda grad: grad*1e-12)
        self.W_ampa = nn.Parameter(torch.empty(n_out, n_in), requires_grad=train_ampa)
        self.tau_ampa = (
            (self.Ut / self.kappa) * self.Campa
        ) / Itau_ampa  # AMPA time constant

        ## SHUNT
        self.train_shunt = train_shunt
        self.Itau_shunt = Itau_shunt
        self.Igain_shunt = Igain_shunt
        self.Iw_shunt = nn.Parameter(torch.tensor(Iw_shunt), requires_grad=train_shunt)
        if train_shunt:
            self.Iw_shunt.register_hook(lambda grad: grad*1e-12)
        self.W_shunt = nn.Parameter(torch.empty(n_out, n_in), requires_grad=train_shunt)
        self.tau_shunt = (
            (self.Ut / self.kappa) * self.Cshunt
        ) / Itau_shunt  # AMPA time constant 

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

    @torch.no_grad()
    def setTau(self, tau):
        # self.tau_mem.data = torch.tensor(tau, device=self.tau_mem.device)
        self.tau_mem = tau
        self.Itau_mem = (self.Ut / self.kappa) * self.Cmem / tau
        # self.beta.data = torch.tensor(1 + self.I0 / self.Itau_mem)

    @torch.no_grad()
    def setItau(self, Itau):
        self.Itau_mem = Itau
        # self.tau_mem.data = torch.tensor((self.Ut / self.kappa) * self.Cmem / Itau, device=self.tau_mem.device)
        self.tau_mem = (self.Ut / self.kappa) * self.Cmem / Itau
        self.beta.data = torch.tensor(self.I0 / Itau, device=self.beta.device)

    def UpdateParams(self, optimizer, args, kwargs):
        self.Itau_mem =  self.I0 / (self.beta)
        self.Igain_mem = self.alpha * self.Itau_mem
        self.tau_mem = (self.Ut / self.kappa) * self.Cmem / self.Itau_mem

        # self.Igain_mem = self.alpha * self.Itau_mem
        # self.Igain_mem.data = self.Igain_mem.clamp_min(self.I0)
        # self.alpha.data = self.Igain_mem / self.Itau_mem
        
        # self.tau_mem = (self.Ut / self.kappa) * self.Cmem / self.Itau_mem
        
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
        Igaba = self.I0
        ########### SYNAPSE ###########
        numSynAmpa = torch.nn.functional.linear(X, round(self.W_ampa))
        numSynShunt = torch.nn.functional.linear(X, round(self.W_shunt))
        if self.training and self.train_ampa:
            numSynAmpa.register_hook(lambda grad: grad*1e10)

        if self.training and self.train_shunt:
            numSynShunt.register_hook(lambda grad: grad*1e10)


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

        # Imem_decay = self.beta * Imem
        # if self.training and self.train_Itau_mem:
        #     Imem_decay.register_hook(lambda grad: grad*1e12)

        ## Soma derivative
        # Ileak = self.Itau_mem + Iahp + Igaba
        # Imem_inf = (self.Igain_mem / self.Itau_mem) * (Iin - Ileak)
        # dImem = (Imem/(self.tau_mem * (Imem + self.Igain_mem))) * (Imem_inf + f_imem - Imem * (1 + Iahp / self.Itau_mem))
        dImem = (
            self.alpha * (Iin - self.Itau_mem - Iahp)
            - Imem - (Iahp/self.I0) * self.beta * Imem
            + f_imem.detach()
        ) / (self.tau_mem * (1 + self.Igain_mem / Imem))

        ########### GRADIENT UPDATE ###########
        Imem = Imem + dImem * self.dt
        Imem = torch.clamp_min(Imem, self.I0)

        Iampa = Iampa + dIampa * self.dt
        Iampa = torch.clamp_min(Iampa, self.I0)

        Ishunt = Ishunt + dIshunt * self.dt
        Ishunt = torch.clamp_min(Ishunt, self.I0)

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
    from tqdm import tqdm

    import pandas as pd
    df = pd.read_csv(f'231010/records/TEK{7:04d}.CSV', header=None)
    dt = float(df[1][1])
    data = df[4].to_numpy() - df[4].to_numpy()[-10:].mean()

    neuronCalib = DPINeuron(
        1,
        1,
        Itau_mem=4.1e-12,
        Igain_mem=500e-12,
        Ith=0.012,
        Idc=0.0e-12,
        refP=0.0,
        Ipfb_th=500e-12,
        Ipfb_norm=1.470e9,
        Itau_ampa=8e-12,
        Igain_ampa=6.5e-12,
        Iw_ampa=50e-12,
        Itau_shunt=4e-12,
        Igain_shunt=10e-12,
        Iw_shunt=400e-12,
        dt=dt,
        train_Igain_mem=False,
        train_Itau_mem=False,
        train_ampa=True,
        train_shunt=True,
    )
    torch.nn.init.constant_(neuronCalib.W_shunt, 0.0)

    totalImem = []
    totalIampa = []
    totalIshunt = []
    totalVmem = []
    with torch.no_grad():
        neuronCalib.eval()
        state = None
        for t in range(2500):
            if t == 500:
                Sin = torch.ones(1,1)
            else:
                Sin = torch.zeros(1,1)
            out, state = neuronCalib(Sin, state)
            (Imem, Iampa, Ishunt, _) = state
            totalImem.append(Imem.numpy()[0])
            totalIampa.append(Iampa.numpy()[0])
            totalIshunt.append(Ishunt.numpy()[0])
            totalVmem.append(neuronCalib.I2V(Imem).numpy()[0])

    plt.plot(np.array(totalImem))
    # plt.plot(data)
    plt.show()

    # exit()

    def train(neuron, optimizer, epochs):
        loss_hist = []
        Itau_hist = []
        Igain_hist = []
        Vmem_hist = []
        neuron.train()
        pbar = tqdm(range(epochs))
        for _ in pbar:
            outAcum = 0.0
            state = None
            totalVmem = []
            for t in range(2000):
                out, state = neuron(torch.zeros(1, 1), state)
                (Imem, _, _, _) = state
                outAcum += out
                
                totalVmem.append(neuron.I2V(Imem).detach().numpy().item())
            totalVmem = np.stack(totalVmem)

            loss = (outAcum.sum() - torch.tensor(5.0)) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            with torch.no_grad():
                Vmem_hist.append(totalVmem)
                Itau_hist.append(neuron.Itau_mem.numpy().item())
                Igain_hist.append(neuron.Igain_mem.numpy().item())
            pbar.set_postfix({'Loss': loss.item()})
        return loss_hist, Vmem_hist, Itau_hist, Igain_hist

    @torch.no_grad()
    def test(neuron):
        state = None
        totalImem = []
        totalVmem = []
        neuron.eval()
        for t in tqdm(range(2000)):
            out, state = neuron(torch.zeros(1, 1), state)
            (Imem, Iampa, _, _) = state
            totalImem.append(Imem.numpy().item())
            totalVmem.append(neuron.I2V(Imem).numpy().item())
        return totalImem, totalVmem

    neuron = DPINeuron(
        1,
        1,
        Itau_mem=4e-12,
        Igain_mem=20e-12,
        Ith=0.012,
        Idc=10e-12,
        refP=0.0,
        Ipfb_th=20e-12,
        Ipfb_norm=2e9,
        Itau_ampa=4e-12,
        Igain_ampa=20e-12,
        Iw_ampa=4e-12,
        dt=1e-3,
        train_Igain_mem=True,
        train_Itau_mem=True,
        train_Idc=False,
        train_ampa=False,
        train_shunt=False
    )
    optimizer = torch.optim.Adam(neuron.parameters(), lr=5e-3)
    optimizer.register_step_post_hook(neuron.UpdateParams)
    # optimizer.register_step_post_hook(neuron.printParameters)
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
    (loss, _, Itau, Igain) = train(neuron, optimizer, epochs)
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
                 r"\normalsize Optimizing leakage and gain currents of the neuron in order to fire 5 spikes in 2 seconds with a DC input of 10$pA$")

    plt.subplots_adjust(top=0.85, hspace=0.5) 
    plt.savefig('/home/ferqui/Work/dynapse/Experiments2/freqAdap/optimization.pdf', bbox_inches='tight', transparent=True, dpi=300)

    plt.figure(figsize=(5,3))
    sns.lineplot(np.array(Igain)/np.array(Itau), label=r'$\frac{I_{th}}{I_{\tau}}$')
    plt.xlabel('Epochs')
    plt.savefig('/home/ferqui/Work/dynapse/Experiments2/freqAdap/alpha.pdf', bbox_inches='tight', transparent=True, dpi=300)

    plt.figure(figsize=(5,3))
    sns.lineplot(np.array(Itau)*1e12, label=r'$I_{\tau}$')
    plt.ylabel(r'Current ($pA$)')
    plt.xlabel('Epochs')
    plt.savefig('/home/ferqui/Work/dynapse/Experiments2/freqAdap/Itau.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()
