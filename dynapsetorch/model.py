import torch
import torch.nn as nn
from torch.nn import init
from typing import Callable, Optional, Sequence
from dynapsetorch.surrogate import fast_sigmoid, triangular, step

import numpy as np

from collections import namedtuple

class DPIneuron(nn.Module):
    DPIneuronState = namedtuple(
        "DPIneuronState",
        ["Imem", "Iahp", "timer_ref", "Iampa", "Inmda", "Ishunt", "Igaba"],
    )

    def __init__(
        self,
        n_in,
        n_out,
        Idc=0,
        t_ref=0,
        C_mem=3e-12,
        alpha=1.470e9,
        Itau_mem=4.25e-12,
        Igain_mem=59.65e-12,
        Iampa_w0=1e-12,
        Iampa_tau=5e-12,
        Iampa_g=10e-12,
        Campa=2e-12,
        dt=1e-3,
        surrogate_fn = fast_sigmoid
    ):
        super(DPIneuron, self).__init__()
        ## Constant values
        KAPPA_N = 0.75
        KAPPA_P = 0.66
        UT = 25.0 * 1e-3
        I0 = 0.5 * 1e-13
        KAPPA = (KAPPA_N + KAPPA_P) / 2
        # KAPPA_2 = KAPPA**2.0
        # KAPPA_prime = KAPPA_2 / (KAPPA + 1.0)

        self.dt = dt
        self.n_in = n_in
        self.n_out = n_out
        self.surrogate_fn = surrogate_fn
        self.register_buffer("I0", torch.tensor(I0))
        self.register_buffer("KAPPA", torch.tensor(KAPPA))
        self.register_buffer("UT", torch.tensor(UT))
        ## Neuron values
        self.register_buffer("Idc", torch.tensor(Idc))
        self.register_buffer("t_ref", torch.tensor(t_ref))
        self.register_buffer("C_mem", torch.tensor(C_mem))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("Itau_mem", torch.tensor(Itau_mem))
        self.register_buffer("Igain_mem", torch.tensor(Igain_mem))
        self.register_buffer(
            "tau_mem", self.C_mem * self.UT / (self.KAPPA * self.Itau_mem)
        )

        ## AMPA values
        self.register_buffer("Iampa_g", torch.tensor(Iampa_g))
        self.register_buffer("Iampa_w0", torch.tensor(Iampa_w0))
        self.register_buffer("Iampa_tau", torch.tensor(Iampa_tau))
        self.register_buffer("Campa", torch.tensor(Campa))
        self.register_buffer(
            "tau_ampa", self.Campa * self.UT / (self.KAPPA * self.Iampa_tau)
        )
        self.W_ampa = nn.Parameter(torch.empty(n_out, n_in))

        ## NMDA values

        ## SHUNT values

        ## GABA values
        nn.init.kaiming_uniform_(self.W_ampa)
        with torch.no_grad():
            self.W_ampa.clamp_min_(0.0)
        self.state = None

    def extra_repr(self):
        return ''
    
    def reset(self):
        self.state = None

    def initialize(self, input):
        self.state = self.DPIneuronState(Imem = torch.zeros(input.shape[0], self.n_out, device=input.device),
                                         Iahp = torch.zeros(input.shape[0], self.n_out, device=input.device),
                                         timer_ref = torch.zeros(input.shape[0], self.n_out, device=input.device),
                                         Iampa = torch.zeros(input.shape[0], self.n_out, device=input.device),
                                         Inmda = torch.zeros(input.shape[0], self.n_out, device=input.device),
                                         Ishunt = torch.zeros(input.shape[0], self.n_out, device=input.device),
                                         Igaba = torch.zeros(input.shape[0], self.n_out, device=input.device))

    def getImem(self):
        if self.state is not None:
            return self.state.Imem.clone()
        else:
            return None
        
    def getVmem(self):
        if self.state is not None:
            return (self.UT / self.KAPPA) * torch.log(self.state.Imem / self.I0)
        else:
            return None
        
    def getIampa(self):
        if self.state is not None:
            return self.state.Iampa.clone()
        else:
            return None

    def forward(self, input):
        if self.state is None:
            self.initialize(input)

        Imem = self.state.Imem
        Iahp = self.state.Iahp
        timer_ref = self.state.timer_ref
        Iampa = self.state.Iampa
        Inmda = self.state.Inmda
        Ishunt = self.state.Ishunt
        Igaba = self.state.Igaba

        ## Simulation
        # Calculate leakage current and neuron tau
        Ileak = self.Itau_mem + Iahp + Igaba
        # tau_mem = ((self.UT / self.KAPPA) * self.C_mem) / self.Itau_mem
        # tau_ampa = Campa * UT / (KAPPA * Iampa_tau)

        # Injection
        Iin = self.Idc + Iampa + Inmda.detach() - Ishunt.detach()
        Iin = Iin * (timer_ref <= 0)
        Iin = torch.max(Iin, self.I0)

        # Positive feedback
        Ifb = (
            self.I0 ** (1 / (self.KAPPA + 1))
            * Imem ** (self.KAPPA / (self.KAPPA + 1))
            / (1 + torch.exp(-self.alpha * (Imem - self.Igain_mem)))
        )
        f_imem = (Ifb / Ileak) * (Imem + self.Igain_mem)

        # Steady state current
        Imem_inf = (self.Igain_mem / self.Itau_mem) * (Iin - Ileak)

        dImem = (
            (Imem / (self.tau_mem * (Imem + self.Igain_mem)))
            * (Imem_inf + f_imem.detach() - Imem * (1 + Iahp.detach() / self.Itau_mem))
            * self.dt
        )

        ## Synapse calculation
        # InputCurrent = torch.nn.functional.linear(input, self.W_ampa, None)
        # InputCurrent = InputCurrent - InputCurrent.detach()
        isyn_inf = (self.Iampa_g / self.Iampa_tau) * self.Iampa_w0 * torch.nn.functional.linear(input, self.W_ampa, None)#(torch.nn.functional.linear(input, torch.round(self.W_ampa), None) + InputCurrent)

        ## Exponential charge, discharge positive feedback factor arrays
        f_charge = 1.0 - np.exp(-10e-6 / self.tau_ampa)
        f_discharge = np.exp(-self.dt / self.tau_ampa)

        ## DISCHARGE in any case
        Iampa = f_discharge * Iampa

        ## CHARGE if spike occurs -- UNDERSAMPLED -- dt >> t_pulse
        Iampa += f_charge * isyn_inf

        ## Gradient update
        Imem = Imem + dImem
        Imem = torch.max(Imem, self.I0)

        # Iampa= Iampa + dIampa
        Iampa = torch.max(Iampa, self.I0)

        spike = self.surrogate_fn(Imem-0.000150)  # 60.6932
        Imem = (1.0 - spike) * Imem + spike * self.I0

        # Set the refractrory timer
        timer_ref = timer_ref - self.dt
        timer_ref = torch.max(timer_ref, torch.tensor(0.0, device=input.device))
        timer_ref = (1.0 - spike) * timer_ref + spike * self.t_ref

        self.state = self.DPIneuronState(
            Imem=Imem,
            Iahp=Iahp,
            timer_ref=timer_ref,
            Iampa=Iampa,
            Inmda=Inmda,
            Ishunt=Ishunt,
            Igaba=Igaba,
        )

        return spike
