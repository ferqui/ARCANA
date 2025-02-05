from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from arcana.surrogate import fast_sigmoid


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
SCALING = 1


class DPINeuron(nn.Module):
    I0: float = 0.5e-13 * SCALING  # Dark current
    UT: float = 25e-3  # Thermal voltage
    KAPPA: float = (0.75 + 0.66) / 2  # Transistor slope factor
    CMEM: float = 3e-12 * SCALING  # Membrane capacitance
    CAMPA: float = 2e-12 * SCALING  # AMPA synapse capacitance
    CNMDA: float = 2e-12 * SCALING  # AMPA synapse capacitance
    CGABA_A: float = 2e-12 * SCALING  # AMPA synapse capacitance
    CGABA_B: float = 2e-12 * SCALING  # AMPA synapse capacitance
    MAX_FANIN: float = 64  # Maximum number of input synapses per neuron

    def __init__(
        self,
        n_in: int,
        n_out: int,
        dt: float = 1e-3,
        surrogate_fn: Callable = fast_sigmoid,
        train_Itau_mem: bool = False,
        train_Igain_mem: bool = False,
        train_Idc: bool = False,
        train_ampa: bool = False,
        train_gabab: bool = False,
        **kwargs,
    ):
        """
        DPI neuron model used in Dynap-SE chip.

        Attributes
        ----------
        n_in: int
            Number of input synapses
        n_out: int
            Number of neuron in the layer
        dt: float
            Simulation timestep in seconds
        surrogate_fn: Callable
            Surrogate gradient function for spiking
        train_Itau_mem: bool
            Flag to train the membrane leakage current bias
        train_Igain_mem: bool
            Flag to train the membrane input gain bias
        train_Idc: bool
            Flag to train the input constant current
        train_ampa: bool
            Flag to train the ampa weight matrix
        train_gabab: bool
            Flag to train the gaba_b weight matrix
        """
        super(DPINeuron, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.surrogate_fn = surrogate_fn
        self.dt = dt

        # SOMA
        # Parameters
        self.Itau_mem = kwargs.get("Itau_mem", 5e-12) * SCALING
        self.Igain_mem = kwargs.get("Igain_mem", 20e-12) * SCALING

        # Alpha and beta are trainable parameters
        # that depends on leakage and gain current
        self.alpha = nn.Parameter(
            torch.tensor(self.Igain_mem / self.Itau_mem), requires_grad=train_Igain_mem
        )
        self.beta = nn.Parameter(
            torch.tensor(1 + DPINeuron.I0 / self.Itau_mem), requires_grad=train_Itau_mem
        )

        # Positive feedback current
        self.Ipfb_th = kwargs.get("Ipfb_th", 500.0e-12) * SCALING
        self.Ipfb_norm = kwargs.get("Ipfb_norm", 1.470e9) / SCALING

        # Other neuron parameters
        self.refP = kwargs.get("refractory", 0.0)
        self.Ith = kwargs.get("Ith", 2000.0e-12) * SCALING  # Firing threshold
        self.Idc = nn.Parameter(
            torch.tensor(kwargs.get("Idc", 1e-12) * SCALING), requires_grad=train_Idc
        )

        # AMPA
        self.train_ampa = train_ampa
        self.Itau_ampa = kwargs.get("Itau_ampa", 20e-12) * SCALING
        self.Igain_ampa = kwargs.get("Igain_ampa", 80e-12) * SCALING
        self.Iw_ampa = nn.Parameter(
            torch.tensor(kwargs.get("Iw_ampa", 80e-12) * SCALING),
            requires_grad=train_ampa,
        )
        if train_ampa:
            self.Iw_ampa.register_hook(lambda grad: grad * 1e-12)
        self.W_ampa = nn.Parameter(torch.empty(n_out, n_in), requires_grad=train_ampa)

        # NMDA
        # self.train_ampa = train_ampa
        self.Itau_nmda = kwargs.get("Itau_nmda", 20e-12) * SCALING
        self.Inmda_thr = kwargs.get("Inmda_thr", 5e-13) * SCALING
        self.Igain_nmda = kwargs.get("Igain_nmda", 80e-12) * SCALING
        self.Iw_nmda = torch.tensor(kwargs.get("Iw_nmda", 80e-12) * SCALING)
        self.W_nmda = nn.Parameter(torch.empty(n_out, n_in), requires_grad=train_ampa)

        # gabaa
        self.Itau_gabaa = kwargs.get("Itau_gabaa", 20e-12) * SCALING
        self.Igain_gabaa = kwargs.get("Igain_gabaa", 80e-12) * SCALING
        self.Iw_gabaa = torch.tensor(kwargs.get("Iw_gabaa", 80e-12) * SCALING)
        self.W_gabaa = nn.Parameter(torch.empty(n_out, n_in), requires_grad=train_ampa)

        # gabab
        self.train_gabab = train_gabab
        self.Itau_gabab = kwargs.get("Itau_gabab", 20e-12) * SCALING
        self.Igain_gabab = kwargs.get("Igain_gabab", 80e-12) * SCALING
        self.Iw_gabab = nn.Parameter(
            torch.tensor(kwargs.get("Iw_gabab", 80e-12) * SCALING),
            requires_grad=train_gabab,
        )
        if train_gabab:
            self.Iw_gabab.register_hook(lambda grad: grad * 1e-12)
        self.W_gabab = nn.Parameter(torch.empty(n_out, n_in), requires_grad=train_ampa)

        # Weights initialization
        nn.init.constant_(self.W_ampa, 1.0)
        nn.init.constant_(self.W_nmda, 0.0)
        nn.init.constant_(self.W_gabaa, 0.0)
        nn.init.constant_(self.W_gabab, 1.0)

        # Mismatch parameters
        self.register_buffer("_Idc_mismatch", torch.zeros(1, self.n_out))

        self.register_buffer("_Itau_mem_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Igain_mem_mismatch", torch.zeros(1, self.n_out))

        self.register_buffer("_Itau_ampa_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Igain_ampa_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Iw_ampa_mismatch", torch.zeros(1, self.n_out))

        self.register_buffer("_Itau_nmda_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Inmda_thr_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Igain_nmda_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Iw_nmda_mismatch", torch.zeros(1, self.n_out))

        self.register_buffer("_Itau_gabaa_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Igain_gabaa_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Iw_gabaa_mismatch", torch.zeros(1, self.n_out))

        self.register_buffer("_Itau_gabab_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Igain_gabab_mismatch", torch.zeros(1, self.n_out))
        self.register_buffer("_Iw_gabab_mismatch", torch.zeros(1, self.n_out))

        self.state = None

    def add_mismatch(self, param: str, mismatch: float = 0.1):
        try:
            mismatch_parameter = getattr(self, f"_{param}_mismatch")
            torch.nn.init.normal_(mismatch_parameter, mean=0.0, std=mismatch)

        except AttributeError:
            raise AttributeError(f"DPINeuron as not mismatch on attribute {param}")

    def initialize(self, X):
        Imem = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        Iampa = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        Inmda = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        Igabaa = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        Igabab = torch.zeros(X.shape[0], self.n_out, device=X.device) + self.I0
        refractory = torch.zeros(X.shape[0], self.n_out, device=X.device)

        return (Imem, Iampa, Inmda, Igabaa, Igabab, refractory)

    @staticmethod
    def I2V(current: float) -> float:
        return (DPINeuron.UT / DPINeuron.KAPPA) * torch.log(current / DPINeuron.I0)

    @staticmethod
    def V2I(voltage: float) -> float:
        return DPINeuron.I0 * np.exp(voltage * DPINeuron.KAPPA / DPINeuron.UT)

    def UpdateParams(self, optimizer, args, kwargs):
        self.Itau_mem = DPINeuron.I0 / (self.beta - 1)
        self.Igain_mem = self.alpha * self.Itau_mem
        self.tau_mem = (DPINeuron.UT / DPINeuron.KAPPA) * DPINeuron.CMEM / self.Itau_mem

        self.Iw_ampa.data = torch.clamp_min(self.Iw_ampa.data, self.I0)
        self.Iw_nmda.data = torch.clamp_min(self.Iw_nmda.data, self.I0)
        self.Iw_gabaa.data = torch.clamp_min(self.Iw_gabaa.data, self.I0)
        self.Iw_gabab.data = torch.clamp_min(self.Iw_gabab.data, self.I0)

        self.W_ampa.data = torch.clamp_min(self.W_ampa.data, 0.0)
        self.W_nmda.data = torch.clamp_min(self.W_nmda.data, 0.0)
        self.W_gabaa.data = torch.clamp_min(self.W_gabaa.data, 0.0)
        self.W_gabab.data = torch.clamp_min(self.W_gabab.data, 0.0)

    def forward(self, X, state=None):
        if state is None:
            state = self.initialize(X)

        (Imem, Iampa, Inmda, Igabaa, Igabab, refractory) = state
        Iahp = DPINeuron.I0

        # Apply mismatch
        Idc = torch.clamp_min(self.Idc * (1 + self._Idc_mismatch), DPINeuron.I0)

        Itau_mem = torch.clamp_min(
            self.Itau_mem * (1 + self._Itau_mem_mismatch), DPINeuron.I0
        )
        Igain_mem = torch.clamp_min(
            self.Igain_mem * (1 + self._Igain_mem_mismatch), DPINeuron.I0
        )

        Itau_ampa = torch.clamp_min(
            self.Itau_ampa * (1 + self._Itau_ampa_mismatch), DPINeuron.I0
        )
        Igain_ampa = torch.clamp_min(
            self.Igain_ampa * (1 + self._Igain_ampa_mismatch), DPINeuron.I0
        )
        Iw_ampa = torch.clamp_min(
            self.Iw_ampa * (1 + self._Iw_ampa_mismatch), DPINeuron.I0
        )

        Inmda_thr = torch.clamp_min(
            self.Inmda_thr * (1 + self._Inmda_thr_mismatch), DPINeuron.I0
        )
        Itau_nmda = torch.clamp_min(
            self.Itau_nmda * (1 + self._Itau_nmda_mismatch), DPINeuron.I0
        )
        Igain_nmda = torch.clamp_min(
            self.Igain_nmda * (1 + self._Igain_nmda_mismatch), DPINeuron.I0
        )
        Iw_nmda = torch.clamp_min(
            self.Iw_nmda * (1 + self._Iw_nmda_mismatch), DPINeuron.I0
        )

        Itau_gabaa = torch.clamp_min(
            self.Itau_gabaa * (1 + self._Itau_gabaa_mismatch), DPINeuron.I0
        )
        Igain_gabaa = torch.clamp_min(
            self.Igain_gabaa * (1 + self._Igain_gabaa_mismatch), DPINeuron.I0
        )
        Iw_gabaa = torch.clamp_min(
            self.Iw_gabaa * (1 + self._Iw_gabaa_mismatch), DPINeuron.I0
        )

        Itau_gabab = torch.clamp_min(
            self.Itau_gabab * (1 + self._Itau_gabab_mismatch), DPINeuron.I0
        )
        Igain_gabab = torch.clamp_min(
            self.Igain_gabab * (1 + self._Igain_gabab_mismatch), DPINeuron.I0
        )
        Iw_gabab = torch.clamp_min(
            self.Iw_gabab * (1 + self._Iw_gabab_mismatch), DPINeuron.I0
        )

        alpha = (
            self.alpha * (1 + self._Igain_mem_mismatch) / (1 + self._Itau_mem_mismatch)
        )
        beta_mismatch = (
            DPINeuron.I0 + self.Itau_mem * (1 + self._Itau_mem_mismatch)
        ) / ((self._Itau_mem_mismatch + 1) * (DPINeuron.I0 + self.Itau_mem))
        beta = self.beta * beta_mismatch.detach()

        # Calculate tau
        tau_mem = (
            (DPINeuron.UT / DPINeuron.KAPPA) * DPINeuron.CMEM
        ) / Itau_mem  # AMPA time constant

        tau_ampa = (
            (DPINeuron.UT / DPINeuron.KAPPA) * DPINeuron.CAMPA
        ) / Itau_ampa  # AMPA time constant

        tau_nmda = (
            (DPINeuron.UT / DPINeuron.KAPPA) * DPINeuron.CNMDA
        ) / Itau_nmda  # AMPA time constant

        tau_gabaa = (
            (DPINeuron.UT / DPINeuron.KAPPA) * DPINeuron.CGABA_A
        ) / Itau_gabaa  # AMPA time constant

        tau_gabab = (
            (DPINeuron.UT / DPINeuron.KAPPA) * DPINeuron.CGABA_B
        ) / Itau_gabab  # AMPA time constant

        # Synapse
        numSynAmpa = torch.nn.functional.linear(X, round(self.W_ampa))
        numSynNmda = torch.nn.functional.linear(X, round(self.W_nmda))
        numSynGabaa = torch.nn.functional.linear(X, round(self.W_gabaa))
        numSynGabab = torch.nn.functional.linear(X, round(self.W_gabab))
        if self.training and self.train_ampa:
            numSynAmpa.register_hook(lambda grad: grad * 1e10)
            numSynNmda.register_hook(lambda grad: grad * 1e10)
            numSynGabaa.register_hook(lambda grad: grad * 1e10)
            numSynGabab.register_hook(lambda grad: grad * 1e10)

        # Synapse derivatives
        dIampa = -Iampa / tau_ampa
        Iampa = Iampa + (Igain_ampa / Itau_ampa) * Iw_ampa * numSynAmpa

        dInmda = -Inmda / tau_nmda
        Inmda = Inmda + (Igain_nmda / Itau_nmda) * Iw_nmda * numSynNmda

        dIgabaa = -Igabaa / tau_gabaa
        Igabaa = Igabaa + (Igain_gabaa / Itau_gabaa) * Iw_gabaa * numSynGabaa

        dIgabab = -Igabab / tau_gabab
        Igabab = Igabab + (Igain_gabab / Itau_gabab) * Iw_gabab * numSynGabab

        # Soma
        # Input current
        Inmda_dp = Inmda / (1 + Inmda_thr / Imem)
        Iin = Idc + Iampa + Inmda_dp.detach() - Igabab
        Iin = Iin * (refractory <= 0)
        Iin = torch.clamp_min(Iin, DPINeuron.I0)

        # Positive feedback
        Ifb = (
            DPINeuron.I0 ** (1 / (DPINeuron.KAPPA + 1))
            * Imem ** (DPINeuron.KAPPA / (DPINeuron.KAPPA + 1))
            / (1 + torch.exp(-self.Ipfb_norm * (Imem - self.Ipfb_th)))
        )
        f_imem = (Ifb / Itau_mem) * (Imem + Igain_mem)

        # Soma derivative
        dImem = (
            alpha * (Iin - Itau_mem - Iahp - Igabaa.detach())
            - beta * Imem
            - ((Igabaa / Itau_mem) * Imem).detach()
            + f_imem.detach()
        ) / (tau_mem * (1 + Igain_mem / Imem))

        # Gradient update
        Imem = Imem + dImem * self.dt
        Imem = torch.clamp_min(Imem, DPINeuron.I0)

        Iampa = Iampa + dIampa * self.dt
        Iampa = torch.clamp_min(Iampa, DPINeuron.I0)

        Inmda = Inmda + dInmda * self.dt
        Inmda = torch.clamp_min(Inmda, DPINeuron.I0)

        Igabaa = Igabaa + dIgabaa * self.dt
        Igabaa = torch.clamp_min(Igabaa, DPINeuron.I0)

        Igabab = Igabab + dIgabab * self.dt
        Igabab = torch.clamp_min(Igabab, DPINeuron.I0)

        # Spike
        spike = self.surrogate_fn(Imem - self.Ith)
        Imem = (1.0 - spike) * Imem + spike * DPINeuron.I0

        refractory = refractory - self.dt
        refractory = torch.clamp_min(refractory, 0.0)
        refractory = (1.0 - spike) * refractory + spike * self.refP

        # Save state
        state = (Imem, Iampa, Inmda, Igabaa, Igabab, refractory)

        return spike, state
