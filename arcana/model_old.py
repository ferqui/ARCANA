import torch
import torch.nn as nn
from torch.nn import init
from typing import Callable, Optional, Sequence
from dynapsetorch.surrogate import fast_sigmoid, triangular, step

import numpy as np

from collections import namedtuple

amp = 1
mA = 1e-3
uA = 1e-6
nA = 1e-9
pA = 1e-12

volt = 1
mV = 1e-3
uV = 1e-6
nV = 1e-9

farad = 1
mF = 1e-3
uF = 1e-6
nF = 1e-9
pF = 1e-12

second = 1
ms = 1e-3
us = 1e-6
ns = 1e-9
ps = 1e-12

kappa_n = 0.75  # Subthreshold slope factor (n-type transistor)
kappa_p = 0.66  # Subthreshold slope factor (p-type transistor)
Ut = 25.0 * mV  # Thermal voltage
I0 = 0.5 * pA  # Dark current

relu = nn.ReLU()


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


class ADM(nn.Module):
    """Adaptive Delta Modulation (ADM) module
    Converts an analog signal into UP and DOWN spikes using the Adaptive Delta Modulation scheme.
    """

    def __init__(
        self,
        N: int,
        threshold_up: float,
        threshold_down: float,
        refractory: int,
        activation_fn: torch.autograd.Function = fast_sigmoid,
    ):
        super(ADM, self).__init__()

        self.activation_fn = activation_fn
        self.refractory = nn.Parameter(
            torch.tensor(refractory).float(), requires_grad=True
        )
        self.threshold = nn.Parameter(torch.tensor(threshold_up), requires_grad=True)
        # self.threshold_up = nn.Parameter(torch.tensor(threshold_up), requires_grad=True)
        # self.threshold_down = nn.Parameter(
        #     torch.tensor(threshold_down), requires_grad=True
        # )
        self.N = N

        self.reset()

    def reset(self):
        self.refrac = None
        self.DC_Voltage = None

    def reconstruct(self, spikes, initial_value=0):
        """Reconstruct an analog signal based on the UP and DOWN spikes produced by the ADM module.
        Everytime the algorithm receives an UP/DOWN spike, the reconstructed signal is increment/decrement by the UP/DOWN threshold amount.
        """
        reconstructed = torch.zeros(
            spikes.shape[0], spikes.shape[1], spikes.shape[2] // 2
        )
        reconstructed[:, 0, :] = initial_value
        for t in range(1, spikes.shape[1]):
            spikes_p = spikes[:, t, : -spikes.shape[-1] // 2]
            spikes_n = spikes[:, t, spikes.shape[-1] // 2 :]

            reconstructed[:, t] = (
                reconstructed[:, t - 1]
                + self.threshold * spikes_p
                - self.threshold * spikes_n
            )

        return reconstructed

    def forward(self, input_signal):
        if self.DC_Voltage is None:
            output = torch.zeros(
                input_signal.shape[0], self.N * 2, device=input_signal.device
            )
            output_p = torch.zeros_like(input_signal)
            output_n = torch.zeros_like(input_signal)
            self.refrac = torch.zeros_like(input_signal)
            self.DC_Voltage = input_signal
        else:
            output_p = (
                self.activation_fn(
                    (input_signal - (self.DC_Voltage.detach() + self.threshold))
                )
                * (self.refrac == 0).float()
            )
            self.refrac = output_p * self.refractory + (1 - output_p) * self.refrac

            output_n = (
                self.activation_fn(
                    ((self.DC_Voltage.detach() - self.threshold) - input_signal)
                )
                * (self.refrac == 0).float()
            )
            self.refrac = output_n * self.refractory + (1 - output_n) * self.refrac

            change_v = (self.refrac == 1).float()
            self.DC_Voltage = change_v * input_signal + (1 - change_v) * self.DC_Voltage

            output = torch.cat([output_p, output_n], dim=1)

            self.refrac = relu(self.refrac - 1)

        return output, output_p, output_n


class Rate(nn.Module):
    def __init__(self, dt=1.0):
        super(Rate, self).__init__()
        self.dt = dt

    def forward(self, rate):
        spike = torch.rand_like(rate) < rate * self.dt
        return spike.float()


class LIF(nn.Module):
    LIFState = namedtuple("LIFState", ["V", "I", "S"])

    def __init__(
        self,
        n_in: int,
        n_out: int,
        thr: float = 1.0,
        tau: float = 20.0,
        tau_I: float = 10.0,
        dt: float = 1.0,
        activation_fn: torch.autograd.Function = fast_sigmoid,
    ):
        super(LIF, self).__init__()

        self.dt = dt
        self.n_in = n_in
        self.n_out = n_out

        self.activation_fn = activation_fn
        self.base_layer = nn.Linear(n_in, n_out, bias=False)

        # distribution = torch.distributions.gamma.Gamma(3, 3 / tau)
        # tau = distribution.rsample((1, n_out)).clamp(3, 100)
        # self.register_buffer("alpha", torch.exp(-dt / tau).float())

        self.register_buffer("alpha", torch.tensor(np.exp(-dt / tau)).float())
        self.register_buffer("beta", torch.tensor(np.exp(-dt / tau_I)).float())
        self.register_buffer("thr", torch.tensor(thr).float())

        self.reset()

    def reset(self):
        self.state = None

    def init_state(self, input):
        self.state = self.LIFState(
            V=torch.zeros(input.shape[0], self.n_out, device=input.device),
            I=torch.zeros(input.shape[0], self.n_out, device=input.device),
            S=torch.zeros(input.shape[0], self.n_out, device=input.device),
        )
        return self.state

    def forward(self, input):
        if self.state is None:
            self.init_state(input)

        V = self.state.V
        I = self.state.I
        S = self.state.S

        I = self.beta * I + (1 - self.beta) * self.base_layer(input)
        V = (self.alpha * V + (1 - self.alpha) * I) * (1 - S.detach())
        S = self.activation_fn(V - self.thr)

        self.state = self.LIFState(V=V, I=I, S=S)

        return S


class AdexLIF(nn.Module):
    AdexLIFState = namedtuple(
        "AdexLIFState", ["Isoma_mem", "Iampa", "Igaba_b", "Isoma_ahp", "refractory"]
    )

    def __init__(
        self,
        n_in: int,
        n_out: int,
        tau_soma: float = 5.0,
        tau_ampa: float = 20.0,
        tau_gaba_b: float = 5.0,
        tau_ahp: float = 2.0,
        dt: float = 1e-3,
        activation_fn: torch.autograd.Function = fast_sigmoid,
    ):
        super(AdexLIF, self).__init__()

        self.dt = dt
        self.n_in = n_in
        self.n_out = n_out

        self.activation_fn = activation_fn

        self.I0 = 1e-3  # pA (scaled to mA for numerical stability)

        self.kappa = (kappa_n + kappa_p) / 2
        self.Ut = 25.0 * 1e-3  # Thermal voltage mV

        self.C_ampa = 2 * 1e-3  # pF (scaled to mF)
        self.C_gaba_b = 2 * 1e-3  # pF (scaled to mF)
        self.C_soma = 2 * 1e-3  # pF (scaled to mF)
        self.C_ahp = 4 * 1e-3  # pF (scaled to mF)

        self.tau_gaba_b = (
            self.C_gaba_b * self.Ut / (self.kappa * tau_gaba_b * self.I0)
        )  # ms
        self.tau_ampa = self.C_ampa * self.Ut / (self.kappa * tau_ampa * self.I0)  # ms
        self.tau_soma = self.C_soma * self.Ut / (self.kappa * tau_soma * self.I0)  # ms
        self.tau_ahp = self.C_ahp * self.Ut / (self.kappa * tau_ahp * self.I0)  # ms

        self.Isoma_dpi_tau = 5 * self.I0  # Leakage current
        self.soma_refP = 5 * 1e-3 / dt  # Refractory period (2 ms)

        # Reset and rest currents
        self.Isoma_reset = 1.2 * self.I0
        self.Vr = self.Isoma_dpi_tau + self.I0  # Isoma_dpi_tau + Igaba_a rest current

        # AdexLIF threshold
        self.Isoma_th = 2000 * self.I0
        self.Isoma_pfb_th = 1000 * self.I0

        #  SCALING FACTORS  #########################################################################################
        self.alpha_soma = 4  # Scaling factor equal to Ig/Itau
        self.alpha_gaba_b = 4  # Scaling factor equal to Ig/Itau
        self.alpha_ahp = 4  # Scaling factor equal to Ig/Itau
        self.alpha_ampa = 4  # Scaling factor equal to Ig/Itau

        # Ampa current
        self.ampa_gain = (
            self.alpha_ampa * 100 * self.I0
        )  # Scaling factor equal to Ig/Itau
        # Times the base synaptic weight current which can be scaled by the .weight parameter

        # GABA B
        self.gaba_b_gain = self.alpha_gaba_b * 100 * self.I0

        # AHP current
        self.ahp_gain = (
            self.alpha_ahp * 2 * self.I0
        )  # Scaling factor equal to Ig/Itau times Leakage current for spike-frequency adaptation
        self.ahp_jump = (
            1 * self.I0 * self.alpha_ahp
        )  # AHP jump height, on post times scaling factor equal to Ig/Itau

        # Positive feedback current
        self.Isoma_pfb_norm = 20 * self.I0  # Positive feedback normalization current
        self.Isoma_pfb_gain = 100 * self.I0  # Positive feedback gain

        self.weight_ampa = nn.Parameter(torch.rand(n_in, n_out), requires_grad=True)
        self.weight_gaba_b = nn.Parameter(torch.rand(n_in, n_out), requires_grad=True)

        self.reset()

    def reset(self):
        self.state = None

    def init_state(self, input):
        self.state = AdexLIF.AdexLIFState(
            Isoma_mem=torch.zeros(input.shape[0], self.n_out).to(input.device)
            + 1.1 * self.I0,
            Iampa=torch.zeros(input.shape[0], self.n_out).to(input.device) + self.I0,
            Igaba_b=torch.zeros(input.shape[0], self.n_out).to(input.device) + self.I0,
            Isoma_ahp=torch.zeros(input.shape[0], self.n_out).to(input.device),
            refractory=torch.zeros(input.shape[0], self.n_out).to(input.device),
        )
        return self.state

    def forward(self, input_ampa, input_gaba_b):
        if self.state is None:
            self.init_state(input_ampa)

        Isoma_mem = self.state.Isoma_mem
        Iampa = self.state.Iampa
        Igaba_b = self.state.Igaba_b
        Isoma_ahp = self.state.Isoma_ahp
        refractory = self.state.refractory

        # Positive feedback current
        Isoma_pfb = self.Isoma_pfb_gain / (
            1 + torch.exp(-(Isoma_mem - self.Isoma_pfb_th) / self.Isoma_pfb_norm)
        )

        # Detached Positive feedback and adaptation
        Isoma_pfb = Isoma_pfb.detach()
        Isoma_ahp = Isoma_ahp.detach()

        dAHP = (-self.ahp_gain - Isoma_ahp) / (
            self.tau_ahp * (1 + (self.ahp_gain / Isoma_ahp))
        )  # Adaptation current

        Iin = torch.clamp_min(Iampa - Igaba_b + self.I0, self.I0)
        Isoma_sum = self.Vr + Isoma_ahp - Isoma_pfb

        dIsoma_mem = (
            self.alpha_soma * (Iin - Isoma_sum)
            - Isoma_sum * Isoma_mem / self.Isoma_dpi_tau
        ) / (
            self.tau_soma
            * (1 + (self.alpha_soma * self.Isoma_dpi_tau / Isoma_mem.detach()))
        )

        dIampa = (self.I0 - Iampa) / self.tau_ampa
        Iampa += self.ampa_gain * input_ampa @ round(self.weight_ampa)

        dIgaba_b = (self.I0 - Igaba_b) / self.tau_gaba_b
        Igaba_b += self.gaba_b_gain * input_gaba_b @ round(self.weight_gaba_b)

        refractory = refractory - (refractory > 0).float()
        Isoma_mem += self.dt * dIsoma_mem * (refractory <= 0)
        Isoma_ahp += self.dt * dAHP
        Iampa += self.dt * dIampa
        Igaba_b += self.dt * dIgaba_b

        ## Fire
        S = fast_sigmoid(Isoma_mem - self.Isoma_th)
        refractory = refractory + (S * self.soma_refP).long()

        Isoma_ahp += self.ahp_jump * S
        Isoma_mem = self.Isoma_reset * S + Isoma_mem * (1 - S)
        Isoma_mem = torch.clamp_min(Isoma_mem, self.I0)

        self.state = AdexLIF.AdexLIFState(
            Isoma_mem=Isoma_mem,
            Iampa=Iampa,
            Igaba_b=Igaba_b,
            Isoma_ahp=Isoma_ahp,
            refractory=refractory,
        )

        return S


class AdexLIFfull(nn.Module):
    AdexLIFState = namedtuple(
        "AdexLIFState",
        [
            "Isoma_mem",
            "Isoma_ahp",
            "refractory",
            "Inmda",
            "Iampa",
            "Igaba_a",
            "Igaba_b",
        ],
    )

    def __init__(
        self,
        num_neurons: int = 1,
        input_per_synapse: Sequence[int] = [1, 1, 1, 1],
        activation_fn: torch.autograd.Function = fast_sigmoid,
    ):
        super(AdexLIFfull, self).__init__()

        self.num_neurons = num_neurons
        self.activation_fn = activation_fn

        #  SUBSTRATE  #########################################################################################
        self.register_buffer(
            "kn", torch.tensor(kappa_n)
        )  # Subthreshold slope factor for nFETs
        self.register_buffer(
            "kp", torch.tensor(kappa_p)
        )  # Subthreshold slope factor for pFETs
        self.register_buffer("Ut", torch.tensor(Ut))  # Thermal voltage
        self.register_buffer("I0", torch.tensor(I0))  # Dark current

        #  SCALING FACTORS  #########################################################################################
        self.register_buffer(
            "alpha_soma", torch.tensor(4)
        )  # Scaling factor equal to Ig/Itau
        self.register_buffer(
            "alpha_ahp", torch.tensor(4)
        )  # Scaling factor equal to Ig/Itau
        self.register_buffer(
            "alpha_nmda", torch.tensor(4)
        )  # Scaling factor equal to Ig/Itau
        self.register_buffer(
            "alpha_ampa", torch.tensor(4)
        )  # Scaling factor equal to Ig/Itau
        self.register_buffer(
            "alpha_gaba_a", torch.tensor(4)
        )  # Scaling factor equal to Ig/Itau
        self.register_buffer(
            "alpha_gaba_b", torch.tensor(4)
        )  # Scaling factor equal to Ig/Itau

        #  Neuron parameters  ###############
        #  SOMA  ##############################################################################################
        self.Isoma_mem_init = 1.1 * self.I0
        self.register_buffer("Csoma_mem", torch.tensor(2 * pF))  # Membrane capacitance
        self.register_buffer(
            "Isoma_dpi_tau", torch.tensor(5 * self.I0)
        )  # Leakage current
        self.register_buffer(
            "Isoma_th", torch.tensor(2000 * self.I0)
        )  # Spiking threshold
        self.register_buffer(
            "Isoma_reset", torch.tensor(1.2 * self.I0)
        )  # Reset current
        self.register_buffer(
            "Isoma_const", torch.tensor(self.I0)
        )  # Additional input current similar to constant current injection
        self.register_buffer("soma_refP", torch.tensor(5 * ms))  # Refractory period

        #  ADAPTATION  ########################################################################################
        self.register_buffer(
            "Csoma_ahp", torch.tensor(4 * pF)
        )  # Spike-frequency adaptation capacitance
        self.register_buffer(
            "Isoma_ahp_tau", torch.tensor(2 * self.I0)
        )  # Leakage current for spike-frequency adaptation
        self.register_buffer("Isoma_ahp_g", torch.tensor(0))  # AHP gain current
        self.register_buffer(
            "Isoma_ahp_w", torch.tensor(1 * self.I0)
        )  # AHP jump height, on post

        #  POSITIVE FEEDBACK ##################################################################################
        self.register_buffer(
            "Isoma_pfb_gain", torch.tensor(100 * self.I0)
        )  # Positive feedback gain
        self.register_buffer(
            "Isoma_pfb_th", torch.tensor(1000 * self.I0)
        )  # Positive feedback activation threshold
        self.register_buffer(
            "Isoma_pfb_norm", torch.tensor(20 * self.I0)
        )  # Positive feedback normalization current

        # Synapse parameters ################
        # # SLOW_EXC, NMDA ########################################################################################
        self.Inmda_init = self.I0
        self.register_buffer("Cnmda", torch.tensor(2 * pF))  # Synapse's capacitance
        self.register_buffer(
            "Inmda_tau", torch.tensor(2 * self.I0)
        )  # Leakage current, i.e. how much current is constantly leaked away (time-constant)
        self.register_buffer(
            "Inmda_w0", self.mismatch(100)  # torch.tensor(100 * self.I0)
        )  # Base synaptic weight, to convert unitless weight (set in synapse) to current
        self.register_buffer(
            "Inmda_thr", torch.tensor(I0)
        )  # NMDA voltage-gating threshold

        # FAST_EXC, AMPA ########################################################################################
        self.Iampa_init = self.I0  # Output current initial value
        self.register_buffer(
            "Campa", torch.tensor(2 * pF)
        )  # Synaptic capacitance, fixed at layout time (see chip for details)
        self.register_buffer(
            "Iampa_tau", torch.tensor(20 * self.I0)
        )  # Synaptic time constant current, the time constant is inversely proportional to I_tau
        self.register_buffer(
            "Iampa_w0", self.mismatch(100)  # torch.tensor(100 * self.I0)
        )  # Base synaptic weight current which can be scaled by the .weight parameter

        # #INH, SLOW_INH, GABA_B, subtractive ##################################################################
        self.Igaba_b_init = self.I0  # Output current initial value
        self.register_buffer(
            "Cgaba_b", torch.tensor(2 * pF)
        )  # Synaptic capacitance, fixed at layout time (see chip for details)
        self.register_buffer(
            "Igaba_b_tau", torch.tensor(5 * self.I0)
        )  # Synaptic time constant current, the time constant is inversely proportional to I_tau
        self.register_buffer(
            "Igaba_b_w0", self.mismatch(100)  # torch.tensor(100 * self.I0)
        )  # Base synaptic weight current which can be scaled by the .weight parameter

        # #FAST_INH, GABA_A, shunting, a mixture of subtractive and divisive ############################################
        self.Igaba_a_init = self.I0  # Output current initial value
        self.register_buffer(
            "Cgaba_a", torch.tensor(2 * pF)
        )  # Synaptic capacitance, fixed at layout time (see chip for details)
        self.register_buffer(
            "Igaba_a_tau", torch.tensor(5 * self.I0)
        )  # Synaptic time constant current, the time constant is inversely proportional to I_tau
        self.register_buffer(
            "Igaba_a_w0", self.mismatch(100)  # torch.tensor(100 * self.I0)
        )  # Base synaptic weight current which can be scaled by the .weight parameter
        # ##################

        self.weight_nmda = torch.nn.Parameter(
            torch.rand(input_per_synapse[0], num_neurons), requires_grad=False
        )
        self.weight_ampa = torch.nn.Parameter(
            torch.rand(input_per_synapse[1], num_neurons), requires_grad=False
        )
        self.weight_gaba_a = torch.nn.Parameter(
            torch.rand(input_per_synapse[2], num_neurons), requires_grad=False
        )
        self.weight_gaba_b = torch.nn.Parameter(
            torch.rand(input_per_synapse[3], num_neurons), requires_grad=False
        )

        self.dt = 1 * ms

        self.reset()

    def mismatch(self, initial):
        return (
            torch.maximum(
                torch.tensor(I0),
                initial + initial * 0.1 * torch.rand(1, self.num_neurons),
            )
            * self.I0
        )

    def reset(self):
        self.state = None

    def init_state(self, input):
        ## Soma states
        Isoma_mem = torch.empty(input.shape[0], self.num_neurons, device=input.device)
        init.constant_(Isoma_mem, self.Isoma_mem_init)
        ## Synapses states
        Inmda = torch.empty(input.shape[0], self.num_neurons, device=input.device)
        init.constant_(Inmda, self.Inmda_init)

        Iampa = torch.empty(input.shape[0], self.num_neurons, device=input.device)
        init.constant_(Iampa, self.Iampa_init)

        Igaba_a = torch.empty(input.shape[0], self.num_neurons, device=input.device)
        init.constant_(Igaba_a, self.Igaba_a_init)

        Igaba_b = torch.empty(input.shape[0], self.num_neurons, device=input.device)
        init.constant_(Igaba_b, self.Igaba_b_init)

        self.state = self.AdexLIFState(
            Isoma_mem=Isoma_mem,
            Isoma_ahp=torch.zeros(
                input.shape[0], self.num_neurons, device=input.device
            ),
            refractory=torch.zeros(
                input.shape[0], self.num_neurons, device=input.device
            ),
            Inmda=Inmda,
            Iampa=Iampa,
            Igaba_a=Igaba_a,
            Igaba_b=Igaba_b,
        )
        return self.state

    def detach(self):
        for state in self.state:
            state._detach()

    def forward(
        self, input_nmda=None, input_ampa=None, input_gaba_a=None, input_gaba_b=None
    ):
        ##### GET STATES VALUES #####
        ## Soma states
        Isoma_mem = self.state.Isoma_mem
        Isoma_ahp = self.state.Isoma_ahp
        refractory = self.state.refractory

        ## Synapses states
        Inmda = self.state.Inmda
        Iampa = self.state.Iampa
        Igaba_a = self.state.Igaba_a
        Igaba_b = self.state.Igaba_b

        Isoma_mem_clip = torch.clip(Isoma_mem.clone(), self.I0, 1)

        kappa = (self.kn + self.kp) / 2

        ## Input calculation
        Inmda_dp = Inmda.clone() / (
            1 + self.Inmda_thr / Isoma_mem_clip
        )  # Voltage gating differential pair block
        Iin_clip = torch.clip(Inmda_dp + Iampa - Igaba_b + self.Isoma_const, self.I0, 1)

        ##### SOMA CALCULATION #####
        ## Isoma_sum components calculation
        low_current_mem = self.I0 * (Isoma_mem.detach() <= self.I0)
        Isoma_pfb = self.Isoma_pfb_gain / (
            1 + torch.exp(-(Isoma_mem - self.Isoma_pfb_th) / self.Isoma_pfb_norm)
        )
        Isoma_pfb_shunt = Isoma_pfb * (Isoma_mem.detach() > self.I0) + low_current_mem
        Isoma_ahp_shunt = (
            Isoma_ahp.clone() * (Isoma_mem.detach() > self.I0) + low_current_mem
        )
        Igaba_a_shunt = (
            Igaba_a.clone() * (Isoma_mem.detach() > self.I0) + low_current_mem
        )
        Isoma_dpi_tau_shunt = (
            self.Isoma_dpi_tau * (Isoma_mem.detach() > self.I0) + low_current_mem
        )
        Isoma_dpi_g_shunt = (
            self.alpha_soma * Isoma_dpi_tau_shunt * (Isoma_mem.detach() > self.I0)
            + low_current_mem
        )

        # Isoma_sum = Isoma_dpi_tau_shunt.detach() + Isoma_ahp_shunt.detach() - Isoma_pfb_shunt.detach() - low_current_mem
        Isoma_sum = (
            Isoma_dpi_tau_shunt.detach()
            + Isoma_ahp_shunt.detach()
            + Igaba_a_shunt.detach()
            - Isoma_pfb_shunt.detach()
            - low_current_mem
        )

        ## Adaptation current
        low_current_ahp = self.I0 * (Isoma_ahp.detach() <= self.I0)
        Isoma_ahp_tau_shunt = (
            self.Isoma_ahp_tau * (Isoma_ahp.detach() > self.I0) + low_current_ahp
        )
        Isoma_ahp_g_shunt = (
            self.alpha_ahp * Isoma_ahp_tau_shunt * (Isoma_ahp.detach() > self.I0)
            + low_current_ahp
        )
        tau_soma_ahp = (self.Csoma_ahp * self.Ut) / (kappa * Isoma_ahp_tau_shunt)
        dIsoma_ahp = (-Isoma_ahp_g_shunt - Isoma_ahp + 2 * low_current_ahp) / (
            tau_soma_ahp * (1 + (Isoma_ahp_g_shunt / Isoma_ahp_shunt))
        )  # Adaptation current

        ## Isoma calculation
        tau_soma = (self.Csoma_mem * self.Ut) / (kappa * Isoma_dpi_tau_shunt)
        dIsoma_mem = (
            self.alpha_soma * (Iin_clip - Isoma_sum)
            - (Isoma_sum - low_current_mem)
            * Isoma_mem_clip.detach()
            / Isoma_dpi_tau_shunt.detach()
        ) / (
            tau_soma.detach()
            * (1 + (Isoma_dpi_g_shunt.detach() / Isoma_mem_clip.detach()))
        )

        ##### NMDA #####
        low_current_nmda = self.I0 * (Inmda.detach() <= self.I0)
        Inmda_g = self.alpha_nmda * self.Inmda_tau
        Inmda_g_shunt = Inmda_g * (Inmda.detach() > self.I0) + low_current_nmda

        Inmda_tau_shunt = self.Inmda_tau * (Inmda.detach() > self.I0) + low_current_nmda
        tau_nmda = self.Cnmda * self.Ut / (kappa * Inmda_tau_shunt)

        dInmda = (-Inmda - Inmda_g_shunt + 2 * low_current_nmda) / (
            tau_nmda * ((Inmda_g_shunt / Inmda) + 1)
        )
        if input_nmda is not None:
            Inmda = Inmda + self.Inmda_w0 * self.alpha_nmda * (
                input_nmda @ round(self.weight_nmda)
            )

        #### AMPA ####
        low_current_ampa = self.I0 * (Iampa.detach() <= self.I0)
        Iampa_g = self.alpha_ampa * self.Iampa_tau
        Iampa_g_shunt = Iampa_g * (Iampa.detach() > self.I0) + low_current_ampa

        Iampa_tau_shunt = self.Iampa_tau * (Iampa.detach() > self.I0) + low_current_ampa
        tau_ampa = self.Campa * self.Ut / (kappa * Iampa_tau_shunt)

        dIampa = (-Iampa - Iampa_g_shunt + 2 * low_current_ampa) / (
            tau_ampa * ((Iampa_g_shunt / Iampa) + 1)
        )
        if input_ampa is not None:
            Iampa = Iampa + self.Iampa_w0 * self.alpha_ampa * (
                input_ampa @ round(self.weight_ampa)
            )

        #### GABA B - inh ####
        low_current_gaba_b = self.I0 * (Igaba_b.detach() <= self.I0)
        Igaba_b_g = (
            self.alpha_gaba_b * self.Igaba_b_tau
        )  # GABA B synapse gain expressed in terms of its tau current
        Igaba_b_g_shunt = (
            Igaba_b_g * (Igaba_b.detach() > self.I0) + low_current_gaba_b
        )  # Shunt g current if Igaba_b goes to self.I0

        Igaba_b_tau_shunt = (
            self.Igaba_b_tau * (Igaba_b.detach() > self.I0) + low_current_gaba_b
        )  # Shunt tau current if Iampa goes to self.I0
        tau_gaba_b = (
            self.Cgaba_b * self.Ut / (kappa * Igaba_b_tau_shunt)
        )  # Synaptic time-constant

        dIgaba_b = (-Igaba_b - Igaba_b_g_shunt + 2 * low_current_gaba_b) / (
            tau_gaba_b * ((Igaba_b_g_shunt / Igaba_b) + 1)
        )
        if input_gaba_b is not None:
            Igaba_b = Igaba_b + self.Igaba_b_w0 * self.alpha_gaba_b * (
                input_gaba_b @ round(self.weight_gaba_b)
            )

        #### # GABA A - shunt ####
        low_current_gaba_a = self.I0 * (Igaba_a.detach() <= self.I0)
        Igaba_a_g = (
            self.alpha_gaba_a * self.Igaba_a_tau
        )  # GABA A synapse gain expressed in terms of its tau current
        Igaba_a_g_shunt = (
            Igaba_a_g * (Igaba_a.detach() > self.I0) + low_current_gaba_a
        )  # Shunt g current if Igaba_a goes to self.I0

        Igaba_a_tau_shunt = (
            self.Igaba_a_tau * (Igaba_a.detach() > self.I0) + low_current_gaba_a
        )  # Shunt tau current if Iampa goes to self.I0
        tau_gaba_a = (
            self.Cgaba_a * self.Ut / (kappa * Igaba_a_tau_shunt)
        )  # Synaptic time-constant

        dIgaba_a = (-Igaba_a - Igaba_a_g_shunt + 2 * low_current_gaba_a) / (
            tau_gaba_a * ((Igaba_a_g_shunt / Igaba_a) + 1)
        )
        if input_gaba_a is not None:
            Igaba_a = Igaba_a + self.Igaba_a_w0 * self.alpha_gaba_a * (
                input_gaba_a @ round(self.weight_gaba_a)
            )

        ## Gradient updates
        refractory = refractory - (refractory > 0).float()
        Isoma_mem += self.dt * dIsoma_mem * (refractory <= 0)
        Isoma_ahp += self.dt * dIsoma_ahp
        Inmda += self.dt * dInmda
        Iampa += self.dt * dIampa
        Igaba_a += self.dt * dIgaba_a
        Igaba_b += self.dt * dIgaba_b

        ## Fire
        firing = self.activation_fn(Isoma_mem - self.Isoma_th)
        refractory = refractory + (firing * self.soma_refP / self.dt).long()

        Isoma_ahp = Isoma_ahp + (self.Isoma_ahp_w * self.alpha_ahp) * firing
        Isoma_mem = self.Isoma_reset * firing + Isoma_mem * (1 - firing)

        ## Save states
        self.state = self.AdexLIFState(
            Isoma_mem=Isoma_mem,
            Isoma_ahp=Isoma_ahp,
            refractory=refractory,
            Inmda=Inmda,
            Iampa=Iampa,
            Igaba_a=Igaba_a,
            Igaba_b=Igaba_b,
        )

        return firing
