import torch
import torch.nn as nn
from torch.nn import init
from DynapSEtorch.surrogate import fast_sigmoid

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

# amp = 1e3
# mA = 1
# uA = 1e-3
# nA = 1e-6
# pA = 1e-9

# volt = 1e3
# mV = 1
# uV = 1e-3
# nV = 1e-6

# farad = 1e3
# mF = 1
# uF = 1e-3
# nF = 1e-6
# pF = 1e-9

# second = 1e3
# ms = 1
# us = 1e-3
# ns = 1e-6
# ps = 1e-9

kappa_n = 0.75  # Subthreshold slope factor (n-type transistor)
kappa_p = 0.66  # Subthreshold slope factor (p-type transistor)
Ut = 25.0 * mV  # Thermal voltage
I0 = 1 * pA  # Dark current


class AdexLIF(nn.Module):
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

    def __init__(self, num_neurons=1, input_per_synapse=[1, 1, 1, 1]):
        super(AdexLIF, self).__init__()

        self.num_neurons = num_neurons

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
        self.Isoma_mem_init = 1.1 * I0
        self.register_buffer("Csoma_mem", torch.tensor(2 * pF))  # Membrane capacitance
        self.register_buffer("Isoma_dpi_tau", torch.tensor(5 * I0))  # Leakage current
        # self.register_buffer('Isoma_th', torch.tensor(2000 * I0))        # Spiking threshold
        self.Isoma_th = torch.nn.Parameter(
            torch.ones(1) * 2000 * I0, requires_grad=False
        )
        self.register_buffer("Isoma_reset", torch.tensor(1.2 * I0))  # Reset current
        self.register_buffer(
            "Isoma_const", torch.tensor(I0)
        )  # Additional input current similar to constant current injection
        self.register_buffer("soma_refP", torch.tensor(5 * ms))  # Refractory period

        #  ADAPTATION  ########################################################################################
        self.register_buffer(
            "Csoma_ahp", torch.tensor(4 * pF)
        )  # Spike-frequency adaptation capacitance
        self.register_buffer(
            "Isoma_ahp_tau", torch.tensor(2 * I0)
        )  # Leakage current for spike-frequency adaptation
        self.register_buffer("Isoma_ahp_g", torch.tensor(0))  # AHP gain current
        self.register_buffer(
            "Isoma_ahp_w", torch.tensor(1 * I0)
        )  # AHP jump height, on post

        #  POSITIVE FEEDBACK ##################################################################################
        self.register_buffer(
            "Isoma_pfb_gain", torch.tensor(100 * I0)
        )  # Positive feedback gain
        self.register_buffer(
            "Isoma_pfb_th", torch.tensor(1000 * I0)
        )  # Positive feedback activation threshold
        self.register_buffer(
            "Isoma_pfb_norm", torch.tensor(20 * I0)
        )  # Positive feedback normalization current

        # Synapse parameters ################
        # # SLOW_EXC, NMDA ########################################################################################
        self.Inmda_init = I0
        self.register_buffer("Cnmda", torch.tensor(2 * pF))  # Synapse's capacitance
        self.register_buffer(
            "Inmda_tau", torch.tensor(2 * I0)
        )  # Leakage current, i.e. how much current is constantly leaked away (time-constant)
        self.register_buffer(
            "Inmda_w0", torch.tensor(100 * I0)
        )  # Base synaptic weight, to convert unitless weight (set in synapse) to current
        self.register_buffer(
            "Inmda_thr", torch.tensor(I0)
        )  # NMDA voltage-gating threshold

        # FAST_EXC, AMPA ########################################################################################
        self.Iampa_init = I0  # Output current initial value
        self.register_buffer(
            "Campa", torch.tensor(2 * pF)
        )  # Synaptic capacitance, fixed at layout time (see chip for details)
        self.register_buffer(
            "Iampa_tau", torch.tensor(20 * I0)
        )  # Synaptic time constant current, the time constant is inversely proportional to I_tau
        self.register_buffer(
            "Iampa_w0", torch.tensor(100 * I0)
        )  # Base synaptic weight current which can be scaled by the .weight parameter

        # #INH, SLOW_INH, GABA_B, subtractive ##################################################################
        self.Igaba_b_init = I0  # Output current initial value
        self.register_buffer(
            "Cgaba_b", torch.tensor(2 * pF)
        )  # Synaptic capacitance, fixed at layout time (see chip for details)
        self.register_buffer(
            "Igaba_b_tau", torch.tensor(5 * I0)
        )  # Synaptic time constant current, the time constant is inversely proportional to I_tau
        self.register_buffer(
            "Igaba_b_w0", torch.tensor(100 * I0)
        )  # Base synaptic weight current which can be scaled by the .weight parameter

        # #FAST_INH, GABA_A, shunting, a mixture of subtractive and divisive ############################################
        self.Igaba_a_init = I0  # Output current initial value
        self.register_buffer(
            "Cgaba_a", torch.tensor(2 * pF)
        )  # Synaptic capacitance, fixed at layout time (see chip for details)
        self.register_buffer(
            "Igaba_a_tau", torch.tensor(5 * I0)
        )  # Synaptic time constant current, the time constant is inversely proportional to I_tau
        self.register_buffer(
            "Igaba_a_w0", torch.tensor(100 * I0)
        )  # Base synaptic weight current which can be scaled by the .weight parameter
        # ##################

        self.weight_nmda = torch.nn.Parameter(
            torch.ones(input_per_synapse[0], num_neurons) * 0, requires_grad=True
        )
        self.weight_ampa = torch.nn.Parameter(
            torch.ones(input_per_synapse[0], num_neurons) * 2, requires_grad=True
        )
        self.weight_gaba_a = torch.nn.Parameter(
            torch.ones(input_per_synapse[0], num_neurons) * 0, requires_grad=True
        )
        self.weight_gaba_b = torch.nn.Parameter(
            torch.ones(input_per_synapse[0], num_neurons) * 0, requires_grad=True
        )

        self.dt = 1 * ms

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

        state = self.AdexLIFState(
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
        return state

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
                input_nmda @ self.weight_nmda
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
                input_ampa @ self.weight_ampa
            )

        #### GABA B - inh ####
        low_current_gaba_b = self.I0 * (Igaba_b.detach() <= self.I0)
        Igaba_b_g = (
            self.alpha_gaba_b * self.Igaba_b_tau
        )  # GABA A synapse gain expressed in terms of its tau current
        Igaba_b_g_shunt = (
            Igaba_b_g * (Igaba_b.detach() > self.I0) + low_current_gaba_b
        )  # Shunt g current if Igaba_a goes to I0

        Igaba_b_tau_shunt = (
            self.Igaba_b_tau * (Igaba_b.detach() > I0) + low_current_gaba_b
        )  # Shunt tau current if Iampa goes to I0
        tau_gaba_b = (
            self.Cgaba_b * self.Ut / (kappa * Igaba_b_tau_shunt)
        )  # Synaptic time-constant

        dIgaba_b = (-Igaba_b - Igaba_b_g_shunt + 2 * low_current_gaba_b) / (
            tau_gaba_b * ((Igaba_b_g_shunt / Igaba_b) + 1)
        )
        if input_gaba_b is not None:
            Igaba_b = Igaba_b + self.Igaba_b_w0 * self.alpha_gaba_b * (
                input_gaba_b @ self.weight_gaba_b
            )

        #### # GABA A - shunt ####
        low_current_gaba_a = self.I0 * (Igaba_a.detach() <= self.I0)
        Igaba_a_g = (
            self.alpha_gaba_a * self.Igaba_a_tau
        )  # GABA A synapse gain expressed in terms of its tau current
        Igaba_a_g_shunt = (
            Igaba_a_g * (Igaba_a.detach() > self.I0) + low_current_gaba_a
        )  # Shunt g current if Igaba_a goes to I0

        Igaba_a_tau_shunt = (
            self.Igaba_a_tau * (Igaba_a.detach() > I0) + low_current_gaba_a
        )  # Shunt tau current if Iampa goes to I0
        tau_gaba_a = (
            self.Cgaba_a * self.Ut / (kappa * Igaba_a_tau_shunt)
        )  # Synaptic time-constant

        dIgaba_a = (-Igaba_a - Igaba_a_g_shunt + 2 * low_current_gaba_a) / (
            tau_gaba_a * ((Igaba_a_g_shunt / Igaba_a) + 1)
        )
        if input_gaba_a is not None:
            Igaba_a = Igaba_a + self.Igaba_a_w0 * self.alpha_gaba_a * (
                input_gaba_a @ self.weight_gaba_a
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
        firing = fast_sigmoid(Isoma_mem - self.Isoma_th)
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
