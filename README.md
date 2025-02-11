[![Documentation Status](https://readthedocs.org/projects/dynapsetorch/badge/?version=latest)](https://dynapsetorch.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Introduction

ARCANA is a library built on PyTorch to provide a simulation model of the DYNAP-SE hardware
The DPI neuron implemented in DYNAP-SE hardware has the following equation:
$$\left(1+\frac{I_{g}}{I_{mem}}\right)\tau\frac{d}{dt}I_{mem} + I_{mem}\left(1 + \frac{I_{ahp}}{I_\tau}\right) = I_\infty + f(I_{mem})$$
$$\tau_{ahp}\frac{d}{dt}I_{ahp} + I_{ahp} = I_{ahp_\infty}u(t)$$
$$I_\infty = \frac{I_{g}}{I_\tau}(I_{in} - I_{ahp} - I_\tau)$$

Where the $I_g$, $I_\tau$ are the gain and leakage currents, $I_{ahp} the adaptation current responsible of spike-frequency adaptation. $I_\infty$ the maximum current that the neuron would reach asymptotically, and $I_{in}$ the input current from the synapses.

The term $f(I_{mem})$ represents the positive feedback current that makes the neuron potential to increase exponentially when it reach an especific threshold.
$$f(I_{mem}) = \frac{I_{fb}}{I_{\tau}}(I_{mem} - I_{g})$$ 
$$I_{fb} = \frac{I_0^{\frac{1}{\kappa+1}}I_{mem}^{\frac{\kappa}{\kappa+1}}}{1 + e^{-\alpha(I_{mem} - I_{g})}}$$
Where $\alpha$ and $I_g$ are tunneable parameters, $I_0$ the dark current and $\kappa$ the transistor slope factor.

ARCANA simulate also the different synapse types that DYNAP-SE implemented: AMPA, NMDA, GABAa, and GABAb. AMPA and NMDA synapses, All implementing DPI circuits, follow the equation
$$\tau\frac{d}{dt}I_{syn} + I_{syn} = \frac{I_g}{I_\tau}I_w$$

In the case of AMPA and NMDA synapses, both are excitatory, GABAa and GABAb are inhibitory. 
NMDA synapse additionally is voltage gated. This mechanism makes the synaptic current dependent on the neuron’s membrane potential reaching a specific threshold.

## Quickstart

The following code demonstrate how to define a simple neural network in ARCANA. In this example a single neuron is created with and trained the leakage and gain current to have an specific output firing rate.
	
```
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from arcana.model import DPINeuron

neuron = DPINeuron(
    1,
    1,
    Itau_mem=4e-12,
    Igain_mem=20e-12,
    Ith=0.012,
    Idc=50e-12,
    refP=0.0,
    Ipfb_th=20e-12,
    Ipfb_norm=2e9,
    dt=1e-3,
    train_Igain_mem=True,
    train_Itau_mem=True,
)
```

Next, we define the training and test function. The training function will receive the neuron, optimizer and the number of epochs to train. To calculate the loss, the mean squared error of the number of output spikes will be used.

```
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
            (Imem, _, _, _, _, _) = state
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
        pbar.set_postfix({"Loss": loss.item()})
    return loss_hist, Vmem_hist, Itau_hist, Igain_hist
```

```
@torch.no_grad()
def test(neuron):
    state = None
    totalImem = []
    totalVmem = []
    neuron.eval()
    for t in tqdm(range(2000)):
        out, state = neuron(torch.zeros(1, 1), state)
        (Imem, Iampa, _, _, _, _) = state
        totalImem.append(Imem.numpy().item())
        totalVmem.append(neuron.I2V(Imem).numpy().item())
    return totalImem, totalVmem
```

Next, we create the optimizer to train the network. In this case is the Adam optimizer with a learning rate of 5e-3.

```
optimizer = torch.optim.Adam(neuron.parameters(), lr=5e-3)
optimizer.register_step_post_hook(neuron.UpdateParams)
print(optimizer)
```

Finally we train the network for 20 epochs, and we see difference of the model behaviour before and after training.

```
epochs = 20
Imem_pre, Vmem_pre = test(neuron)
(loss, _, Itau, Igain) = train(neuron, optimizer, epochs)
Imem_post, Vmem_post = test(neuron)
```

![alt text](https://github.com/ferqui/ARCANA/blob/main/docs/_static/single_neuron.svg?raw=true)

## Installation

```
pip install git+https://github.com/ferqui/ARCANA.git
```

## Contributing

If you want to contribute to this package development code, you can install it in edit mode:

```
git clone https://github.com/ferqui/ARCANA.git
cd ARCANA
pip install -e .
```

## Acknowledgments

ARCANA is currently maintained by the University of Cádiz and The Institute of Neuroinformatics (INI), UZH and ETHZ.

## Citation

If you find ARCANA useful in your work, please cite the following source:

LINK to reference

```
@misc{quintana2024ARCANA,
    title={A Realistic Simulation Framework for Analog/Digital Neuromorphic Architectures},
    author={Fernando M. Quintana and Maryada and Pedro L. Galindo and Elisa Donati and Giacomo Indiveri and Fernando Perez-Peña},
    year={2024},
    eprint={2409.14918},
    archivePrefix={arXiv},
    primaryClass={cs.NE},
    url={https://arxiv.org/abs/2409.14918},
}
```

## License & Copyright

ARCANA source code is published under the terms of the GPL-3.0 license. ARCANA's documentation is licensed under [Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](http://creativecommons.org/licenses/by-sa/4.0/?ref=chooser-v1>)
