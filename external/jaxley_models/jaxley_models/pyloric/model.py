from io import StringIO

import jax.numpy as jnp
import jaxley as jx
import pandas as pd

from jaxley_models.pyloric.channels import (A, CaNernstReversal, CaS, CaT, H,
                                            KCa, Kd, Leak, Na)
from jaxley_models.pyloric.synapses import (CholinergicSynapse,
                                            GlutamatergicSynapse)

prinz04_params = """
# Parameters for the Pyloric network (Prinz et al. 2004, Tab. 2)
model_neuron,gNa,gCaT,gCaS,gA,gKCa,gKd,gH,gLeak
ab_pd,0.400,0.0025,0.006,0.050,0.010,0.100,0.00001,0.00000
ab_pd,0.100,0.0025,0.004,0.050,0.005,0.100,0.00001,0.00000
ab_pd,0.200,0.0025,0.004,0.050,0.005,0.050,0.00001,0.00000
ab_pd,0.200,0.0050,0.004,0.040,0.005,0.125,0.00001,0.00000
ab_pd,0.300,0.0025,0.002,0.010,0.005,0.125,0.00001,0.00000
lp,0.100,0.0000,0.008,0.040,0.005,0.075,0.00005,0.00002
lp,0.100,0.0000,0.006,0.030,0.005,0.050,0.00005,0.00002
lp,0.100,0.0000,0.010,0.050,0.005,0.100,0.00000,0.00003
lp,0.100,0.0000,0.004,0.020,0.000,0.025,0.00005,0.00003
lp,0.100,0.0000,0.004,0.030,0.000,0.075,0.00005,0.00002
py,0.100,0.0025,0.002,0.050,0.000,0.125,0.00005,0.00001
py,0.200,0.0075,0.000,0.050,0.000,0.075,0.00005,0.00000
py,0.200,0.0100,0.000,0.050,0.000,0.100,0.00005,0.00000
py,0.400,0.0025,0.002,0.050,0.000,0.075,0.00005,0.00000
py,0.500,0.0025,0.002,0.040,0.000,0.125,0.00001,0.00003
py,0.500,0.0025,0.002,0.040,0.000,0.125,0.00000,0.00002
"""


def PyloricNetwork() -> jx.Network:
    """Model of the pyloric circuit.

    Model of the pyloric circuit according to:

    Prinz et al. 2003, J Neurophysiol

    We model the network using three cells using the `jx.Network` class:
    - ab_pd: abducens and pre-dorsal pyloric neuron
    - lp: lateral pyloric neuron
    - py: posteroventral pyloric neuron
    which can be accessed by the group names "ab_pd", "lp", and "py" respectively.

    The network is connected as follows using glutamatergic and cholinergic synapses:
    - ab_pd -> lp : glutamatergic
    - ab_pd -> lp : cholinergic
    - ab_pd -> py : glutamatergic
    - ab_pd -> py : cholinergic
    - lp -> ab_pd : glutamatergic
    - lp -> py : glutamatergic
    - py -> lp : glutamatergic

    The simulator was re-implemented based on the cython implementation from:
    https://github.com/mackelab/pyloric and matches the results from the original code.

    Returns:
        The pyloric circuit network.
    """
    # build network
    net = jx.Network([jx.Cell()] * 3)
    net.cell(0).add_to_group("ab_pd")
    net.cell(1).add_to_group("lp")
    net.cell(2).add_to_group("py")

    # insert channels
    net.insert(CaNernstReversal())
    net.insert(Na())
    net.insert(CaT())
    net.insert(CaS())
    net.insert(A())
    net.insert(KCa())
    net.insert(Kd())
    net.insert(H())
    net.insert(Leak())

    # set cell parameters
    area = 0.6283 * 1e-3  # cm²
    C = 0.628 * 1e-3  # uF
    radius = jnp.sqrt(area / (2 * jnp.pi)) * 1e4  # um
    net.set("capacitance", C / area)  # uF / cm²
    net.set("length", radius)
    net.set("radius", radius)
    net.set("v", -50.0)

    params = pd.read_csv(StringIO(prinz04_params), comment="#")
    params = params.groupby("model_neuron").last().T.to_dict()
    for neuron, conductances in params.items():
        for conductance, value in conductances.items():
            net.__getattr__(neuron).set(f"{conductance[1:]}_{conductance}", value)

    # Create synaptic connections
    glut = GlutamatergicSynapse()
    chol = CholinergicSynapse()
    jx.connect(net.ab_pd, net.lp, glut)  # AB-LP
    jx.connect(net.ab_pd, net.lp, chol)  # PD-LP

    jx.connect(net.ab_pd, net.py, glut)  # AB-PY
    jx.connect(net.ab_pd, net.py, chol)  # PD-PY

    jx.connect(net.lp, net.ab_pd, glut)  # LP-PD
    jx.connect(net.lp, net.py, glut)  # LP-PY

    jx.connect(net.py, net.lp, glut)  # PY-LP
    return net
