from typing import Dict, Optional

from jaxley.solver_gate import exponential_euler
from jaxley.synapses import Synapse

from jaxley_models.utils import sigmoid, temp_factor


class PyloricSynapse(Synapse):
    """Base class for pyloric synapses modelled after Prinz et al. 2004.

    https://www.nature.com/articles/nn1352.pdf

    ds/dt = (s_inf(V_pre) - s) / tau_s

    s_inf(V_pre) = 1 / (1 + exp((V_th - V_pre) / delta))
    tau_s = k_minus * (1 - s_inf(V_pre))
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,
            f"{prefix}_q10_s": 1.7,
            f"{prefix}_q10_g": 1.5,
            f"{prefix}_temperature": 283.0,  # param sharing not possible for synapses currently
            f"{prefix}_e_syn": -70.0,
            f"{prefix}_k_minus": 1 / 40.0,
            f"{prefix}_v_th": -35.0,  # mV
            f"{prefix}_delta": 5.0,  # mV
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(
        self,
        states: Dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: Dict,
    ) -> Dict:
        prefix = self._name
        v_th = params[f"{prefix}_v_th"]
        delta = params[f"{prefix}_delta"]
        k_minus = params[f"{prefix}_k_minus"]
        q10_s = params[f"{prefix}_q10_s"]
        temp = params[f"{prefix}_temperature"]

        s = states[f"{prefix}_s"]

        s_inf = sigmoid(-pre_voltage, v_th, delta)
        tau_s = (1.0 - s_inf) / k_minus
        tau_s_at_temp = tau_s / temp_factor(q10_s, temp)
        new_s = exponential_euler(s, delta_t, s_inf, tau_s_at_temp)
        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        g_syn = params[f"{prefix}_gS"]
        e_syn = params[f"{prefix}_e_syn"]
        q10_g = params[f"{prefix}_q10_g"]
        temp = params[f"{prefix}_temperature"]

        s_syn = states[f"{prefix}_s"]

        g_syn_at_temp = g_syn * temp_factor(q10_g, temp)
        return g_syn_at_temp * s_syn * (post_voltage - e_syn)


class GlutamatergicSynapse(PyloricSynapse):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params.update(
            {
                f"{prefix}_e_syn": -70.0,
                f"{prefix}_k_minus": 1 / 40.0,
            }
        )


class CholinergicSynapse(PyloricSynapse):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params.update(
            {
                f"{prefix}_e_syn": -80.0,
                f"{prefix}_k_minus": 1 / 100.0,
            }
        )
