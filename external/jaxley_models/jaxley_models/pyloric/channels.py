from typing import Dict, Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp

from jaxley_models.utils import double_exp, sigmoid, temp_factor


class PyloricChannel(Channel):
    """Base class for pyloric channels.

    The channels were modelled after Prinz et al. 2003.

    All channels in the pyloric network are based on this class. It implements
    the gating dynamics for the m and h gates, as well as the channel current.

    dm/dt = (m_inf(V) - m) / tau_m(V)
    dh/dt = (h_inf(V) - h) / tau_h(V)

    I = g_Ion * m^p * h * (V - E_Ion),
    where p raises m to some power between 1 and 4.

    To implement a specific network channel m_inf(V), h_inf(V), tau_m(V), tau_h(V) need
    to be implemented and the parameters g_Ion, E_Ion, p need to be set.

    For specifics see Prinz et al. 2003.
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_states = {}
        self.channel_params = {
            "temperature": 283.0,  # K
            f"{prefix}_q10_m": 2.4,
            f"{prefix}_q10_h": 2.8,
            f"{prefix}_q10_g": 1.5,
        }
        self.current_name = f"i_{prefix}"
        self.name_gIon = f"{prefix}_g{prefix}"
        self.name_eIon = f"{prefix}_e{prefix}"
        self.pow_of_m = 3

    def m_inf(self, v, states, params):
        return 0.0

    def h_inf(self, v, states, params):
        return 0.0

    def tau_m(self, v, states, params):
        return 0.0

    def tau_h(self, v, states, params):
        return 0.0

    def update_m(self, states, dt, v, params):
        prefix = self._name
        m = states[f"{prefix}_m"]
        q10_m = params[f"{prefix}_q10_m"]
        temp = params["temperature"]
        m_inf = self.m_inf(v, states, params)
        tau_m = self.tau_m(v, states, params)
        tau_m_at_temp = tau_m / temp_factor(q10_m, temp)
        return exponential_euler(m, dt, m_inf, tau_m_at_temp)

    def update_h(self, states, dt, v, params):
        prefix = self._name
        h = states[f"{prefix}_h"]
        q10_h = params[f"{prefix}_q10_h"]
        temp = params["temperature"]
        h_inf = self.h_inf(v, states, params)
        tau_h = self.tau_h(v, states, params)
        tau_h_at_temp = tau_h / temp_factor(q10_h, temp)
        return exponential_euler(h, dt, h_inf, tau_h_at_temp)

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        prefix = self._name
        new_states = {}
        if f"{prefix}_m" in states:
            new_states[f"{prefix}_m"] = self.update_m(states, dt, v, params)
        if f"{prefix}_h" in states:
            new_states[f"{prefix}_h"] = self.update_h(states, dt, v, params)
        return new_states

    def pyloric_channel_current(self, states, v, gIon, eIon, pow_of_m=3):
        prefix = self._name
        m = states[f"{prefix}_m"] if f"{prefix}_m" in states else 1.0
        h = states[f"{prefix}_h"] if f"{prefix}_h" in states else 1.0
        return gIon * (m**pow_of_m) * h * (v - eIon)

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        prefix = self._name
        gIon = params[self.name_gIon]
        eIon = params[self.name_eIon]
        q10_g = params[f"{prefix}_q10_g"]
        temp = params["temperature"]
        gIon_at_temp = gIon * temp_factor(q10_g, temp)
        return self.pyloric_channel_current(
            states, v, gIon_at_temp, eIon, self.pow_of_m
        )

    def init_state(self, states, v, params, delta_t):
        prefix = self._name
        init_states = {}
        if f"{prefix}_m" in self.channel_states:
            m_inf = self.m_inf(v, states, params)
            init_states[f"{prefix}_m"] = m_inf
        if f"{prefix}_h" in self.channel_states:
            h_inf = self.h_inf(v, states, params)
            init_states[f"{prefix}_h"] = h_inf
        return init_states


class Na(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h": 0.0,
        }
        self.channel_params.update(
            {
                f"{prefix}_gNa": 0.2,
                f"{prefix}_eNa": 50.0,
            }
        )
        self.current_name = f"i_Na"
        self.name_gIon = f"{prefix}_gNa"
        self.name_eIon = f"{prefix}_eNa"
        self.pow_of_m = 3

    def m_inf(self, v, states, params):
        return sigmoid(v, 25.5, -5.29)

    def h_inf(self, v, states, params):
        return sigmoid(v, 48.9, 5.18)

    def tau_m(self, v, states, params):
        return 2.64 - 2.52 * sigmoid(v, 120.0, -25.0)

    def tau_h(self, v, states, params):
        return (1.34 * sigmoid(v, 62.9, -10.0)) * (1.5 + sigmoid(v, 34.9, 3.6))


class CaT(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params.update(
            {
                f"{prefix}_gCaT": 0.0026,
            }
        )
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h": 0.0,
            "eCa": 50.0,
        }
        self.current_name = f"i_Ca"
        self.name_gIon = f"{prefix}_gCaT"
        self.name_eIon = f"eCa"
        self.pow_of_m = 3

    def m_inf(self, v, states, params):
        return sigmoid(v, 27.1, -7.2)

    def h_inf(self, v, states, params):
        return sigmoid(v, 32.1, 5.5)

    def tau_m(self, v, states, params):
        return 43.4 - 42.6 * sigmoid(v, 68.1, -20.5)

    def tau_h(self, v, states, params):
        return 210.0 - 179.6 * sigmoid(v, 55.0, -16.9)

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        prefix = self._name
        gIon = params[self.name_gIon]
        eIon = states[self.name_eIon]
        q10_g = params[f"{prefix}_q10_g"]
        temp = params["temperature"]
        gIon_at_temp = gIon * temp_factor(q10_g, temp)
        return self.pyloric_channel_current(
            states, v, gIon_at_temp, eIon, self.pow_of_m
        )


class CaNernstReversal(Channel):
    """Calcium dynamics following Prinz et al. 2003.

    The intracellular calcium concentration follows:
    d[Ca] / dt = (-f*I_Ca - [Ca] + Ca_0) / tau_Ca

    The reversal potential for calcium is computed using the Nernst equation:
    E_Ca = RT / (2F) * ln([Ca_ext] / [Ca_i])
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "Ca_ext": 3000.0,
            "tau_Ca": 200,  # ms
            "Ca_0": 0.05,  # muM
            "f": 14961.0,
            "q10_Ca": 2.0,
            "temperature": 283.0,
        }
        self.channel_states = {
            "eCa": 50.0,
            "concCa": 0.05,
        }
        self.current_name = f"i_Ca"

    def RToverzF(self, temp):
        R = 8.31451e3  # mJ / (mol * K)
        F = 96485.3415  # C / mol
        z = 2  # Ca is divalent
        return R * temp / (z * F)  # mJ / (mol * K) * K / (C / mol) = mV

    def update_states(
        self, states: Dict[str, jnp.ndarray], dt, v, params: Dict[str, jnp.ndarray]
    ):
        concCa = states["concCa"]
        Ca_ext = params["Ca_ext"]
        f = params["f"]
        tau_Ca = params["tau_Ca"]
        Ca_0 = params["Ca_0"]
        i_Ca = states["i_Ca"]
        temp = params["temperature"]
        q10_Ca = params["q10_Ca"]
        r = params["radius"]
        l = params["length"]
        area = r * l * 2 * jnp.pi * 1e-8

        Ca_inf = Ca_0 - f * i_Ca * 1000.0 * area
        tau_Ca_at_temp = tau_Ca / temp_factor(q10_Ca, temp)
        Ca_new = exponential_euler(concCa, dt, Ca_inf, tau_Ca_at_temp)

        eCa_new = self.RToverzF(temp) * jnp.log(Ca_ext / concCa)
        return {"concCa": Ca_new, f"eCa": eCa_new}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        return 0.0

    def init_state(self, states, v, params, delta_t):
        return {"concCa": params["Ca_0"]}


class CaS(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params.update(
            {
                f"{prefix}_gCaS": 0.004,
                "Ca_ext": 3000.0,
                "tau_Ca": 200,  # ms
                "Ca_0": 0.05,  # muM
                "f": 14961.0,
                f"{prefix}_q10_m": 2.0,  # Caplan 2014
            }
        )
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h": 0.0,
            "eCa": 50.0,
            "concCa": 0.05,
        }
        self.current_name = f"i_Ca"
        self.name_gIon = f"{prefix}_gCaS"
        self.name_eIon = f"eCa"
        self.pow_of_m = 3

    def m_inf(self, v, states, params):
        return sigmoid(v, 33.0, -8.1)

    def h_inf(self, v, states, params):
        return sigmoid(v, 60.0, 6.2)

    def tau_m(self, v, states, params):
        return 2.8 + (14.0 / double_exp(v, 27.0, 10.0, 70.0, -13.0))

    def tau_h(self, v, states, params):
        return 120.0 + (300.0 / double_exp(v, 55.0, 9.0, 65.0, -16.0))

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        prefix = self._name
        gIon = params[self.name_gIon]
        eIon = states[self.name_eIon]
        q10_g = params[f"{prefix}_q10_g"]
        temp = params["temperature"]
        gIon_at_temp = gIon * temp_factor(q10_g, temp)
        return self.pyloric_channel_current(
            states, v, gIon_at_temp, eIon, self.pow_of_m
        )


class A(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h": 0.0,
        }
        self.channel_params.update(
            {
                f"{prefix}_gA": 0.04,
                f"{prefix}_eK": -80.0,
            }
        )
        self.current_name = f"i_K"
        self.name_gIon = f"{prefix}_gA"
        self.name_eIon = f"{prefix}_eK"
        self.pow_of_m = 3

    def m_inf(self, v, states, params):
        return sigmoid(v, 27.2, -8.7)

    def h_inf(self, v, states, params):
        return sigmoid(v, 56.9, 4.9)

    def tau_m(self, v, states, params):
        return 23.2 - 20.8 * sigmoid(v, 32.9, -15.2)

    def tau_h(self, v, states, params):
        return 77.2 - 58.4 * sigmoid(v, 38.9, -26.5)


class KCa(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params.update(
            {
                f"{prefix}_gKCa": 0.004,
                f"{prefix}_eK": -80.0,
                f"{prefix}_q10_m": 1.6,  # Caplan 2014
            }
        )
        self.channel_states = {
            f"{prefix}_m": 0.0,
            "concCa": 0.05,
        }
        self.current_name = f"i_K"
        self.name_gIon = f"{prefix}_gKCa"
        self.name_eIon = f"{prefix}_eK"
        self.pow_of_m = 4

    def m_inf(self, v, states, params):
        return (states["concCa"] / (states["concCa"] + 3.0)) * sigmoid(v, 28.3, -12.6)

    def tau_m(self, v, states, params):
        return 180.6 - 150.2 * sigmoid(v, 46.0, -22.7)


class Kd(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params.update(
            {
                f"{prefix}_gKd": 0.09,
                f"{prefix}_eK": -80.0,
            }
        )
        self.channel_states = {
            f"{prefix}_m": 0.0,
        }
        self.current_name = f"i_K"
        self.name_gIon = f"{prefix}_gKd"
        self.name_eIon = f"{prefix}_eK"
        self.pow_of_m = 4

    def m_inf(self, v, states, params):
        return sigmoid(v, 12.3, -11.8)

    def tau_m(self, v, states, params):
        return 14.4 - 12.8 * sigmoid(v, 28.3, -19.2)


class H(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params.update(
            {
                f"{prefix}_gH": 0.00003,
                f"{prefix}_eH": -20.0,
            }
        )
        self.channel_states = {
            f"{prefix}_m": 0.0,
        }
        self.current_name = f"i_H"
        self.name_gIon = f"{prefix}_gH"
        self.name_eIon = f"{prefix}_eH"
        self.pow_of_m = 1

    def m_inf(self, v, states, params):
        return sigmoid(v, 75.0, 5.5)

    def tau_m(self, v, states, params):
        return 2.0 / (save_exp((v + 169.7) / -11.6) + save_exp((v - 26.7) / 14.3))


class Leak(PyloricChannel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params.update(
            {
                f"{prefix}_gLeak": 0.00001,
                f"{prefix}_eLeak": -50.0,
            }
        )
        self.channel_states = {}
        self.current_name = f"i_Leak"
        self.name_gIon = f"{prefix}_gLeak"
        self.name_eIon = f"{prefix}_eLeak"
        self.pow_of_m = 0
