from typing import List, Callable

from qutip import *
import numpy as np
import scipy.integrate as integrate

# Note: it is OK if object/function names from qutip are not
# highlighted and underlined (e.g. "qutip" or "Qobj"). However,
# it is recommended you test your solutions in a Jupyter notebook
# before submitting


class HardwareSolutions:
    def __init__(self):
        pass

    def hardware_1(self, omega: float) -> Qobj:
        H = -omega / 2 * sigmaz()
        return H

    def hardware_2(
        self, Omega_R: float, omega: float, omega_drive: float, t: float
    ) -> Qobj:
        H_TLS = -omega / 2 * sigmaz()
        H_E = -Omega_R * np.cos(omega_drive * t) * sigmax()
        return H_TLS + H_E

    def hardware_3(
        self, Omega_R: float, omega: float, omega_drive: float, t: List[float]
    ) -> float:
        def H(time):
            return self.hardware_2(Omega_R, omega, omega_drive, time)

        initial_density_matrix = basis(2, 0) * basis(2, 0).dag()
        density_matrix_evolution = mesolve(H, initial_density_matrix, t).states
        return expect(initial_density_matrix, density_matrix_evolution)

    def hardware_4(
        self,
        psi_n: Callable[[float], float],
        psi_n_1: Callable[[float], float],
        E_1: float,
        q: float,
    ) -> float:
        p_12 = integrate.quad(lambda x: psi_n(x) * psi_n_1(x) * x, -np.inf, np.inf)
        return q * E_1 * p_12

    def hardware_5(self, Omega_R: float, delta: float) -> float:
        return np.square(Omega_R) / (np.square(Omega_R) + np.square(delta))
