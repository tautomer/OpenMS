import math

import numpy as np

from openms.qmd.es_driver import QuantumDriver


class TullyModel(QuantumDriver):
    """Base class for all 3 Tully models.
    Reference https://pubs.aip.org/aip/jcp/article/93/2/1061/454789/Molecular-dynamics-with-electronic
    """

    def __init__(self, molecule):
        self.mol = molecule
        self.H = np.empty((2, 2), dtype=float)
        self.dH = np.empty((2, 2), dtype=float)
        super().__init__()

    def get_H(self, x):
        return NotImplemented

    def diag(self):
        e, wf = np.linalg.eig(self.H)
        idx = np.argsort(e)
        self.e = np.take_along_axis(e, idx, axis=0)
        self.wf = wf[:, idx]

    def get_energies(self, x=None):
        self._assign_energies(self.mol[0], self.e)
        return self.e

    def nuc_grad(self):
        f = []
        for i in range(2):
            tmp = self.wf[:, i] @ self.dH @ self.wf[:, i]
            f.append([tmp, 0, 0])
        f = -np.array(f)
        return f

    def _assign_forces(self, molecule, force):
        for i in range(len(force)):
            molecule.states[i].forces = force[i]

    def _assign_energies(self, molecule, energy):
        for i in range(len(energy)):
            molecule.states[i].energy = energy[i]

    def calculate_forces(self):
        forces = self.nuc_grad()
        self._assign_forces(self.mol[0], forces)

    def get_nact(self, veloc: np.array):
        d = self.get_nacr()[0]
        return np.array([d * veloc.flatten()[0]])

    def get_nacr(self):
        tmp = self.wf[:, 0] @ self.dH @ self.wf[:, 1]
        d = tmp / (self.e[1] - self.e[0])
        d = self.fix_phases(d)
        return np.array([d])

    def fix_phases(self, d):
        return d


class TullyModel1(TullyModel):
    """Tully model 1, simple avoided crossing (SAC) model"""

    def __init__(self, molecule, a=0.01, b=1.6, c=0.005, d=1.0):
        self.a = a
        self.b = b
        # set c to 0 to disable NAC
        self.c = c
        # set d to 0 for constant NAC
        self.d = d
        super().__init__(molecule)

    def get_H(self):
        x = self.mol[0].coords.flatten()[0]
        if x > 0:
            e0 = self.a * (1 - np.exp(-self.b * x))
            f0 = self.a * self.b * np.exp(-self.b * x)
        else:
            e0 = self.a * (np.exp(self.b * x) - 1)
            f0 = self.a * self.b * np.exp(self.b * x)
        d = self.c * np.exp(-self.d * x**2)
        # self.H = np.array([[e0, d], [d, -e0]])
        self.H[0, 0] = e0
        self.H[0, 1] = d
        self.H[1, 0] = d
        self.H[1, 1] = -e0
        d *= -2 * self.d * x
        # self.dH = np.array([[f0, d], [d, -f0]])
        self.dH[0, 0] = f0
        self.dH[0, 1] = d
        self.dH[1, 0] = d
        self.dH[1, 1] = -f0

    def fix_phases(self, d):
        if d < 0:
            d = -d
        return d


class TullyModel2(TullyModel):
    """Tully model 2, dual avoided crossing (DAC) model"""

    def __init__(self, molecule, a=0.1, b=0.28, c=0.015, d=0.06, e1_0=0.05):
        self.a = a
        self.b = b
        # set c to 0 to disable NAC
        self.c = c
        # set d to 0 for constant NAC
        self.d = d
        self.e1_0 = e1_0
        super().__init__(molecule)

    def get_H(self):
        x = self.mol[0].coords.flatten()[0]
        f1 = self.a * math.exp(-self.b * x**2)
        e1 = self.e1_0 - f1
        f1 *= 2 * self.b * x
        dac = self.c * math.exp(-self.d * x**2)
        self.H[0, 0] = 0
        self.H[0, 1] = dac
        self.H[1, 0] = dac
        self.H[1, 1] = e1

        dac *= -2 * self.d * x
        self.dH[0, 0] = 0
        self.dH[0, 1] = dac
        self.dH[1, 0] = dac
        self.dH[1, 1] = f1

    def fix_phases(self, d):
        x = self.mol[0].coords.flatten()[0]
        # for model 2, NACR is an odd function
        if x * d > 0:
            d = -d
        return d


class TullyModel3(TullyModel):
    """Tully model 3, extended coupling with reflection (ECWR) model"""

    def __init__(self, molecule, a=6e-4, b=0.1, c=0.9):
        self.a = a
        self.b = b
        # set c to 0 to disable NAC
        self.c = c
        super().__init__(molecule)

    def get_H(self, x):
        x = x.flatten()[0]
        if x > 0:
            tmp = self.b * math.exp(-self.c * x)
            dac = 2 * self.b - tmp
        else:
            tmp = self.b * math.exp(self.c * x)
            dac = tmp
        # self.H = np.array([[e0, d], [d, -e0]])
        self.H[0, 0] = self.a
        self.H[0, 1] = dac
        self.H[1, 0] = dac
        self.H[1, 1] = -self.a
        d = tmp * self.c
        # self.dH = np.array([[f0, d], [d, -f0]])
        self.dH[0, 0] = 0
        self.dH[0, 1] = d
        self.dH[1, 0] = d
        self.dH[1, 1] = 0

    def fix_phases(self, d):
        if d < 0:
            d = -d
        return d

