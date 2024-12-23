"""
basic MQC module
"""

import datetime
import math
import os
import shutil
import textwrap
from copy import copy
from typing import List, Union

import numpy as np
import torch

# This should be equivalent to the APC subroutine in NEXMD
# TODO: implement one version ourselves?
from scipy.optimize import linear_sum_assignment

import openms
from openms.lib.misc import Molecule, au2A, call_name, fs2au, typewriter
from openms.qmd.es_driver import QuantumDriver
from openms.qmd.mqc import MQC
from openms.qmd.propagator import rk45

# TODO: this kind of definition should be part of backend
ArrayLike = Union[np.ndarray, torch.Tensor]


class SH(MQC):
    """Class for nuclear/electronic propagator used in surface hopping dynamics

    Attributes:
       :param object molecule: Molecule object

    """

    def __init__(
        self,
        molecule: List[Molecule],
        init_states: int,
        init_coef: np.array,
        qm: QuantumDriver,
        algorithm="FS",
        thermostat=None,
        decoherence=True,
        frustrated_hop=True,
        first_state=1,
        **kwargs,
    ):
        """Surface hopping

        :param molecule: list of molecular objects
        :type molecule: List[Molecule]
        :param init_states: initial states of each molecule
        :type init_states: np.array
        :param init_coef: initial wavefunction coefficients of each molecule
        :type init_coef: np.array
        :param qm: electronic structure driver
        :type qm: QuantumDriver
        :param thermostat: thermostat, defaults to None
        :type thermostat: object, optional
        """
        # Initialize input values
        super().__init__(molecule, init_states, init_coef, qm, thermostat, **kwargs)
        self.__dict__.update(kwargs)
        self.md_type = self.__class__.__name__
        # self.curr_ham = np.empty((len(self.mol), self.nstates, self.nstates))
        # a quick and dirty way to find the number of states
        if len(init_coef) > 30:
            self.find_hop = self._find_hop_numpy
        else:
            self.find_hop = self._find_hop_loop
        self.decoherence = decoherence
        self.frustrated_hop = frustrated_hop
        self.first_state = first_state
        self.curr_coords = np.array([])
        self.curr_veloc = np.array([])
        self.sync_variables("coords")
        self.sync_variables("veloc")
        self.curr_ham = np.empty(
            (self.nstates - self.first_state, self.nstates - self.first_state),
            dtype=complex,
        )
        if algorithm == "FS":
            self.electronic_propagator = self._electronic_propagator_fs
            self.velocity_rescaling = self._velocity_rescaling_fs
            self.hopping_probability = self._hopping_probability_fs
        elif algorithm == "LZ":
            self.electronic_propagator = self._electronic_propagator_lz
            self.velocity_rescaling = self._velocity_rescaling_lz
            self.hopping_probability = self._hopping_probability_lz
        else:
            raise ValueError("Unknown algorithm for hopping probability")
        self.curr_time: float
        self.saved_coords: ArrayLike
        self.probability: ArrayLike

    def initialize(self, *args):
        r"""Prepare the initial conditions for both quantum and classical EOM
        It should be implemented in derived class!
        """
        # call the BaseMD.initialize for the nuclear part.
        base_dir, md_dir, qm_log_dir = super().initialize(*args)

        # initialization for electronic part (TBD)
        return base_dir, md_dir, qm_log_dir
        # return NotImplementedError("Method not implemented!")

    def sync_variables(self, var_name, forward=True):
        current = f"curr_{var_name}"
        # from self.mol[:].var_name to self.curr_{var_name}
        if forward:
            tmp = np.array([getattr(mol, var_name) for mol in self.mol])
            if hasattr(getattr(self, current), "__trunc__"):
                setattr(self, current, tmp[0])
            else:
                setattr(self, current, tmp)
        else:
            tmp = getattr(self, current)
            if hasattr(tmp, "__trunc__"):
                [setattr(mol, var_name, tmp) for mol in self.mol]
            else:
                [setattr(mol, var_name, v) for mol, v in zip(self.mol, tmp)]

    def _electronic_propagator_fs(self):
        coef = copy(self.coef)
        p = np.zeros(len(coef))
        # self.sync_variables("coords")
        self.sync_variables("veloc")
        for _ in range(self.nesteps):
            coef = self.quantum_step(self.curr_time, coef)
            # print("coefficients", coef)
            density = self.get_densities(coef)
            p = self.hopping_probability(p, density)
            self.curr_time += self.edt
            self.curr_coords = self.saved_coords
        self.sync_variables("coords", forward=False)
        # print("sh", self.curr_coords)
        # e = self.qm.get_energies(self.mol[0].coords)
        # nacr = self.qm.get_nacr()
        # self.check_hops(coef, p, e, nacr)
        self.coef = coef
        self.probability = p
        # assign updated coefficients to each molecule
        # FIXME: this will only work if there is one molecule
        # for i, m in enumerate(self.mol):
        #     m.coef = coef[i]

    # the name of this function is for symmetry only
    def _electronic_propagator_lz(self):
        self.sync_variables("veloc")
        p = np.zeros(len(self.coef))
        for _ in range(self.nesteps):
            p = self.hopping_probability(p, None)
        self.sync_variables("coords")
        self.probability = p

    def quantum_step(self, t: float, coef: np.array):
        """Propagate one quantum step

        :param t: current time t
        :type t: float
        :param coef: coefficients at current time
        :type coef: np.array
        :return: coefficient at time t + quantum_step
        :rtype: np.array
        """
        return rk45(self.get_coef_dot, t, coef, self.edt)
        # return scipy_integrator(self.get_coef_dot, t, coef, self.edt)

    def get_coef_dot(self, t: float, coef: np.array):
        """Calculate the acceleration for the coefficient at given Hamiltonian

        :param t: current time
        :type t: float
        :param coef: coefficients at current time
        :type coef: np.array
        :return: time derivative of the coefficients at current time
        :rtype: np.array
        """
        # if saved time is different from given current time
        # the Hamiltonian needs to be recalculated
        # if t != self.curr_time:
        # FIXME: in many molecule case, this will not work.
        # self.curr_coords += self.curr_veloc * self.edt / 2
        # energies = self.qm.get_energies(self.curr_coords)
        # nact = self.qm.get_nact(self.curr_coords)
        # self.curr_ham = self.get_H(energies, nact)
        # self.curr_time = t
        # self.curr_ham = self.get_H(t)
        # coords = self.curr_coords + self.curr_veloc * (t - self.curr_time)
        # FIXME: better way to pass the "temporary" coordinates to qm!
        coords = self.curr_coords.copy()
        self.curr_coords += self.curr_veloc * (t - self.curr_time)
        self.sync_variables("coords", forward=False)
        # coords = self.curr_coords + self.curr_veloc * (t - self.curr_time)
        energies = self.qm.get_energies()[0]
        nact = self.qm.get_nact(self.curr_veloc)
        ham = self.get_H(energies, nact)
        c_dot = -1j * ham @ coef
        # print("Hamiltonian", ham)
        # print("c dot", c_dot)
        self.saved_coords = self.curr_coords
        self.curr_coords = coords
        return c_dot

    def dump_step(self):
        r"""Output coordinates, velocity, energies, electronic populations, etc.
        Universal properties will be dumped here (velocity, coordinate, energies)
        Other will be dumped in derived class!
        """

        return NotImplementedError("Method not implemented!")

    def get_H(self, energies: np.array, nact: np.array):
        """Function to assemble the Hamiltonian at the current quantum step.
        side.

        :param energies: potential energies of all excited states
        :type E: np.array
        :param nact: non-adiabatic coupling term
        :type nact: np.array
        :return: Hamiltonian of the system in matrix representation
        :rtype: np.array
        """
        ham = np.zeros_like(self.curr_ham)
        nact_cmplx = -1j * nact
        for i in range(self.first_state, len(energies)):
            # counter = 0
            idx = i - self.first_state
            ham[idx, idx] = energies[i]
        # fixme: dirty fix for one molecule only
        ham += nact_cmplx[0]
        self.curr_ham = ham
        return ham

    def get_densities(self, coef: np.array):
        # return np.einsum("i, j -> ij", np.conj(self.coef), self.coef)
        return np.outer(np.conj(coef), coef)

    def _hopping_probability_fs(self, p: np.array, density: np.array):
        j = self.current_states - self.first_state
        for n in range(self.nstates - self.first_state):
            if n != j:
                tmp = np.real(1j * density[n, j] * self.curr_ham[j, n])
                tmp *= 2 * self.edt / density[j, j].real
                # TODO: use trapezoid?
                p[n] += tmp
        return p

    def _get_de_update_position(self):
        # no position update yet, so the "current" energy is actually e- for this step
        e_minus = self.qm.get_energies()[0].copy()
        # move 2 electronic dt forward to get e+
        self._next_position_dt(2 * self.edt)
        e_plus = self.qm.get_energies()[0].copy()
        # move 1 electronic dt back to get the current energy
        # the position is retained
        self._next_position_dt(-self.edt)

        e = self.qm.get_energies()[0].copy()
        return e_minus, e, e_plus

    def _hopping_probability_lz(self, p: np.array, density: np.array):

        e_minus, e, e_plus = self._get_de_update_position()

        j = self.current_states
        for n in range(self.first_state, self.nstates):
            if n != j:
                de = e[n] - e[j]
                de_minus = e_minus[n] - e_minus[j]
                de_plus = e_plus[n] - e_plus[j]
                if de < 0:
                    de = -de
                    de_minus = -de_minus
                    de_plus = -de_plus
                if de < de_minus and de < de_plus:
                    d2edt2 = (de_plus - 2 * de + de_minus) / self.edt**2
                    tmp = math.exp(-math.pi / 2 * math.sqrt(de**3 / d2edt2))
                else:
                    tmp = 0
                p[n - self.first_state] += tmp
        return p

    def minimum_cost_solver(
        self, trans_den_mat_new: np.array, trans_den_mat_old: np.array
    ):
        """Solver to find the order of states to maximize the trace of the overlap
        matrix.

        :param trans_den_mat_new: transition density matrix at the current step
        :type trans_den_mat_new: np.array
        :param trans_den_mat_old: transition density matrix at the previous step
        :type trans_den_mat_old: np.array
        :return: order of all states at the current step
        :rtype: np.array
        """
        overlap_matrix = np.matmal(trans_den_mat_old, trans_den_mat_new.T)
        nstates = len(overlap_matrix)
        for i in range(nstates):
            for j in range(nstates):
                if j < i - 2 or j > i + 2:
                    overlap_matrix[i, j] = -1e5
        # idx will be the indices that leads to the minium cost
        _, idx = linear_sum_assignment(overlap_matrix, maximize=True)
        return idx, overlap_matrix

    def check_crossing(self, trans_den_mat_new: np.array, trans_den_mat_old: np.array):
        "check overlap for the same states before and after possible crossing"

        order = np.arange(len(trans_den_mat_new))
        cross = np.empty_like(order)
        state_idx_new, overlap = self.minimum_cost_solver(
            trans_den_mat_new, trans_den_mat_old
        )
        for i in range(len(order)):
            j = state_idx_new[i]
            if j != i:
                if i < j:
                    order[[i, j]] = order[[j, i]]
                if i < j or i == self.states:
                    if abs(overlap[i, j]) >= 0.9:
                        # trivial crossing
                        cross[i] = 2
                    else:
                        # reduce time step
                        cross[i] = 1
                else:
                    cross[i] = 0
            else:
                cross[i] = 0
        return order, cross

    def check_hops(self):
        final_state = self.find_hop(self.probability)
        # print(probability, rand)
        # no possible hop is identified
        if final_state > -1:
            final_state += self.first_state
            # print("possible hop here", self.probability, final_state)
            e = self.qm.get_energies()
            self.sync_variables("veloc")
            (
                self.current_states,
                self.curr_veloc,
                hop_type,
            ) = self.velocity_rescaling(
                self.current_states, final_state, self.curr_veloc, e
            )
            print(f"     Hop type {hop_type}")
            self.sync_variables("veloc", forward=False)
            # if hop is successful, exit
            if hop_type == 1:
                # TODO: unify the variable names to use self.sync_variables
                for mol in self.mol:
                    mol.current_state = self.current_states
                # TODO: other decoherence schemes
            if self.decoherence:
                # apply instantaneous decoherence
                self.instantaneous_decoherence(self.current_states)

    def _find_hop_numpy(self, p: ArrayLike):
        """Find possible hop with operations vectorized. When the number of states is
        large (> 50), the function will be faster.

        :param p: array of hop probabilities
        :type p: np.array
        :return: final state index
        :rtype: integer
        """
        rand = self.prng.random()
        p[p < 0] = 0
        # calculate the cumulative sum of the probability array
        p_sum = np.cumsum(p)
        # return the idx if we were to insert `rand` into the array
        idx = np.searchsorted(p_sum, rand)
        # rand > sum(p), no hop
        if idx == len(p):
            return -1
        # if rand is exactly the same as the p_sum[idx], numpy will return idx
        # which is invalid for SH algorithm
        # practically, this will probably never happen
        elif rand == p_sum[idx]:
            return -1
        return idx

    def _find_hop_loop(self, p: ArrayLike):
        """Find possible hop with a loop. When the number of states is small
        (around < 30), the loop approach is likely faster than numpy.

        :param p: array of hop probabilities
        :type p: ArrayLike
        :return: final state index
        :rtype: integer
        """
        # generate random number
        rand = self.prng.random()
        # print("random number", rand)
        # print(np.sum(p), rand)
        # manually compute cumulative sum
        cumsum = 0
        for idx, prob in enumerate(p):
            if prob <= 0:
                continue
            cumsum += prob
            if cumsum > rand:
                return idx
        return -1

    def _velocity_rescaling_fs(
        self,
        state_i: int,
        state_f: int,
        veloc: np.array,
        energies: np.array,
    ):
        """
        Updates velocity by rescaling the *momentum* in the specified direction and amount

        :param direction: the direction of the *momentum* to rescale
        :param reduction: how much kinetic energy should be damped
        """
        # normalize
        nacr = self.qm.get_nacr()
        nacr = nacr[0, state_i - self.first_state, state_f - self.first_state]
        nacr = nacr / np.linalg.norm(nacr)
        inverse_mass = 1 / self.mol[0].mass
        a = np.sum(inverse_mass.reshape(-1, 1) * nacr**2)
        energies = energies[0]
        e_i = energies[state_i]
        e_f = energies[state_f]
        dE = e_f - e_i
        b = 2 * np.einsum("ij, ij", nacr, veloc[0])
        c = 2 * dE
        delta = b**2 - 4 * a * c
        # frustrated hops
        if delta < 0:
            if self.frustrated_hop:
                veloc = -veloc
            return state_i, veloc, 2
        # hops allowed
        if b < 0:
            factor = -(b + math.sqrt(delta)) / (2 * a)
        else:
            factor = -(b - math.sqrt(delta)) / (2 * a)
        veloc += factor * inverse_mass.reshape(-1, 1) * nacr
        return state_f, veloc, 1

    def _velocity_rescaling_lz(
        self,
        state_i: int,
        state_f: int,
        veloc: np.array,
        energies: np.array,
    ):
        e_i = energies[0, state_i]
        e_f = energies[0, state_f]
        dE = e_f - e_i
        ekin = self.mol[0].ekin
        # no enough energy to hop
        if dE > ekin:
            if self.frustrated_hop:
                veloc = -veloc
            return state_i, veloc, 2
        else:
            fac = math.sqrt(1 - dE / ekin)
            return state_f, fac * veloc, 1

    def instantaneous_decoherence(self, state):
        self.coef = np.zeros_like(self.coef, dtype=complex)
        self.coef[state - self.first_state] = 1 + 0j
