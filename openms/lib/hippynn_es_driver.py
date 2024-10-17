#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Xinyang Li <lix@lanl.gov>
#

import os
from typing import List, Union

try:
    from hippynn.experiment.serialization import load_model_from_cwd
    from hippynn.graphs import Predictor

    HIPPYNN_AVAILABLE = True
except ImportError:
    HIPPYNN_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
    torch._C._set_grad_enabled(True)
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

from openms.lib.misc import Molecule, periodictable
from openms.qmd.es_driver import QuantumDriver

r"""
HIPNN-based electronic structure driver
"""

if HIPPYNN_AVAILABLE and TORCH_AVAILABLE:

    class NNDriver(QuantumDriver):
        def __init__(
            self,
            molecule: Union[Molecule, List[Molecule]],
            nstates: int,
            model_path: str,
            coords_unit="Angstrom",
            energy_conversion=0.036749405469679,
            model_device="cpu",
            multi_targets=True,
            ml_model_nstates=None,
        ):
            r"""Initialize the driver with a HIPNN model.

            :param molecule: object of a molecule or list of molecules
            :type molecule: Union[Molecule, List[Molecule]]
            :param nstates: number of states needed in the simulation.
            :type nstates: int
            :param model_path: the directory where the saved model is located.
            :type model_path: str
            :param model_device: the device where the model will be loaded.
                For better compatibility, defaults to "cpu".
            :type model_device: str, optional
            :param multi_targets: whether multi-targets nodes are used in
                hippynn, defaults to True.
            :type multi_targets: bool, optional
            :raises RuntimeError: error exit when PyTorch or hippynn is missing.
            """
            if not HIPPYNN_AVAILABLE or not TORCH_AVAILABLE:
                raise RuntimeError(
                    "PyTorch and hippynn must be installed to use the neural network"
                    " electronic structure driver."
                )
            super().__init__()
            self.mol = molecule
            self.get_Z()
            # fixme: properly deal with ground state included or not
            self.state_pairs = torch.triu_indices(nstates - 1, nstates - 1, 1).T + 1
            self.nstates = nstates
            if ml_model_nstates is not None:
                if ml_model_nstates < nstates:
                    raise ValueError(
                        "The number of states in the ML models must be larger than or equal to the number of states used in the simulations."
                    )
                elif ml_model_nstates > nstates:
                    idx_orig = list(zip(*np.triu_indices(ml_model_nstates, k=2)))
                    idx_new = list(zip(*np.triu_indices(nstates, k=2)))
                    self.slices = torch.tensor(np.isin(idx_orig, idx_new).all(axis=1))
                self.ml_model_nstates = ml_model_nstates
            else:
                self.ml_model_nstates = nstates
            current_dir = os.getcwd()
            # TODO: hippynn doesn't have the ability to load model other than the current
            # TODO: working directory. Might be useful to add such a function on the hippynn
            # TODO: side
            os.chdir(model_path)
            dtype = torch.get_default_dtype()
            # dtype = torch.float32
            self.model = load_model_from_cwd(model_device=model_device).to(dtype)
            os.chdir(current_dir)
            self.predictor = Predictor.from_graph(
                self.model, model_device=model_device, requires_grad=True
            )
            self.multi_targets = multi_targets
            self.coords_unit = coords_unit
            # eV to au
            self.energy_conversion = energy_conversion
            if coords_unit == "Angstrom":
                # Angstrom to au
                self.coords_conversion = 1.8897259886
            else:
                self.coords_conversion = 1
            self.R: torch.Tensor
            self.pred: dict[str, torch.Tensor]
            self.old_nacr: torch.Tensor
            self.old_dipole: torch.Tensor
            self.coords_change: bool

        def _check_coordinate_update(func: callable):
            """A decorator to check coordinates update. If there is a change in the
            coordinates, make new predictions. Otherwise, requested quantities will be
            extracted from the existing dictionary.

            Args:
                func (callable): methods in the NNDriver class
            """

            def wrapped_func(self, *args):
                coord = self.get_R()
                # the coordinate is updated, make new predictions
                if not hasattr(self, "R") or torch.not_equal(self.R, coord).any():
                    self.R = coord
                    # print("prediction")
                    self.make_predictions()
                    self.coords_change = True
                else:
                    self.coords_change = False
                return func(self, *args)

            return wrapped_func

        def get_Z(self):
            r"""Convert the element symbols to a tensor of atomic numbers (Z). This only
            need to be run once. when the class is initialized."""
            Z = [[periodictable[a]["z"] for a in mol.elements] for mol in self.mol]
            # lower case tensor to keep the int type
            self.Z = torch.tensor(Z)
            del Z

        def get_R(self):
            r"""Convert the molecular positions to a tensor of coordinates (R). This
            conversion needs to be done every time when the coordinates are updated,
            for example, in MD simulations.
            """
            R = [_.coords for _ in self.mol]
            # R = [_.atom_coords(unit=self.coords_unit) for _ in self.mol]
            # converting list of np.nparray to tensor is extremely slow
            # could use some optimizations here
            # convert unit from a.u. to Angstrom
            return torch.Tensor(np.array(R)) / self.coords_conversion

        # At this moment the model is assumed to
        #   1. have the same number of states as the simulation
        #   2. use multi targets, i.e., one node to predict all states
        #   3. have the node name I normally use
        # TODO: node name and no of states should be determined at least semi-automatically
        def make_predictions(self):
            r"""Make predictions for electronic structure properties based on the geometry of
            the input molecule or list of molecules. The predicted results are saved
            within the class, as NNDriver.pred.

            """
            self.pred = self.predictor(Z=self.Z, R=self.R)
            if self.ml_model_nstates > self.nstates:
                for k, v in self.pred.items():
                    if type(k) is not str:
                        continue
                    if k == "E" or k == "D":
                        self.pred[k] = v[:, : self.nstates]
                    elif "NACR" in k.upper():
                        self.pred[k] = v[:, self.slices]
            self.pred["E"] *= self.energy_conversion
            self.e = self.pred["E"]

        def _nuc_grad(self):
            e = self.pred["E"]
            force = []
            for i in range(self.nstates):
                force.append(
                    torch.autograd.grad(e[:, i].sum(), self.R, retain_graph=True)[0]
                )
            force = -torch.stack(force, dim=1) / self.coords_conversion
            self.pred["F"] = force
            return self.pred["F"]

        def _assign_forces(self, molecule: List[Molecule], force: torch.Tensor):
            for i, mol in enumerate(molecule):
                for j in range(self.nstates):
                    mol.states[j].forces = force[i, j]

        @_check_coordinate_update
        def calculate_forces(self):
            if "F" not in self.pred:
                forces = self._nuc_grad().detach().numpy()
                self._assign_forces(self.mol, forces)
            else:
                forces = self.pred["F"].detach().numpy()
            return forces

        @_check_coordinate_update
        def get_nact(self, veloc: torch.Tensor):
            r"""Return the non-adiabatic coupling terms (NACT) between all pairs of excited
               states. Calculated from NACR and nuclear velocities.

            .. math::
               :nowrap:

               \begin{align*}
                  NACT(t_i) = NACR(t_i) \cdot v(t_i)
               \begin{align*}

            :param nacr: NACR. Shape (n_molecules, n_state_pairs, natoms, 3).
            :type nacr: torch.Tensor
            :return: NACT. Shape (n_molecules, n_state_pairs).
            :rtype: torch.Tensor
            """
            # v = [_.veloc for _ in self.mol]
            # v = torch.Tensor(np.array(v))
            if "nact" not in self.pred:
                if not isinstance(veloc, torch.Tensor):
                    veloc = torch.Tensor(veloc)
                n_molecules, n_atoms, n_dims = veloc.shape
                # reshape velocities for batched matrix multiplications
                veloc = veloc.reshape(n_molecules, 1, 1, n_atoms * n_dims)
                self._get_nacr()
                nacr = self.pred["NACR"]
                nacr = nacr.reshape(
                    n_molecules, len(self.state_pairs), n_atoms * n_dims, 1
                )
                # resulting a tensor with a shape of (n_molecules, n_pairs, 1, 1)
                # use squeeze to remove 1's
                # TODO: for torch < 2.0, torch.squeeze only accepts one integer as the argument. We can restore the tuple way if we decide to force a dependency of torch >= 2.0. At this moment, hippynn only requires torch >= 1.9, so we will do two .squeeze operation instead.
                # self.pred["nact"] = torch.matmul(veloc, nacr).squeeze(dim=(2, 3))
                self.pred["NACT"] = (
                    torch.matmul(veloc, nacr).squeeze(dim=3).squeeze(dim=2)
                )
            return self.reshape_nac(self.pred["NACT"])

        @_check_coordinate_update
        def get_nact2(self, veloc: torch.Tensor):
            r"""Return the non-adiabatic coupling terms (NACT) between all pairs of excited
               states. Calculated from NACR and nuclear velocities.

            .. math::
               :nowrap:

               \begin{align*}
                  NACT(t_i) = NACR(t_i) \cdot v(t_i)
               \begin{align*}

            :param nacr: NACR. Shape (n_molecules, n_state_pairs, natoms, 3).
            :type nacr: torch.Tensor
            :return: NACT. Shape (n_molecules, n_state_pairs).
            :rtype: torch.Tensor
            """
            # v = [_.veloc for _ in self.mol]
            # v = torch.Tensor(np.array(v))
            if "nact" not in self.pred:
                if not isinstance(veloc, torch.Tensor):
                    veloc = torch.Tensor(veloc)
                self._get_nacr()
                nacr = self.pred["NACR"]
                self.pred["NACT"] = torch.einsum("ijkl, ikl -> ij", nacr, veloc)
            return self.reshape_nac(self.pred["NACT"])

        @_check_coordinate_update
        def _get_nacr(self):
            r"""Return the non-adiabatic coupling vectors (NACR) between all pairs of excited
                states.

            :return: NACR. Shape (n_molecules, n_state_pairs, natoms, 3).
            :rtype: torch.Tensor
            """
            if "NACR" not in self.pred:
                # only take excited state energies
                # e = self.get_energies()[:, 1:]
                # if e is None:
                #     e = self.e
                # e = e[:, 1:]
                # direct hippynn output is NACR * dE
                # in the shape of (n_molecules, npairs, natoms * ndim)
                nacr_de = self.pred["NACRdE"]
                # nacr_de = self.pred["ScaledNACR"]
                de = []
                for i, j in self.state_pairs:
                    # energy difference between two states
                    de.append(self.e[:, j] - self.e[:, i])
                de = torch.stack(de, dim=1)
                nacr = nacr_de / de.unsqueeze(2)
                # NACR in NEXMD should have an unit of 1/A
                # the numerator is in eV/A
                # as the energies are already converted to a.u.
                # we convert both eV and A in the numerator to a.u.
                # nacr *= self.energy_conversion * self.coords_conversion
                # nacr /= self.coords_conversion
                if hasattr(self, "old_nacr"):
                    wrong_phase = torch.einsum("ijk,ijk -> ij", nacr, self.old_nacr) < 0
                    nacr[wrong_phase] *= -1
                self.old_nacr = nacr.detach().clone()
                # reshape into (n_molecules, npairs, natoms, ndim)
                nacr = nacr.reshape(*nacr.shape[:2], -1, 3)
                self.pred["NACR"] = nacr

        def phase_correction(
            self,
        ):
            pass

        def get_nacr(self):
            self._get_nacr()
            return self.reshape_nac(self.pred["NACR"])

        def reshape_nac(self, nac: torch.Tensor):
            nac = nac.detach().numpy()
            n_mol = nac.shape[0]
            n_exc_states = self.nstates - 1
            nac_mat = np.zeros((n_mol, n_exc_states, n_exc_states, *nac.shape[2:]))
            for i in range(n_mol):
                count = 0
                for j in range(n_exc_states):
                    for k in range(j + 1, n_exc_states):
                        nac_mat[i, j, k] = nac[i, count]
                        nac_mat[i, k, j] = -nac[i, count]
                        count += 1
            return nac_mat

        # current model is in eV
        # consider retrain the model
        @_check_coordinate_update
        def get_energies(self):
            r"""Return the molecular energies for all molecules and all states (the ground
                state and *nstates* excited states).

            :return: molecular energies. Shape (n_molecules, nstates + 1).
            :rtype: torch.Tensor
            """
            self.e = self.pred["E"]
            e = self.e.detach().numpy()
            self._assign_energies(self.mol, e)
            return e

        def _assign_energies(self, molecule: List[Molecule], energy: torch.Tensor):
            for i, mol in enumerate(molecule):
                for j in range(self.nstates):
                    mol.states[j].energy = energy[i, j]

        def update_potential(self):
            e = self.pred["E"]
            for i, mol in enumerate(self.mol):
                self._assign_energies(mol, e[i])

        @_check_coordinate_update
        def get_dipoles(self):
            r"""Return the transition dipoles of all states.

            :return: transition dipoles. Shape (n_molecules, n_states, 3).
            :rtype: torch.Tensor
            """
            d = self.pred["D"]
            if hasattr(self, "old_dipole"):
                wrong_phase = torch.einsum("ijk,ijk -> ij", d, self.old_dipole) < 0
                d[wrong_phase] *= -1
            d = d.detach()
            self.old_dipole = d.clone()
            return d.numpy()

        @_check_coordinate_update
        def get_dipole_grad(self):
            r"""Return the gradients of transition dipoles

            :return: the gradients of the transition dipoles.
                Shape (n_molecules, n_states, natoms, 3).
            :rtype: torch.Tensor
            """
            if "dD" not in self.pred:
                self.get_dipoles()
                d = self.pred["D"]
                d_grad = []
                for i in range(self.nstates):
                    d_grad.append(
                        torch.autograd.grad(d[:, i].sum(), self.R, retain_graph=True)[0]
                    )
                self.pred["dD"] = torch.stack(d_grad, dim=1)
            return self.pred["dD"].detach().cpu().numpy()
