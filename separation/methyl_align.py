from mace.calculators import mace_anicc, mace_mp, mace_off
from ase.optimize import BFGS, FIRE
from ase.visualize import view
import torch
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md import Langevin
from ase.neb import NEB
import copy
import time as t
import numpy as np
import matplotlib.pyplot as plt
from ase.constraints import Hookean
from ase.build import minimize_rotation_and_translation
from ase.calculators.qchem_mod import QChem
from ase.build import molecule


def qchem_calc_preset(charge=0,
                      multiplicity=1,
                      xc="BLYP",  # wB97X-V B3LYP
                      basis="6-31G",  # 6-31G* 6-311G** 6-31G(d,p) 6-311++G**
                      n_t=10,
                      f_fast=False,
                      f_solv=False,
                      f_disp=False,
                      f_neo=False,
                      neo_idx=None,
                      neo_epc='epc172',  # LDA epc172, GGA epc19
                      neo_preset="PB4-D",
                      neo_isotope="1",
                      scf_algorithm="DIIS",  # DIIS GDM DIIS_GDM
                      solv_extra=None
                      ):
    if neo_idx is None:
        neo_idx = [0]
    inpt_dict = {
        'label': 'calc/data',
        'charge': charge,
        'multiplicity': multiplicity,
        'method': xc,
        'basis': basis,
        'n_t': n_t,
        'scf_convergence': "8",
        'max_scf_cycles': "100",
        'scf_algorithm': scf_algorithm,
    }

    if f_solv:
        inpt_dict.update({'solvent_method': 'PCM'})  # kirkwood, COSMO, PCM, SMD

    if f_disp:
        inpt_dict.update({'dft_d': 'D4'})
        # inpt_dict.update({'dft_d': 'D3_BJ'})

    if f_fast:
        inpt_dict.update({'fast_xc': 'True'})
        inpt_dict.update({'xc_smart_grid': 'True'})

    if f_neo:
        inpt_dict.update({'neo': 'True'})
        inpt_dict.update({'point_group_symmetry': 'False'})
        inpt_dict.update({'neo_epc': neo_epc})
        inpt_dict.update({'neo_preset': neo_preset})
        inpt_dict.update({'neo_idx': neo_idx})
        inpt_dict.update({'neo_isotope': neo_isotope})
    # Add solvent extra
    if solv_extra is not None and f_solv is True:
        return QChem(solv_extra=solv_extra, **inpt_dict)
    else:
        return QChem(**inpt_dict)


# fmax = 0.05  # maximum force following geometry optimisation
# f_name = "mGC_can_cheap_qchem_0.traj"  # "N624_O6mG_w000.traj"
# f_name = "N624_O6mG_w000.traj"
# f_name = "N624_O6mG_w011.traj"
# atoms = read(f_name)
# solv_extra = """
# $solvent
#     Dielectric 8.0
# $end"""
# calc = qchem_calc_preset(xc="B3LYP", basis="6-31G",f_solv=True, solv_extra=solv_extra)
# atoms.calc = calc
# print("energy: ", atoms.get_potential_energy())
# # opti = BFGS(atoms, trajectory="opti.traj")
# # opti.run(fmax=fmax)

atoms = [read("5mC-Gcan_opt.traj", index="-1")]
atoms.append(read("O6mG-Ccan_opt.traj", index="-1"))
view(atoms)
