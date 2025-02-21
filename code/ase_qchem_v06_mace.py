import numpy as np
from ase.calculators.qchem import QChem
import os
from ase.io import read, write
import time as t
import copy
from ase.visualize import view
from ase.optimize import BFGS
import matplotlib.pyplot as plt
from ase.neb import NEB
from catlearn.optimize.mlneb import MLNEB
from mace.calculators import mace_anicc #mace_off

plt.rcParams['axes.linewidth'] = 2.0

def boltz_prob(E, T=300.0):
    """
    Calculate the Boltzmann probability of a state given the energy and temperature.
    """
    kb_ev = 8.617333262145e-5
    return np.exp(-E / (kb_ev * T))

def n_plot(xlab, ylab, xs=14, ys=14):
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None

def remove_cell(atoms):
    atoms.set_cell([0, 0, 0])
    atoms.set_pbc([0, 0, 0])
    return None

def ml_interpolate(react, prod, calc, f_max=0.1):
    print("Running ML-NEB")
    # load the optimised geometries
    if f_opti_r is False:
        react = read(f_name_r, index="-1")
    if f_opti_p is False:
        prod = read(f_name_p, index="-1")

    react.calc = calc_ml
    prod.calc = calc_ml
    # create the NEB images
    images = [react]
    for ii in range(N_images - 2):
        tmp = react.copy()
        tmp.calc = calc_ml
        images.append(tmp)
    images += [prod]

    neb = NEB(images, k=8.0,
              method='eb')  # climb=False, method="spline", "eb"
    neb.interpolate()

    for i, image in enumerate(neb.images):
        image.calc = copy.copy(calc_ml)
        image.get_potential_energy()

    BFGS(neb, trajectory='neb.traj').run(fmax=f_max)
    return neb

def qchem_calc_preset(charge=0,
                      multiplicity=1,
                      xc="B3LYP",  # wB97X-V, B3LYP
                      basis="6-311++G**",  # 6-31G* 6-311G** 6-31G(d,p) 6-311++G**
                      n_t=10, #4 for laptop, 1 for desktop
                      f_fast=False,
                      f_solv=False,
                      f_disp=True,
                      f_neo=False,
                      neo_idx=None,
                      neo_epc='epc19',  # LDA epc172, GGA epc19
                      neo_preset="PB4-D",
                      neo_isotope="1",
                      scf_algorithm="DIIS_GDM",  # DIIS GDM DIIS_GDM
                      solv_extra=None,
                      scf_convergence=7         #default 5 for SPE and 8 for opt
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
        'scf_convergence': str(scf_convergence),
        'thresh': str(scf_convergence+4),
        'max_scf_cycles': "100",
        'scf_algorithm': scf_algorithm,
    }
    # inpt_dict.update({'': ''})

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
        inpt_dict.update({'NEO_N_SCF_CONVERGENCE': '4'})

# Add solvent extra
    if solv_extra is not None and f_solv is True:
        return QChem(solv_extra=solv_extra, **inpt_dict)
    else:
        return QChem(**inpt_dict)

solv_extra = """
    $solvent
        Dielectric 8.0 
    $end
    """

######### Load Structures #########
f_name_r = "GCcan.traj"
f_name_p = "GCtaut.traj"
neo_idx = [9, 27]

print("Building calculator")
calc = qchem_calc_preset(f_solv=True, solv_extra=solv_extra)
calc_neo_2 = qchem_calc_preset(f_solv=True, solv_extra=solv_extra, f_neo=True, neo_idx=neo_idx)
calc_ml = mace_anicc()

f_max = 0.05
N_images = 15
f_opti_r = True
f_opti_p = True
f_resub_neb = False
f_run_ml_interpolate = False
f_run_neb = True
f_run_neo = True


if f_opti_r:
    react = read(f_name_r, index="-1")
    remove_cell(react)
    react.calc = calc
    print("Running react")
    t0 = t.time()
    energy = react.get_potential_energy()
    fmax = np.max(react.get_forces())
    t1 = t.time()
    total_time = t1 - t0
    print("React energy: {:.2f}".format(energy))
    print("Fmax: {:.2f}".format(fmax))
    print("Total time taken: {:.2f} s, {:.2f} m".format(total_time, total_time / 60.0))
    BFGS(react, trajectory=f_name_r).run(fmax=f_max)

if f_opti_p:
    prod = read(f_name_p, index="-1")
    remove_cell(prod)
    prod.calc = calc_ml
    print("Running prod")
    t0 = t.time()
    energy = prod.get_potential_energy()
    fmax = np.max(prod.get_forces())
    t1 = t.time()
    total_time = t1 - t0
    print("Prod energy: {:.2f}".format(energy))
    print("Fmax: {:.2f}".format(fmax))
    print("Total time taken: {:.2f} s, {:.2f} m".format(total_time, total_time / 60.0))
    BFGS(prod, trajectory=f_name_p).run(fmax=f_max)

if f_run_ml_interpolate:
    react = read(f_name_r, index="-1")
    prod = read(f_name_p, index="-1")
    atoms = ml_interpolate(react, prod, calc_ml, f_max=f_max)
    #view(atoms)

if f_run_neb:
    print("Running NEB")
    # we check if the neb file exists
    if os.path.isfile("neb.traj"):
        f_resub_neb = True
        print("Resubmitting NEB")
    if f_resub_neb:
        # images = read("neb.traj", index=":15")
        # neb = NEB(images)
        # for i, image in enumerate(neb.images):
        #    image.calc = copy.copy(calc)
        #    image.get_potential_energy()
        # BFGS(neb, trajectory='neb.traj').run(fmax=f_max)
        # view(neb)

        neb_catlearn = MLNEB(start=read(f_name_r, index="-1"),
                             end=read(f_name_p, index="-1"),
                             ase_calc=calc,
                             n_images=N_images,
                             interpolation='idpp', restart=False)

        neb_catlearn.run(fmax=f_max, trajectory='neb.traj')

    else:
        # load the optimised geometries
        if f_opti_r is False:
            react = read(f_name_r, index="-1")
        if f_opti_p is False:
            prod = read(f_name_p, index="-1")

        if f_run_ml_interpolate:
            images = ml_interpolate(react, prod, calc_ml, f_max=0.05)
        else:
            react.calc = calc
            prod.calc = calc
            # create the NEB images
            images = [react]
            for ii in range(N_images - 2):
                tmp = react.copy()
                tmp.calc = calc
                images.append(tmp)
            images += [prod]

        neb = NEB(images, climb=True)   #method="spline"

        if f_run_ml_interpolate is False:
            neb.interpolate()
            neb.interpolate(method="idpp")
        #view(neb.images)

        energies = np.zeros(N_images, dtype=float)
        for i, image in enumerate(neb.images):
            image.calc = copy.copy(calc)
            energies[i] = image.get_potential_energy()
            print("energy: ", energies[i], " forces: ", np.max(image.get_forces()))
        energies = energies - min(energies)

        BFGS(neb, trajectory='neb.traj').run(fmax=f_max)

if f_run_neo:
    print("Running NEO")
    # load the last N NEB images
    images = read("neb.traj", index=str(-N_images) + ":")
    #view(images)
    energies = np.zeros([N_images, 3], dtype=float)
    for i, atoms in enumerate(images):
        atoms.calc = calc
        energies[i, 0] = atoms.get_potential_energy()
        atoms.calc = calc_neo_2
        energies[i, 1] = atoms.get_potential_energy()
        print("Index: ", i, " DFT: ", energies[i, 0], " NEO: ", energies[i, 1])

    e_diff_0 = energies[:, 0] - min(energies)
    e_diff_1 = energies[:, 1] - min(energies)
    print("DFT energies:")
    print(e_diff_0)
    print("NEO energies:")
    print(e_diff_1)

    # get the barrier heights
    b0 = np.max(e_diff_0)
    b1 = np.max(e_diff_1)
    print("DFT barrier: ", b0)
    print("NEO barrier: ", b1)

    a0 = e_diff_0[-1]
    a1 = e_diff_1[-1]
    print("DFT final: ", a0)
    print("NEO final: ", a1)


    kb_ev = 8.617333262145e-5
    kappa_1 = boltz_prob(b1) / boltz_prob(b0)
    print("kappa_1: ", kappa_1)

    plt.plot(energies[:, 0] - min(energies[:, 0]), label="DFT")
    plt.plot(energies[:, 1] - min(energies[:, 1]), label="NEO")
    n_plot("Image", "Energy (eV)")
    plt.legend()
    plt.savefig("neo_neb_energies.pdf")

