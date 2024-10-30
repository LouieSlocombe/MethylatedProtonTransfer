from mace.calculators import mace_anicc, mace_mp, mace_off
from ase.optimize import BFGS, FIRE
from ase.optimize.bfgslinesearch import BFGSLineSearch
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
import getpass
from ase.constraints import Hookean
import os
from ase.constraints import FixAtoms
from ase.build import molecule

plt.rcParams['axes.linewidth'] = 2.0


def remove_cell(atoms):
    # Remove the cell
    atoms.set_cell([0, 0, 0])
    # Remove the periodic boundary conditions
    atoms.set_pbc([0, 0, 0])
    # move the atoms to the centre of the cell
    atoms.translate(-atoms.get_center_of_mass())
    return None


def n_plot(xlab, ylab, xs=14, ys=14):
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def get_fmax(atoms):
    return np.linalg.norm(atoms.get_forces(), axis=1).max()


# sets the chosen atom to be half way between the two atoms
def set_halfway(atoms, idx0, idx1, idx2):
    pos1 = atoms.get_positions()[idx1]
    pos2 = atoms.get_positions()[idx2]
    atoms[idx0].position = ((pos1 + pos2) / 2.0)
    return None


# set the chosen atom to be some fraction of the way between the two atoms
def set_fraction(atoms, idx0, idx1, idx2, fraction):
    pos1 = atoms.get_positions()[idx1]
    pos2 = atoms.get_positions()[idx2]
    atoms[idx0].position = (pos1 + (pos2 - pos1) * fraction)
    return None


def make_tautomer(atoms, calc, idx_h, idx_a, idx_b, f_max=0.001, f_view=True):
    """
    Make the tautomer by swapping the bond lengths between the hydrogen and the two atoms
    Assumes A...H-B and converts to A-H...B
    """
    traj_tmp = "taut_maker.traj"
    atoms = atoms.copy()
    # Get the bond length between idx1 and idx0
    bond_length = atoms.get_distance(idx_a, idx_h)
    # Swap the bond lengths
    atoms.set_distance(idx_b, idx_h, bond_length, fix=0)

    # Attach the calculator
    atoms.set_calculator(calc)

    # Constrain all the atoms but the hydrogen
    fix_indices = list(range(len(atoms)))
    fix_indices.remove(idx_h)
    c = FixAtoms(indices=fix_indices)
    # Constrain all the atoms but the hydrogen
    atoms.set_constraint(c)
    # Optimise the geometry
    BFGSLineSearch(atoms, trajectory=traj_tmp).run(fmax=f_max)
    # Load the optimised geometry
    atoms = read(traj_tmp, index=":")
    if f_view:
        os_view(atoms)
    # Remove temp file
    os.remove(traj_tmp)
    atoms = atoms[-1]
    # Remove the constraint
    atoms.set_constraint()
    return atoms


def make_tautomers(atoms, calc, idx_h, idx_a, idx_b, f_max=0.001, f_view=True):
    """
    Make the tautomer by swapping the bond lengths between the hydrogen and the two atoms
    Assumes A...H-B and converts to A-H...B
    """
    traj_tmp = "taut_maker.traj"
    atoms = atoms.copy()
    # Get the bond length
    bond_length_1 = atoms.get_distance(idx_a[0], idx_h[0])
    bond_length_2 = atoms.get_distance(idx_a[1], idx_h[1])
    # Swap the bond lengths
    atoms.set_distance(idx_b[0], idx_h[0], bond_length_1, fix=0)
    atoms.set_distance(idx_b[1], idx_h[1], bond_length_2, fix=0)

    # Attach the calculator
    atoms.set_calculator(calc)

    # Constrain all the atoms but the hydrogen
    fix_indices = list(range(len(atoms)))
    fix_indices.remove(idx_h[0])
    fix_indices.remove(idx_h[1])
    c = FixAtoms(indices=fix_indices)
    # Constrain all the atoms but the hydrogen
    atoms.set_constraint(c)
    # Optimise the geometry
    BFGSLineSearch(atoms, trajectory=traj_tmp).run(fmax=f_max)
    # Load the optimised geometry
    atoms = read(traj_tmp, index=":")
    if f_view:
        os_view(atoms)
    # Remove temp file
    os.remove(traj_tmp)
    atoms = atoms[-1]
    # remove the constraint
    atoms.set_constraint()
    return atoms


# check if running on windows or linux
def os_show(check=False):
    if os.name == "nt" or check:
        plt.show()
    else:
        plt.close()
    return None


# check if running on windows or linux
def os_view(atoms, check=False):
    if os.name == "nt" or check:
        view(atoms)
    return None


def calc_values(md):
    e_pot = [atoms.get_potential_energy() for atoms in md]
    e_kin = [atoms.get_kinetic_energy() for atoms in md]
    e_tot = [atoms.get_total_energy() for atoms in md]
    temp = [atoms.get_temperature() for atoms in md]
    return e_pot, e_kin, e_tot, temp


def calc_spring_force(atoms, idx1, idx2, k, f_si=True):
    # check if atoms is a list
    if isinstance(atoms, list):
        # Get the distance between the two pushed atoms
        distance = [atoms.get_distance(idx1, idx2) for atoms in atoms]
    else:
        # Get the distance between the two pushed atoms
        distance = atoms.get_distance(idx1, idx2)
    # Calculate the spring force
    spring_force = np.abs(np.multiply(distance, k))
    if f_si:
        # Convert to pN
        spring_force = np.max(spring_force) * units._e * units.m * 1e12
    return spring_force


def make_methyl(atoms, calc, idx, f_max=0.001, f_view=True):
    traj_tmp = "methyl_maker.traj"
    atoms = atoms.copy()
    # Make the methyl group
    meth = molecule("CH3")
    meth.rotate(-45, "y")

    # find the closest atom to the atom to be replaced
    dist = atoms.get_distances(idx, indices=list(range(len(atoms))))
    # find the index of the closest atom
    idx_close = np.argmin(np.delete(dist, idx))

    # set the distance between the carbon atom in the methyl group and the closest atom
    atoms.set_distance(idx_close, idx, 1.5, fix=0)

    # get the position of the atom to be replaced
    pos = atoms[idx].position
    # set the position of the methyl group
    meth.set_positions(meth.get_positions() + pos)

    # remove the atom to be replaced
    atoms.pop(idx)
    # add the methyl group
    atoms += meth

    # Attach the calculator
    atoms.set_calculator(calc)

    # Constrain all the atoms but the methyl group
    fix_indices = list(range(len(atoms)))
    n_meth = len(meth)
    fix_indices = fix_indices[:-n_meth]
    c = FixAtoms(indices=fix_indices)
    # Constrain all the atoms but the methyl
    atoms.set_constraint(c)

    # Optimise the geometry
    BFGSLineSearch(atoms, trajectory=traj_tmp).run(fmax=f_max)
    # Load the optimised geometry
    atoms = read(traj_tmp, index=":")
    if f_view:
        os_view(atoms)

    # Remove temp file
    os.remove(traj_tmp)
    atoms = atoms[-1]

    # Remove the constraint
    atoms.set_constraint()
    return atoms


user = getpass.getuser()
file_path = r"C:\Users\{}\OneDrive - University of Surrey\Papers\paper_methyl\structures\mace".format(user)

f_max = 0.01  # maximum force following geometry optimisation
n_steps = 400  # Number of steps to run
n_steps_pre = 20  # Number of steps to run before the main run
dt = 1.0  # Time step in fs
friction = 0.1  # Friction coefficient in 1/fs
temperature = 300.0  # temperature in K
neb_climb = False
neb_file = "neb.traj"
md_file = "_mdsplit.traj"
md_file_pre = "_mdsplit_pre.traj"
md_type = "Langevin"  # Pick: VelocityVerlet Langevin

# Set the options for the spring pushing the system apart
push_spring_const = -0.2  # k (eV Angstrom^-2) possible values are -0.01

# Set the option for the spring keeping the helicase residue in place
hel_spring_const = 5.0  # k (eV Angstrom^-2) corresponding to a C-C bond
hel_rt = 1.54  # Resting length (Angstrom) corresponding to a C-C bond

N_images = 11
f_make_methyl = False
f_make_taut = False
f_opti_r = False
f_opti_p = False
f_opti_load = "-1"  # 0 or -1
f_run_neb = False
f_run_md = False
f_pre_md = False
f_md_select = True

calc = mace_anicc()
#calc = mace_off(model="small", default_dtype="float32", dispersion=False)
print("torch.cuda.is_available() :", torch.cuda.is_available())

# f_name_r = "N624_w000.traj"
# f_name_p = "N624_w011.traj"

# f_name_r = "A624_w000.traj"
# f_name_p = "A624_w011.traj"

f_name_r = "N624_O6mG_w000.traj"
f_name_p = "N624_O6mG_w011.traj"

# f_name_r = "N624_5mC_w000.traj"
# f_name_p = "N624_5mC_w011.traj"

f_name_md_init = f_name_r

if f_name_md_init == "N624_w000.traj" or f_name_md_init == "N624_w011.traj":
    push_const_idx = [1, 17] # Index of the pushed atoms
    hel_const_idx = 30  # Index of the helicase residue
    taut_idx_h = [27, 9]
    taut_idx_a = [11, 24]
    taut_idx_b = [26, 8]
elif f_name_md_init == "N624_O6mG_w000.traj" or f_name_md_init == "N624_O6mG_w011.traj":
    push_const_idx = [1, 16] # Index of the pushed atoms
    hel_const_idx = 29  # Index of the helicase residue
    taut_idx_h = [26, 4]
    taut_idx_a = [8, 23]
    taut_idx_b = [25, 3]
elif f_name_md_init == "N624_5mC_w000.traj" or f_name_md_init == "N624_5mC_w011.traj":
    push_const_idx = [1, 17] # Index of the pushed atoms
    hel_const_idx = 29 # Index of the helicase residue
    taut_idx_h = [26, 9]
    taut_idx_a = [11, 23]
    taut_idx_b = [25, 8]

full_path_r = os.path.join(file_path, f_name_r)
full_path_p = os.path.join(file_path, f_name_p)
full_path_md_init = os.path.join(file_path, f_name_md_init)
full_path_neb = os.path.join(file_path, neb_file)
full_path_md = full_path_md_init.split(".")[0][:-3] + md_file
full_path_md_pre = full_path_md_init.split(".")[0][:-3] + md_file_pre

if f_make_methyl:
    # Load the starting geometry
    atoms = read(full_path_md_init, index="-1")
    # Clean the cell
    remove_cell(atoms)
    # Remove previous constraints
    atoms.set_constraint()
    atoms = make_methyl(atoms, calc, 21)
    os_view(atoms)
    # Write the tautomer
    write("N624_5mC_w000.traj", atoms)

if f_make_taut:
    # Load the starting geometry
    atoms = read(full_path_md_init, index="-1")
    # Clean the cell
    remove_cell(atoms)
    # Remove previous constraints
    atoms.set_constraint()
    atoms = make_tautomers(atoms, calc, taut_idx_h, taut_idx_a, taut_idx_b)
    os_view(atoms)
    # Write the tautomer
    write(full_path_md_init.split(".")[0][:-2] + "11_check.traj", atoms)

if f_opti_r:
    print("Running react")
    react = read(full_path_r, index=f_opti_load)
    # Clean the cell
    remove_cell(react)
    # Remove previous constraints
    react.set_constraint()

    # # Add constraints to the backbone
    # c_r0 = Hookean(a1=push_const_idx[0], a2=react[push_const_idx[0]].position, rt=hel_rt, k=hel_spring_const)
    # c_r1 = Hookean(a1=push_const_idx[1], a2=react[push_const_idx[1]].position, rt=hel_rt, k=hel_spring_const)
    # # Set the constraint for the helicase residue to keep it in place
    # c_hel = Hookean(a1=hel_const_idx, a2=react[hel_const_idx].position, rt=hel_rt, k=hel_spring_const)
    # react.set_constraint([c_r0, c_r1, c_hel])

    react.set_constraint(FixAtoms(indices=[push_const_idx[0], push_const_idx[1], hel_const_idx]))

    react.calc = calc
    t0 = t.time()
    energy = react.get_potential_energy()
    fmax = get_fmax(react)
    t1 = t.time()
    total_time = t1 - t0
    print("React energy: {:.2f}".format(energy))
    print("Fmax: {:.2f}".format(fmax))
    print("Total time taken: {:.2f} s, {:.2f} m".format(total_time, total_time / 60.0))
    if fmax > f_max:
        BFGSLineSearch(react, trajectory=full_path_r).run(fmax=f_max)
        react = read(full_path_r, index=":")
        os_view(react)

if f_opti_p:
    print("Running prod")
    prod = read(full_path_p, index=f_opti_load)
    # Clean the cell
    remove_cell(prod)
    # Remove previous constraints
    prod.set_constraint()

    # # Add constraints to the backbone
    # c_r0 = Hookean(a1=push_const_idx[0], a2=prod[push_const_idx[0]].position, rt=hel_rt, k=hel_spring_const)
    # c_r1 = Hookean(a1=push_const_idx[1], a2=prod[push_const_idx[1]].position, rt=hel_rt, k=hel_spring_const)
    # # Set the constraint for the helicase residue to keep it in place
    # c_hel = Hookean(a1=hel_const_idx, a2=prod[hel_const_idx].position, rt=hel_rt, k=hel_spring_const)
    # prod.set_constraint([c_r0, c_r1, c_hel])
    prod.set_constraint(FixAtoms(indices=[push_const_idx[0], push_const_idx[1], hel_const_idx]))

    prod.calc = calc
    t0 = t.time()

    energy = prod.get_potential_energy()
    fmax = get_fmax(prod)
    t1 = t.time()
    total_time = t1 - t0
    print("React energy: {:.2f}".format(energy))
    print("Fmax: {:.2f}".format(fmax))
    print("Total time taken: {:.2f} s, {:.2f} m".format(total_time, total_time / 60.0))
    if fmax > f_max:
        BFGSLineSearch(prod, trajectory=full_path_p).run(fmax=f_max)
        prod = read(full_path_p, index=":")
        os_view(prod)

if f_run_neb:
    print("Running NEB")
    # Load the optimised geometries
    react = read(full_path_r, index="-1")
    prod = read(full_path_p, index="-1")
    # Attach the calculator
    react.calc = calc
    prod.calc = calc

    # Create the NEB images
    images = [react]
    for ii in range(N_images - 2):
        images.append(react.copy())
    images += [prod]

    # Attach the calculator to the images
    for image in images:
        image.calc = copy.copy(calc)

    # Interpolate the images
    neb = NEB(images, climb=neb_climb)
    neb.interpolate()
    neb.interpolate(method="idpp")
    os_view(neb.images)

    input("Viewing: NEB input. Press Enter to continue...")
    array_energies = np.zeros(N_images, dtype=float)
    array_max_forces = np.zeros(N_images, dtype=float)
    for i, image in enumerate(neb.images):
        image.calc = copy.copy(calc)
        array_energies[i] = image.get_potential_energy()
        array_max_forces[i] = get_fmax(image)
        print("energy: {:.2f} eV, forces: {:.2f} eV/A".format(array_energies[i], array_max_forces[i]))
    array_energies = array_energies - min(array_energies)
    plt.plot(array_energies)
    n_plot("Image", "Energy (eV)")
    plt.savefig("interpol_energies.pdf")
    os_show()
    plt.plot(array_max_forces)
    n_plot("Image", "Max Force (eV/A)")
    plt.savefig("interpol_forces.pdf")
    os_show()

    # Run the NEB
    BFGS(neb, trajectory=full_path_neb).run(fmax=f_max)
    os_view(neb.images)
    array_energies = np.zeros(N_images, dtype=float)
    array_max_forces = np.zeros(N_images, dtype=float)
    for i, image in enumerate(neb.images):
        image.calc = copy.copy(calc)
        array_energies[i] = image.get_potential_energy()
        array_max_forces[i] = get_fmax(image)
        print("energy: {:.2f}eV forces: {:.2f}eV/A".format(array_energies[i], array_max_forces[i]))
    array_energies = array_energies - min(array_energies)
    plt.plot(array_energies)
    n_plot("Image", "Energy (eV)")
    plt.savefig("neb_energies.pdf")
    os_show()
    plt.plot(array_max_forces)
    n_plot("Image", "Max Force (eV/A)")
    plt.savefig("neb_forces.pdf")
    os_show()

if f_run_md:
    # Load the starting geometry
    atoms = read(full_path_md_init, index="-1")
    # Clean the cell
    remove_cell(atoms)
    # Remove previous constraints
    atoms.set_constraint()
    os_view(atoms)

    # Set the calculator
    atoms.set_calculator(calc)

    # Set the momenta
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # zero linear momentum
    ZeroRotation(atoms)  # zero angular momentum

    if f_pre_md:
        print("Running pre MD")
        if md_type == "VelocityVerlet":
            VelocityVerlet(atoms,
                           timestep=dt * units.fs,
                           trajectory=full_path_md_pre,
                           logfile="-",
                           loginterval=1).run(n_steps_pre)
        elif md_type == "Langevin":
            Langevin(atoms,
                     timestep=dt * units.fs,
                     temperature_K=temperature,
                     friction=friction / units.fs,
                     trajectory=full_path_md_pre,
                     logfile="-",
                     loginterval=1).run(n_steps_pre)

        # Load the md trajectory
        atoms = read(full_path_md_pre, index=":")
        # os_view(atoms)
        atoms = atoms[-1]
        # Set the calculator
        atoms.set_calculator(calc)

    # Set the constraint for the pushed atoms
    c_push = Hookean(a1=push_const_idx[0], a2=push_const_idx[1], rt=0.0, k=push_spring_const)
    # Set the constraint for the helicase residue to keep it in place
    c_hel = Hookean(a1=hel_const_idx, a2=atoms[hel_const_idx].position, rt=hel_rt, k=hel_spring_const)
    atoms.set_constraint([c_push, c_hel])

    # Set the momenta
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # zero linear momentum
    ZeroRotation(atoms)  # zero angular momentum

    print("Running MD")
    if md_type == "VelocityVerlet":
        VelocityVerlet(atoms,
                       timestep=dt * units.fs,
                       trajectory=full_path_md,
                       logfile="-",
                       loginterval=1).run(n_steps)
    elif md_type == "Langevin":
        Langevin(atoms,
                 timestep=dt * units.fs,
                 temperature_K=temperature,
                 friction=friction / units.fs,
                 trajectory=full_path_md,
                 logfile="-",
                 loginterval=1).run(n_steps)

    # Load the md trajectory
    md = read(full_path_md, index=":")
    # Write the md trajectory as an xyz file
    write(full_path_md.split(".")[0] + ".xyz", md)

    os_view(md)
    # Create the time array
    time_array = np.arange(0, len(md) * dt, dt)

    # Get the potential, kinetic, and total energies
    e_pot, e_kin, e_tot, temp = calc_values(md)

    # Determine the statistical fluctuation in the temperature
    temp_expect_std = temperature * 2.0 / (3.0 * np.sqrt(len(atoms)))
    print("Expected temperature std: {:.2f} [K]".format(temp_expect_std))

    # Calculate the temperature std
    temp_std = np.std(temp)
    print("Temperature std: {:.2f} [K]".format(temp_std))

    # Get the distance between the two pushed atoms
    distance = [atoms.get_distance(push_const_idx[0], push_const_idx[1]) for atoms in md]

    # get the extension of the spring
    extension = np.array(distance) - distance[0]

    # Calculate the spring force
    spring_force = np.abs(np.multiply(distance, push_spring_const))
    # Convert to pN
    si_spring_force = calc_spring_force(md, push_const_idx[0], push_const_idx[1], push_spring_const, f_si=True)
    print("Max spring force: {:.2f} pN".format(si_spring_force))

    # Get the maximum force
    force_max = [get_fmax(atoms) for atoms in md]

    plt.plot(time_array, e_kin)
    n_plot("Time (fs)", "Kinetic Energy (eV)")
    os_show()

    plt.plot(time_array, e_pot)
    n_plot("Time (fs)", "Potential Energy (eV)")
    os_show()

    plt.plot(time_array, e_tot)
    n_plot("Time (fs)", "Total Energy (eV)")
    os_show()

    plt.plot(time_array, temp)
    n_plot("Time (fs)", "Temperature (K)")
    os_show()

    plt.plot(time_array, distance)
    n_plot("Time (fs)", "Distance (Angstrom)")
    os_show()

    plt.plot(time_array, extension)
    n_plot("Time (fs)", "Extension (Angstrom)")
    os_show()

    plt.plot(time_array, force_max, color="black")
    plt.plot(time_array, spring_force, color="red")
    n_plot("Time (fs)", "Max Force (eV/Angstrom)")
    os_show()

if f_md_select:
    # Load the starting geometry
    md = read(full_path_md, index=":")

    # Create the time array
    time_array = np.arange(0, len(md) * dt, dt)

    # Get the distance between the two pushed atoms
    distance = [atoms.get_distance(push_const_idx[0], push_const_idx[1]) for atoms in md]

    # get the extension of the spring
    extension = np.array(distance) - distance[0]

    # Pick points evenly spaced along the trajectory
    n_pick = 5
    idx_pick = np.round(np.linspace(0, len(md) - 1, n_pick)).astype(int)
    md_pick = [md[i] for i in idx_pick]
    # view(md_pick)

    plt.plot(time_array, extension)
    plt.scatter(time_array[idx_pick], extension[idx_pick], color="red")
    n_plot("Time (fs)", "Extension (Angstrom)")
    os_show()

    # Loop over the picked geometries
    for i, atoms in enumerate(md_pick):
        ext = extension[idx_pick[i]]
        print("Geometry: ", i, " Extension: {:.2f} A".format(ext))

        # Add constraints to the backbone
        atoms.set_constraint(FixAtoms(indices=[push_const_idx[0], push_const_idx[1], hel_const_idx]))
        # Write the picked geometry
        write(full_path_md_init.split(".")[0][:-3] + "_can_mdsplit_{:.3f}.traj".format(ext), atoms)
        # Create the tautomer
        tautomer = make_tautomers(atoms, calc, taut_idx_h, taut_idx_a, taut_idx_b)
        tautomer.set_constraint(FixAtoms(indices=[push_const_idx[0], push_const_idx[1], hel_const_idx]))
        # Write the tautomer
        write(full_path_md_init.split(".")[0][:-3] + "_taut_mdsplit_{:.3f}.traj".format(ext), tautomer)
        os_view(atoms)


