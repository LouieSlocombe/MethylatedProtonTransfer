import numpy as np
from ase.io import read, write
from ase.visualize import view
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.build import molecule
from ase.visualize import view as os_view
import sys
from mace.calculators import mace_anicc, mace_mp, mace_off
import os


def make_methyl(atoms, calc, idx, f_max=0.001, f_view=True):
    traj_tmp = "methyl_maker.traj"
    atoms = atoms.copy()
    # Make the methyl group
    meth = molecule("CH3")

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

    atoms.calc = calc
    # Constrain all the atoms but the methyl group
    fix_indices = list(range(len(atoms)))
    n_meth = len(meth)
    fix_indices = fix_indices[:-n_meth]
    c = FixAtoms(indices=fix_indices)
    # Constrain all the atoms but the methyl
    atoms.set_constraint(c)
    view(atoms)
    # Optimise the geometry
    BFGS(atoms, trajectory=traj_tmp).run(fmax=f_max)
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


# Attach the calculator
calc = mace_off(model="large", default_dtype="float64", device="cpu")

atoms = read("cyto_react.traj", index="-1")
view(atoms)

idx = 4
methylated_atoms = make_methyl(atoms, calc, idx)
# Write a file a traj file with the methylated atoms
write("m5_cyto_react_corrected.traj", methylated_atoms)