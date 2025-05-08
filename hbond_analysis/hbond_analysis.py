# standard packages
import os
import math
import numpy as np
import pandas as pd
import itertools
import statistics as stats

# MD analysis packages
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.topology.tpr import setting
from MDAnalysis.analysis.distances import dist
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

# visualisation packages
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.axes import Axes


# functions
def hbonds_analysis(top, trj, donors, hydrogens, acceptors, DA_cutoff=3.5, DHA_angle_cutoff=150, start=None, stop=None, step=None, update=True):

    '''
    Calculate the hydrogen bonds in the system throughout the trajectory, and also returns the number of acceptor groups h-bonded to the donor group (see MDAnalysis' Hydrogen Bond Analysis documentation).

    ----------
    PARAMETERS
    ----------
    top : .tpr
        topology file
    trj : .xtc/.trr
        trajectory file
    donors : str
        hydrogen bond donors according to MDAnalysis' selection language
    hydrogens : str
        hydrogen atoms according to MDAnalysis' selection language
    acceptors : str 
        acceptor atoms according to MDAnalysis' selection language 
    DA_cutoff : float
        donor-acceptor distance cutoff (default = 3.5 Å)
    DHA_angle_cutoff : float
        donor-hydrogen-acceptor angle cutoff (150°)
    start : int
        frame to start analysis, "None" takes the first frame (default = None)
    stop : int
        frame to stop analysis, "None" takes the final frame (default = None)
    start : int
        step size to skip frames, "None" uses a step size of 1 (default = None)
    update_selections : bool
        updates atom selections if True (default = True)

    -------
    RETURNS
    -------
    hbonds : array
        hydrogen bond analysis output in the form [[frame, donor_id, hydrogen_id, acceptor_id, distance (Å), distance angle (°)]]
    '''
    
    # create universe
    u = mda.Universe(top, trj, refresh_offsets=True)

    # calculate all hydrogen bonds from selected atom groups
    hbonds= HydrogenBondAnalysis(universe=u, 
                                 donors_sel=donors,
                                 hydrogens_sel=hydrogens,
                                 acceptors_sel=acceptors,
                                 d_a_cutoff=DA_cutoff,
                                 d_h_a_angle_cutoff=DHA_angle_cutoff,
                                 update_selections=update)

    # run
    hbonds.run(start=start, stop=stop, step=step, verbose=True)

    print('Total H bonds:', len(hbonds.results.hbonds))
    print('Total unique H-bonds:', len(hbonds.count_by_ids()))
      
    return hbonds


def hbonds_to_csv(hbonds, path):

    '''
    Saves hydrogen bond analysis results to a .csv file.

    ----------
    PARAMETERS
    ----------
    hbonds : array
        hydrogen bond analysis output in the form [[frame, donor_id, hydrogen_id, acceptor_id, distance (Å), distance angle (°)]]
    path : str
        path to save .csv file

    '''
    # create dataframe 
    df = pd.DataFrame(hbonds.results.hbonds, 
                      columns=['Frame', 'Donor ID', 'Hydrogen ID', 'Acceptor ID', 'Distance (Å)', 'Angle (Degrees)']
                     )
    # save as csv
    df.to_csv(path, index=False)

    return None


def ab_distances(top, trj, atom_a, atom_b, path=None):

    '''
    Calculates distances between two selected atoms per frame in the trajectory file.

    ----------
    PARAMETERS
    ----------
    top : .tpr
        topology file
    trj : .xtc/.trr
        trajectory file
    atom_A: str
        atom A according to MDAnalysis' selection language
    atom_B : str
        atom B according to MDAnalysis' selection language
    path : str
        path to save calculated distances per frame in a .csv (default = None)

    -------
    RETURNS
    -------
    dists = list
        list of distances

    '''
    # create universe
    u = mda.Universe(top, trj, refresh_offsets=True)

    # select atoms
    n1 = u.select_atoms("resnum 1 and name N1", )
    n9 = u.select_atoms("resnum 24 and name N9")
    
    # calculate distance per timestep
    dists = []
    for ts in u.trajectory:
        # calculate distance with PBC
        dists.append(distance_array(n1, n9, box=ts.dimensions)[0][0])

    # calculate times
    dt = u.trajectory.dt
    n_frames = len(u.trajectory)
    times = [i * dt for i in range(n_frames)]

    # save to a csv
    if path != None:
        df = pd.DataFrame(data = {'Time (ns)': [i/1000 for i in times],
                                  'Distance (Å)': dists}
                         )
    
        df.to_csv(path, index=False)

    return dists



# top and trj files
top = '../MD.tpr'
trj = '../traj_comp.xtc'

# calculate backbone distances
print('------------------------------')
print('Calculating Backbone Distances')
print('------------------------------')

u = mda.Universe(top, trj, refresh_offsets=True)

n1 = "resnum 1 and name N1"
n9 = "resnum 24 and name N9"

dists = ab_distances(top, trj, n1, n9, path=f'results/dists.csv')

print(f'Number of frames where the N(1):N(9) distance is >= 10.08 A: {len([i for i in dists if i >= 10.08])}')

print()



print('----------------------')
print('Hydrogen Bond Analysis')
print('----------------------')
# distance and angle cutoffs 
DA_cutoff = 3.5
DHA_angle_cutoff = 150

# N1 in G to H2O
donor = 'resnum 24 and name N1'
hydrogen = 'resnum 24 and name H1'
acceptor = 'resname SOL and name OW'

hbonds_1 = hbonds_analysis(top, trj, donor, hydrogen, acceptor, DA_cutoff, DHA_angle_cutoff)
hbonds_to_csv(hbonds_1, f'results/hbonds-N1-OW.csv')

# H2O to O2 in C
donor = 'resname SOL and name OW'
hydrogen = 'resname SOL and name HW1 or name HW2'
acceptor = 'resnum 1 and name O2'

hbonds_2 = hbonds_analysis(top, trj, donor, hydrogen, acceptor, DA_cutoff, DHA_angle_cutoff)
hbonds_to_csv(hbonds_2, f'results/hbonds_OW-O2.csv')

# H2O to N3 in C
donor = 'resname SOL and name OW'
hydrogen = 'resname SOL and name HW1 or name HW2'
acceptor = 'resnum 1 and name N3'

hbonds_3 = hbonds_analysis(top, trj, donor, hydrogen, acceptor, DA_cutoff, DHA_angle_cutoff)
hbonds_to_csv(hbonds_3, f'results/hbonds_OW-N3.csv')



# comparing distances to hbonds for N1:OW
print('---------------------------------------------')
print('Comparing Distance and Hydrogen Bond Analysis')
print('---------------------------------------------')

# get frames
frames = [int(i[0]) for i in hbonds_1.results.hbonds]

count = 0
# count occurrences where the distance exceeds 1.4 A separation
for i in frames:
    print(dists[i])
    if dists[i]>=10.08:
        count+=1

print(f'Total number of hydrogen bonds when N(1)-N(9) distance is >= 10.08 is {count}')

df = pd.DataFrame(data={'Frames': frames,
                        'Distances (Å)': [dists[i] for i in frames]}
                  )

df.to_csv(f'results/hbonds_1_N_backbone_distances.csv', index=False)


# FIGURE 1
colors1=['teal', 'orange', 'gray']
colors2=['paleturquoise', 'bisque', 'lightgrey']

# create plot
fig, ax_ = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

# subplot
ax = plt.subplot(111)

x = np.arange(3)
xlabels = ['N1-OW', 'OW-O2', 'OW-N3']

ax.bar(x-0.175, 
       [len(hbonds_1.results.hbonds), len(hbonds_2.results.hbonds), len(hbonds_3.results.hbonds)],
       color=colors1,
       width=0.35)
ax.bar(x+0.175, 
       [len(hbonds_1.count_by_ids()), len(hbonds_2.count_by_ids()), len(hbonds_3.count_by_ids())],
       color=colors2,
       width=0.35)

# axes and labels
ax.set_xticks(x, xlabels)
ax.set_ylabel('Number of hydrogen bonds')# fontsize=10)
ax.set_yscale('log')
ax.set_ylim(1, 10000)
#ax.yaxis.set_major_locator(MultipleLocator(20))
#ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(which='major', width=2, length=5)# labelsize=10)
ax.tick_params(which='minor', width=1, length=3.5)

# values
for i, value in enumerate([len(hbonds_1.results.hbonds), len(hbonds_2.results.hbonds), len(hbonds_3.results.hbonds)]):
    ax.text(i-0.175, value + 1, str(value), ha='center', va='bottom')# fontsize=10)

for i, value in enumerate([len(hbonds_1.count_by_ids()), len(hbonds_2.count_by_ids()), len(hbonds_3.count_by_ids())]):
    ax.text(i+0.175, value + 1, str(value), ha='center', va='bottom')# fontsize=10)
        
# adjust formatting
plt.tight_layout()
# save figure
plt.savefig(f'figures/hbonds_barplot.png', dpi=300)


# FIGURE 2
# create plot
fig, ax = plt.subplots(figsize=(6, 4))

# plot histogram
ax.hist([i-8.67 for i in dists], bins=90, rwidth=(max(dists)-min(dists)/90), color='lightgrey', edgecolor='black')

# axes settings
ax.set_xlabel('Separation Distance (Å)')
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.set_ylabel('Frequency')

# add inset
ax_in = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
# plot histogram
ax_in.hist([i-8.67 for i in dists], bins=90, rwidth=(max(dists)-min(dists)/90), color='red', edgecolor='black')
# inset axes settings
ax_in.set_xlim(1.3, 3.6)
ax_in.xaxis.set_major_locator(MultipleLocator(0.7))
ax_in.xaxis.set_minor_locator(MultipleLocator(0.35))
ax_in.set_ylim(0, 8)

# adjust formatting
plt.tight_layout()
# save figure
plt.savefig(f'figures/dists_hist.png', dpi=300)


# generate pdb files for N1 atoms
for i in hbonds_1.results.hbonds:
    u.trajectory[int(i[0])]

    sel = u.select_atoms(f"not resname SOL or resnum {u.select_atoms(f'id {int(i[3])}').resnums[0]}", periodic=True)

    sel.write(f'N1-OW_pdbs/{int(i[0])}-{int(i[3])+1}.pdb', bonds=None)



print()
print('----')
print('DONE')
print('----')