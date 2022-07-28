## Description of files and folders

1) files in the main directory are the files required to make the figures in the main text and SI.

2) the community_simulator folder containts code from the Community simulator package [Marsland et al 2020 Plos One]. The folder with the package is provided because the package was modified for use in the simulations. Some of these modifications, such as using different ode solvers for simulation stability, may have been implemented in more recent versions of community simulator.

3) the 'vestigial files' folder contains code for related analysis of ecological landscapes that is not presented in the paper.

## Overview of simulation workflow

The script Submit_CR_SSjobs.py submits simulation jobs over the desired parameter range.
There are 10 replicate simulations for each set of parameters. A replicate varies only by the realization of the consumption matrices of the species (chosen randomly from appropriate distribution.) 
Each set of parameters has a separate directory, with simulations for each replicate in a separate sub-directory within this directoy. Thus files are stored in a tree-like directory form.

The script Submit_CR_SSjobs.py also runs analyze_landscape.py after the simulations are completed. analyze_landscape.py calculates the various ruggedness metrics.


The simulation parameters for each set of simulations are in separate comment blocks in the .py files. 
