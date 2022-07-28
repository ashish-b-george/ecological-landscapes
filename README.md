# Overview of workflow

The script Submit_CR_SSjobs.py submits simulation jobs over the desired parameter range.
There are 10 replicate simulations for each set of parameters. A replicate varies only by the realization of the consumption matrices of the species (chosen randomly from appropriate distribution.) 
Each set of parameters has a separate directory, with simulations for each replicate in a separate sub-directory within this directoy. Thus files are stored in a tree-like directory form.

The script Submit_CR_SSjobs.py also runs analyze_landscape.py after the simulations are completed. analyze_landscape.py calculates the various ruggedness metrics.


The simulation parameters for each set of simulations are in separate comment blocks in the .py files. 
