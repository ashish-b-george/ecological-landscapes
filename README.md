This repository contains code related to the publication: 
Ashish B. George and Kirill S. Korolev. "Ecological landscapes guide the assembly of optimal microbial communities."  Biorxiv (2022)

Assembling optimal microbial communities is key for various applications in biofuel production, agriculture, and human health. Finding the optimal community is challenging because the number of possible communities grows exponentially with the number of species, and so an exhaustive search cannot be performed even for a dozen species. Here we analyze heuristic search strategies for finding the optimal community using data from experiments and simulations of microbial consumer-resource models. In particular, we analyze the "landscapes" of community function and identify statistical metrics that control the efficacy of the search process. These metrics, called ruggedness metrics, can be inferred from preliminary, noisy data. We also study how various biological properties such as niche overlap, resource distributions, and cross-feeding affect search success. Overall, our results establish the conditions necessary for the success of the heuristic search and provide concrete design principles for building high-performing microbial consortia




Part of  code generates and analyzes data of microbial communities 
s

## Description of files and folders

1) files in the main directory are the files required to make the figures in the main text and SI.

2) the community_simulator folder containts code from the Community simulator package [Marsland et al 2020 Plos One]. The folder with the package is provided because the package was modified for use in the simulations. Some of these modifications, such as using different ode solvers for simulation stability, may have been implemented in more recent versions of community simulator.

3) the 'vestigial files' folder contains code for related analysis of ecological landscapes that is not presented in the paper.

## Overview of simulation workflow

The script Submit_CR_SSjobs.py submits simulation jobs over the desired parameter range.
There are 10 replicate simulations for each set of parameters. A replicate varies only by the realization of the consumption matrices of the species (chosen randomly from appropriate distribution.) 
Each set of parameters has a separate directory, with simulations for each replicate in a separate sub-directory within this directoy. Thus files are stored in a tree-like directory form.

The script 'Submit_CR_SSjobs.py' also runs 'analyze_landscape.py' after the simulations are completed. analyze_landscape.py calculates the various ruggedness metrics and other quantities concerning landscape structure.'aggregate_analysis_output.py' aggregates the analysis results of the different simulations into a single data structure that can be used for plotting and further inspection. 'average_analysis_output.py' has a similar function but it averages the results instead of aggregating it; this is used only for a couple of figures. 

The simulation parameters for each set of simulations are in separate comment blocks in the .py files. 


The jupyter notebook file 'figures_final.ipynb' has the code used to make the figures the publication based on the analysis data.
