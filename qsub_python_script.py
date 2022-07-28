#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:13:29 2019

@author: ashish
qsubs arbitrary python script on the cluster
"""

'''
needs to be run on cluster every time. ( just python3 did not work for a while dirichlet distribution was broken, & module load python3/3.6.5 was used)
module load python3/3.6.9
export PACKAGEINSTALLDIR=/projectnb/qedksk/ashishge/mypython
export PYTHONPATH=$PACKAGEINSTALLDIR/lib/python3.6/site-packages:$PYTHONPATH
'''


import subprocess
import argparse
import os
import numpy as np
import sys
if os.getcwd().startswith("/Users/ashish"):
    print ("this is for the cluster.")
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
   
parser = argparse.ArgumentParser()
parser.add_argument("-n", help="file name", default=None) ## only a single folder handled for now!args = parser.parse_args()
args = parser.parse_args()
if args.n is not None:
    python_file_name=args.n#


assert python_file_name[-3:]=='.py', "runs python file only"

subscript_str = f"""#!/bin/bash -l
module load python3/3.6.9
module load cvxpy/1.0.25
export PACKAGEINSTALLDIR=/projectnb/qedksk/ashishge/mypython
export PYTHONPATH=$PACKAGEINSTALLDIR/lib/python3.6/site-packages:$PYTHONPATH
#$ -j y
#$ -l h_rt=10:00:00
python {python_file_name}"""

with open('sub_python_script.sh', "w") as script:
        print(subscript_str, file = script)
        script.close()
qsub_command = "qsub sub_python_script.sh"
print(qsub_command)
exit_status = subprocess.call(qsub_command, shell=True)
if exit_status == 1:  # Check to make sure the job submitted        
    print("\n Job {0} failed to submit".format(qsub_command))
    
    
os.remove("sub_python_script.sh")    
    
    
    
    
    