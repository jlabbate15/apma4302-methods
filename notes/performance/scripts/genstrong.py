#!/usr/bin/env python3

# WARNING:  You will need to edit this file to match your batch system!!

from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np

intro = '''
Write SLURM batch files for strong scaling study using ch6/fish.c.  Example:
    ./genstrong.py -email xx@yy.edu -lev 6 -queue t2standard -minP 4 -maxP 64 -pernode 4 -minutes 60
Solves 3D poisson equation using  Newton CG+GMG solver
and 9x9x9 coarse grid.  With -lev 7 (i.e. -da_refine 7) is 257x257x257
fine grid with N = 16974593 degrees of freedom.
'''

parser = ArgumentParser(description=intro,formatter_class=RawTextHelpFormatter)
parser.add_argument('-account',metavar='ACCOUNT', type=str, 
                     default='apam',help='Slurm account')
parser.add_argument('-email', metavar='EMAIL', type=str,
                    default='USERNAME@columbia.edu', help='email address')
parser.add_argument('-lev', type=int, default=7, metavar='X',
                    help='''refinement level for -da_refine in {4,5,6,7,8}''')
parser.add_argument('-maxP', type=int, default=16, metavar='P',
                    help='''maximum number of MPI processes;
power of 2 like 8,16,64,128,... recommended''')
parser.add_argument('-minP', type=int, default=2, metavar='P',
                    help='''minimum number of MPI processes;
power of 2 like 1,2,4,8,16,... recommended''')
parser.add_argument('-minutes', type=int, default=120, metavar='T',
                    help='''max time in minutes for SLURM job''')
parser.add_argument('-pernode', type=int, default=2, metavar='K',
                    help='''maximum number of MPI processes to assign to each node;
small value may increase streams bandwidth and performance''')

args = parser.parse_args()

print('settings: %d max tasks per node, %s as email, request time %d minutes'
      % (args.pernode,args.email,args.minutes))

m_min = int(np.floor(np.log(float(args.minP)) / np.log(2.0)))
m_max = int(np.floor(np.log(float(args.maxP)) / np.log(2.0)))
Plist = np.round(2.0**np.arange(m_min,m_max+1)).astype(int).tolist()
print('runs (ch6/fish.c) will use P in'),
print(Plist)

rawpre = r'''#!/bin/bash
#SBATCH --account=%s
#SBATCH --exclusive
#SBATCH --ntasks=%d
#SBATCH --tasks-per-node=%d
#SBATCH --time=%d
#SBATCH --mail-user=%s
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%s

# set some environment variables for apptainer
module load openmpi/gcc/64/4.1.7a1
module load singularity 
export APPTAINER_TMPDIR=$SINGULARITY_TMPDIR
export APPTAINER_BINDPATH=$SINGULARITY_BINDPATH

# set the container for firedrake
SIF=/burg/home/mws6/sifs/firedrake-ts.sif

# Launches the MPI application.
GO="mpiexec -n $SLURM_NTASKS apptainer exec  $SIF"
cd $SLURM_SUBMIT_DIR
'''

rawminimal = r'''
# FISH:  solve 3D Poisson equation
# using optimal CG+GMG solver
# -da_refine 7 is 257x257x257
# ~ 12 Gb memory
# coarse grid is 9x9x9 so should work up to several hundred processors
$GO ./fish -fsh_dim 3 -da_refine %d -pc_mg_levels %d -pc_type mg -ksp_type cg -snes_type ksponly -ksp_converged_reason -ksp_monitor -log_view

'''


for P in Plist:
    grid = 2**(args.lev+1) + 1
    run = rawminimal % (args.lev,args.lev-1)

    pernode = min(P,args.pernode)
    nodes = P / pernode
    print('  case: run with %d nodes, %d tasks per node, and P=%d processes'
          % (nodes,pernode,P))
    print('        on %d x %d x %d grid; each process has %d degrees of freedom'
          % (grid,grid,grid, grid**3/P))

    root = f"strong_fish_N{grid:03}_P{P:03}_p{pernode:02}"
    preamble = rawpre % (args.account,P,pernode,args.minutes,args.email,
                         root + r'.o.%j')

    batchname = root + '.sh'
    print('    writing %s ...' % batchname)
    batch = open(batchname,'w')
    batch.write(preamble)
    batch.write(run)
    batch.close()

