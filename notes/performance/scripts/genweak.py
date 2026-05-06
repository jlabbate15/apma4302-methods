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
parser.add_argument('-lev0', type=int, default=3, metavar='X',
                    help='refinement level for a single core 3 is 17x17x17''')
parser.add_argument('-Plevel', type=int, default=4, metavar='P',
                    help='maximum refinement level for MPI processes P=(Plevel+1)**3')
parser.add_argument('-minutes', type=int, default=60, metavar='T',
                    help='''max time in minutes for SLURM job''')
parser.add_argument('-pernode', type=int, default=8, metavar='K',
                    help='''maximum number of MPI processes to assign to each node;
small value may increase streams bandwidth and performance''')

args = parser.parse_args()

print('settings: %d max tasks per node, %s as email, request time %d minutes'
      % (args.pernode,args.email,args.minutes))

levels = args.Plevel - args.lev0
Plist = [ (p+1)**3 for p in range(args.Plevel) ]
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
module load singularity 
export APPTAINER_TMPDIR=$SINGULARITY_TMPDIR
export APPTAINER_BINDPATH=$SINGULARITY_BINDPATH
module load openmpi/gcc/64/4.1.7a1

# set the container for firedrake
SIF=/burg/home/mws6/sifs/firedrake-ts.sif

# Launches the MPI application.
GO="mpiexec -n $SLURM_NTASKS apptainer exec  $SIF"
cd $SLURM_SUBMIT_DIR
'''

rawfish = r'''
# FISH:  solve 3D Poisson equation
# using optimal CG+GMG solver
# -da_refine 7 is 257x257x257
# ~ 12 Gb memory
# coarse grid is 9x9x9 so should work up to several hundred processors
$GO ./fish -fsh_dim 3 -da_refine %d -pc_mg_levels %d -pc_type mg -ksp_type cg -snes_type ksponly -ksp_converged_reason -ksp_monitor -log_view


'''


for l,P in enumerate(Plist):
    rlev = args.lev0 + l  # refinement level
    grid = 2**(rlev+1) + 1
    grid0 = 2**(args.lev0+1)+1
    wrun = rawfish % (rlev, rlev-1)

    pernode = args.pernode
    nodes = np.ceil(P / pernode)
    print(f'  case: {nodes} nodes, {pernode} tasks per node, and P={P} processes on {grid}x{grid}x{grid} grid')

    root = f'weak_fish_N0{grid0}_P{P:03}_p{pernode}'
    preamble = rawpre % (args.account,P,pernode,args.minutes,args.email,
                         root + r'.o.%j')

    batchname = root + '.sh'
    print('    writing %s ...' % batchname)
    batch = open(batchname,'w')
    batch.write(preamble)
    batch.write(wrun)
    batch.close()

