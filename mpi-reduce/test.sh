#!/bin/bash
#PBS -l select=1:ncpus=4:mem=2gb # set max execution time
#PBS -l walltime=0:05:00
# imposta la coda di esecuzione #PBS -q short_cpuQ
module load mpich-3.2 mpirun.actual -n 2 ./main