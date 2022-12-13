#!/bin/bash

#PBS -l select=1:ncpus=1:mem=4gb -l place=pack:excl
#PBS -l walltime=00:60:00
#PBS -q short_cpuQ

module load mpich-3.2

export PERFORMANCE_CHECKS=TRUE

mpirun.actual -np 64 ~/hpc2022/cmake-build-debug/hpc2022 -l tuning \
                                   -i ~/hpc2022/data/iris_training.csv \
                                   -I ~/hpc2022/data/iris_validation.csv \
                                   -H ~/hpc2022/hyperparameters.json \
                                   -r 70 -R 30 -c 5 -t 5
