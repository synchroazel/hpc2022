#!/bin/bash

#PBS -l select=4:ncpus=16:mem=4gb
#PBS -l walltime=0:60:00
#PBS -q short_cpuQ

module load mpich-3.2

export PERFORMANCE_CHECKS=TRUE

mpirun.actual -np 64 ~/hpc2022/cmake-build-debug/hpc2022 -l tuning \
                                   -i ~/hpc2022/data/iris_train.csv \
                                   -I ~/hpc2022/data/iris_validation.csv \
                                   -H ~/hpc2022/hyperparameters.json \
                                   -r 70 -R 30 -c 5 -t 5
