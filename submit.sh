#!/bin/bash

#PBS -l select=1:ncpus=2:mem=2gb
#PBS -l walltime=0:01:00
#PBS -q short_cpuQ

module load mpich-3.2

mpirun.actual -np 64 ./cmake-build-debug/hpc2022 -l tuning \
                                   -i /antonio.padalino/hpc2022/data/iris_train.csv \
                                   -I /antonio.padalino/hpc2022/data/iris_validation.csv \
                                   -H /antonio.padalino/hpc2022/hyperparameters.json \
                                   -r 70 -R 30 -c 5 -t 5
