#!/bin/bash
#PBS -l select=8:ncpus=8:mem=4gb
#PBS -l walltime=0:60:00
#PBS -q short_cpuQ
module load mpich-3.2
export PERFORMANCE_CHECKS=TRUE




mpirun.actual -np 64~/hpc2022/cmake-build-optimized/hpc2022 -l tuning \
				-i ./data/gene_expr_training.csv \
				-I ./data/gene_expr_validation.csv \
				-r 55 				-R 16 -c 2001 -t 2001 \
				             