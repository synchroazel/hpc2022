# Parallel Support Vector Machines - hpc2022 final project

TODO: some generic stuff here


## Usage on local machine

An example of the program usage is the following:

```bash
mpiexec -np 8 ./cmake-build-debug/hpc2022 -l tuning \
                    -i ./data/iris_training.csv \
                    -I ./data/iris_validation.csv \
                    -H hyperparameters.json \
                    -r 70 -R 30 -c 5 -t 5
```

this will perform tuning on the <tt>iris</tt> dataset, with the specified number of columns and rows for the training
and validation sets, respectively, and the hyperparameters inside the <tt>hyperparameters.json</tt> file. 

The usage of all available arguments can be found by running_
    
```bash
mpiexec -np  ./cmake-build-debug/hpc2022 -h
```

## Usage on cluster

### Load necessary modules

First, the necessary modules must be loaded:

```bash
module load mpich-3.2
module load cmake-3.15.4
```

- MPI is used, of course, for parallel communications;
- CMake is the compiler used to build the project.

### Build the project

To build the project with CMake, run the following:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -S ./ -B ./cmake-build-debug
cmake --build cmake-build-debug --target all -j 8
```

To build the optimized version (<tt>-O2</tt> flag), run the following:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -DCMAKE_CXX_FLAGS="-O2" -S ./ -B ./cmake-build-optimized
cmake --build cmake-build-optimized --target all -j 8t
```

### Submit job request

To finally submit the job request:

```bash
qsub submit.sh
```

and to monitor the job in each second:

```bash
watch -n 1 qstat <job_ID>
```