# Parallel Support Vector Machines - hpc2022 final project

Support Vector Machines are a popular machine learning algorithm for classification, aimed at finding a hyperplane that
separates the given data points into classes while maximizing the margin between the hyperplane and the closest data
points, called Support Vectors. When the data is not linearly separable, the algorithm can be extended to use kernels to
map the data points into a higher dimensional space where they can be separated by a hyperplane.

Here we implement a parallel version of the Kernel SVM algorithm using MPI Communications. Information about usage
(both local and on cluster) can be found later in the file, together with a working use example.

## Usage on local machine

You can compile the program through

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -DCMAKE_CXX_FLAGS="-O2" -S ./ -B ./cmake-build-optimized
cmake --build cmake-build-optimized --target all
```

The program supports cli arguments. You can inspect them running the program with the `--help` flag.

### A working example

Say you want to run the code on the `iris` dataset, which is included in the repository.

Note that it has been already split into training, validation and test partitions, otherwise you would have needed to
run the `ds_preprocessing.py` script first, specifying the dataset with the `-f` option.

Also note that to perform time checks and benchmarks, the environment variable `PERFORMANCE_CHECKS` needs to be set
to true with:

```bash
export PERFORMANCE_CHECKS=TRUE
```

If you do not need to perform time checks, you can skip this step, and you can safely ignore related warnings in
runtime. On the cluster usage, as we will see later, this step is not needed as the environment variable is already
specified in the job script, and will be set automatically.

You can start by tuning an SVM to find the best parameters for the dataset. Assuming the program has been compiled with
cmake in the folder `cmake-build-optimized` and linked to the `hpc2022` executable, you can run the following:

```bash
mpiexec -np 8 ./cmake-build-optimized/hpc2022 -l tuning \
                    -i ./data/iris_training.csv \
                    -I ./data/iris_validation.csv \
                    -H hyperparameters.json \
                    -r 70 -R 30 -c 5 -t 5
```

The output presents the set of hyperparameters yielding the best accuracy on the validation set. Say the best SVM has
linear kernel and `C=0.01`, you can then train the SVM on the whole training set and test it on the test set with the
following:

```bash
mpiexec -np 8 ./cmake-build-optimized/hpc2022 -l training \
                    -i ./data/iris_training.csv \
                    -S ./saved_svm/ \
                    -r 70 -c 5 -t 5 -k r -C 0.01 -g 1
```

Having trained the SVM, you can now use the saved `.svm` file to classify the test set. The file is stored under the
location specified earlier with the `-S` option. In this case, the file is stored in the `saved_svm` folder. You can now
proceed to the test with the following:

```bash
mpiexec -np 8 ./cmake-build-optimized/hpc2022 -l testing \
                    -i ./data/iris_test.csv \
                    -s ./saved_svm/radialr_C0.010000_G1.000000.svm \
                    -r 19 -c 5 -t 5
```

The output is the accuracy of the SVM on the test set, divided by class.

## Usage on cluster

### Load necessary modules

First, the necessary modules must be loaded:

```bash
module load mpich-3.2
module load cmake-3.15.4
```

- `MPI` is used for parallel communications;
- `CMake` is the compiler used to build the project.

### Build the project

To build the project with CMake, run the following:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -DCMAKE_CXX_FLAGS="-O2" -S ./ -B ./cmake-build-optimized
cmake --build cmake-build-optimized --target all
```

The option `-DCMAKE_CXX_FLAGS="-O2"` is used to optimized code build and speed up runtimes, but is completely optional.

### Submit job request

To submit a job request simply use `qsub` followed by the name of the script with the submission. For example:

```bash
qsub submit.sh
```

The file `submit.sh` included in this repository contains a job submission example in which tuning is performed on the
`iris` dataset using 64 cores.

To monitor the job in each second:

```bash
watch -n 1 qstat <job_ID>
```