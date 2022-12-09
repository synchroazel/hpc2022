# Parallel Support Vector Machines - hpc2022 final project

TODO: some generic stuff here

## Usage on local machine

You can compile the program through

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -DCMAKE_CXX_FLAGS="-O2" -S ./ -B ./cmake-build-optimized
cmake --build cmake-build-optimized-o2 --target all
```

The program supports cli arguments. You can inspect them running the program with the `--help` flag.

### A working example

Let's say you want to run the code on the `iris` dataset.

Note that the `iris` dataset included in this repository has been already split into training, validation and test
partitions, otherwise you would have needed to run the `ds_preprocessing.py` script first, specifying the dataset
with the `-f` option.

You can start by tuning a SVM to find the best parameters for the dataset. Assuming the program has been compiled with
cmake in the folder `cmake-build-optimized-o2` and linked to the `hpc2022` executable, you can run the following:

```bash
mpiexec -np 8 ./cmake-build-debug-o2/hpc2022 -l tuning \
                    -i ./data/iris_training.csv \
                    -I ./data/iris_validation.csv \
                    -H hyperparameters.json \
                    -r 70 -R 30 -c 5 -t 5
```

The output presents the set of hyperparameters yielding the best accuracy on the validation set. Say the best SVM has
linear kernel and `C=0.01`, you can then train the SVM on the whole training set and test it on the test set with the
following:

```bash
mpiexec -np 8 ./cmake-build-optimized-o2/hpc2022 -l training \
                    -i ./data/iris_training.csv \
                    -S ./saved_svm/ \
                    -r 70 -c 5 -t 5 -k r -C 0.01 -g 1
```

Having trained the SVM, you can now use the saved `.svm` file to classify the test set. The file is stored under the
location specified earlier with the `-S` option. In this case, the file is stored in the `saved_svm` folder. You can now
proceed to the test with the following:

```bash
mpiexec -np 8 ./cmake-build-optimized-o2/hpc2022 -l testing \
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
cmake --build cmake-build-optimized --target all -j 8
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