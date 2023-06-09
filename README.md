# Nuclear Reactor Simulation Program

This is a parallel program for simulating a nuclear reactor. The program uses MPI for parallelization.

## Files

The following files are included in this program:

- `src/main.cpp`: Main program file.
- `src/simulation_configuration.c` and `src/simulation_configuration.h`: Configuration files for the
  simulation.
- `src/simulation_mpi.cpp` and `src/simulation_mpi.hpp`: MPI code for parallelizing the simulation.
- `src/simulation_support.c` and `src/simulation_support.h`: Support functions for the simulation.
- `app.job`: Job submission script for running the program.
- `config_simple.txt`: Configuration file for the simulation.
- `makefile`: Makefile used to build the program.
- `reactor`: Executable file generated by the makefile.
- `README.md`: This file.
- `result.txt`: Output file for the simulation.

## Building the Program

To build the program, load the necessary modules by running the following commands:

```bash
module load mpt
module load intel-compilers-19
module load gcc
```
Then, run the command `make` in the terminal to build the program.

## Running the Program

To run the program, use the following command format:

```bash
mpirun -np <number_of_processes> ./reactor <configuration_file> <result_file>
```

Here, `<number_of_processes>` is the number of MPI processes to use, `<configuration_file>` is the
path to the configuration file for the program (use `config_simple.txt`), and `<result_file>` is the
path to the output file (use `result.txt`).

The configuration file `config_simple.txt` contains various parameters that can be modified for the
simulation, such as the size of the reactor, the fuel proportion, the moderator, and the control rod
insertion. The program will output the results in both the file and console based on the frequency
set in the configuration file.

To set the OpenMP threads, use the command `export OMP_NUM_THREADS=<number_of threads>`.

It is recommended to use only one thread for this simulation.

## Running a batch job

To run the nuclear reactor simulation program, the SLURM batch system must be used. To
submit a batch job, use the following command: `sbatch app.job`. Once the job is submitted, the
batch system will provide a unique ID for the job. It can be monitored using the command `squeue -u
$USER`. After the job finishes, the output will be available in a file named `slurm-XXXXX.out` in
the root directory. Ensure that the #SBATCH --account value is set to an appropriate account name to
utilize the machine.

To modify the number of MPI processes or threads used, edit the `app.job` file and change the
values of the `tasksPerNode` (to modify the number of MPI processes used) and `cpusPerTask` (to
modify the number of threads used) variables.