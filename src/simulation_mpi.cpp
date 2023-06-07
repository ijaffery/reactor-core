#include "simulation_mpi.hpp"
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <numeric>

static int mpi_rank;
static int mpi_size;
static MPI_Datatype neutron_type;

/**
 * Initialize MPI and define MPI datatype for neutron_struct.
 * 
 * @param argc The number of arguments passed to the program.
 * @param argv An array of strings containing the arguments passed to the program.
 * 
 * @return void
 */
bool initialize_mpi(int argc, char* argv[])
{
	// Initialize MPI.
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
	if(MPI_Query_thread(&provided) != MPI_THREAD_SINGLE)
	{
		return false;
	}

	// Get the rank and size of the MPI communicator.
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// Define MPI datatype for neutron_struct.
	define_neutron_datatype();

	return true;
}

/**
 * Get the rank of the calling process within the MPI communicator.
 * 
 * @param void
 * 
 * @return The rank of the calling process within the MPI communicator.
 */
int get_rank()
{
	return mpi_rank;
}

/**
 * Finalize MPI.
 * 
 * @param void
 * 
 * @return void
 */
void finalize_mpi()
{
	// Finalize MPI.
	MPI_Finalize();
}

/**
 * Define MPI datatype for neutron_struct.
 * The datatype consists of three blocks of different types: 
 * MPI_SHORT, MPI_DOUBLE, and MPI_CXX_BOOL.
 * 
 * @param void
 * 
 * @return void
 */
void define_neutron_datatype()
{
	// Define block_lengths array.
	const int block_lengths[3] = {3, 4, 1};

	// Define displacements array.
	MPI_Aint displs[3];

	// Define dummy struct and get base address.
	neutron_struct dummy;
	MPI_Aint base_addr;
	MPI_Get_address(&dummy, &base_addr);

	// Get addresses of dummy struct members and calculate displacements.
	MPI_Get_address(&dummy.x, &displs[0]);
	MPI_Get_address(&dummy.pos_x, &displs[1]);
	MPI_Get_address(&dummy.active, &displs[2]);
	displs[0] = MPI_Aint_diff(displs[0], base_addr);
	displs[1] = MPI_Aint_diff(displs[1], base_addr);
	displs[2] = MPI_Aint_diff(displs[2], base_addr);

	// Define types array.
	MPI_Datatype types[3] = {MPI_SHORT, MPI_DOUBLE, MPI_CXX_BOOL};

	// Create MPI datatype for neutron_struct using block_lengths, displs, and types arrays.
	MPI_Datatype tmp_type;
	MPI_Type_create_struct(3, block_lengths, displs, types, &tmp_type);

	// Resize the MPI datatype to match the size of neutron_struct.
	MPI_Type_create_resized(tmp_type, 0, sizeof(neutron_struct), &neutron_type);

	// Commit the MPI datatype.
	MPI_Type_commit(&neutron_type);
}

/**
 * Distribute neutron data from root process to other processes using MPI scatter and scatterv.
 *
 * @param neutrons A vector of neutron_structs containing neutron data.
 * @param neutrons_count The number of neutrons in the neutrons vector.
 *
 * @return long int The number of neutrons after distribution to the current process.
 */
long int distribute_neutrons_from_root(std::vector<neutron_struct>& neutrons, long int neutrons_count)
{
	int local_count, count, extra;
	std::vector<int> displs(mpi_size);
	std::vector<int> counts(mpi_size);

	// Root process calculates and distributes counts of neutrons to each process.
	if(mpi_rank == ROOT)
	{
		count = neutrons_count / mpi_size;
		extra = neutrons_count % mpi_size;
		for(int i = 0; i < mpi_size; ++i)
		{
			counts[i] = count;
			if(i == mpi_size - 1)
			{
				counts[i] = count + extra;
			}
		}
	}

	// Each process receives their count from root process.
	MPI_Scatter(counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	// Root process calculates and distributes displacements of neutrons to each process.
	if(mpi_rank == ROOT)
	{
		displs[0] = 0;
		for(int i = 1; i < mpi_size; ++i)
		{
			displs[i] = displs[i - 1] + counts[i - 1];
		}
	}

	// Distribute neutrons to each process using MPI_Scatterv.
	MPI_Scatterv(neutrons.data(),
				 counts.data(),
				 displs.data(),
				 neutron_type,
				 mpi_rank == ROOT ? MPI_IN_PLACE : neutrons.data() + neutrons_count,
				 local_count,
				 neutron_type,
				 ROOT,
				 MPI_COMM_WORLD);

	// Return the number of neutrons distributed to the current process.
	if(mpi_rank == ROOT)
	{
		return local_count;
	}
	return neutrons_count + local_count;
}

/**
 * Gathers neutron data from all MPI processes to the root process and appends the gathered data to
 * a vector.
 *
 * @param neutrons A reference to a vector of neutron_struct that contains the neutrons from the
 * fuel channel.
 * @param neutrons_to_root A reference to a vector of neutron_struct that contains the neutrons to
 * be gathered to the root process.
 * @param neutrons_count The number of neutrons in the `neutrons` vector.
 *
 * @return The total number of neutrons after the gathered neutrons are appended to the `neutrons`
 * vector. Returns the same value as `neutrons_count` for non-root processes.
 */
long int append_neutrons_to_root(std::vector<neutron_struct>& neutrons,
								 std::vector<neutron_struct>& neutrons_to_root,
								 long int neutrons_count)
{
	// Define vectors to hold displacements and counts for MPI_Gatherv() function.
	std::vector<int> displs(mpi_size);
	std::vector<int> counts(mpi_size);

	// Get the number of neutrons in the `neutrons_to_root` vector for each MPI process and gather
	// the results to the `counts` vector on the root process.
	int neutrons_in_fuel_channel_count = neutrons_to_root.size();
	MPI_Gather(&neutrons_in_fuel_channel_count, 1, MPI_INT, counts.data(), 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	// Calculate the displacements for the root process.
	if(mpi_rank == ROOT)
	{
		displs[0] = 0;
		for(int i = 1; i < mpi_size; ++i)
		{
			displs[i] = displs[i - 1] + counts[i - 1];
		}
	}

	// Gather the neutron data to the root process using MPI_Gatherv() function.
	MPI_Gatherv(neutrons_to_root.data(),
				neutrons_to_root.size(),
				neutron_type,
				mpi_rank == ROOT ? neutrons.data() + neutrons_count : nullptr,
				counts.data(),
				displs.data(),
				neutron_type,
				ROOT,
				MPI_COMM_WORLD);

	// Calculate the total number of neutrons on the root process.
	if(mpi_rank == ROOT)
	{
		long int total_transferred_count = std::accumulate(counts.begin(), counts.end(), 0, std::plus<int>{});
		return neutrons_count + total_transferred_count;
	}

	// Return the number of neutrons for non-root processes.
	return neutrons_count;
}

/**
 * Reduce the sum of integers sent by all processes to the root process.
 * 
 * @param send The integer value to be sent.
 * @param recv Pointer to the memory location where the result will be stored.
 * 
 * @return void
 */
void reduce_int_sum_in_root(long int send, long int* recv)
{
	// Reduce the sum of integers sent by all processes to the root process.
	MPI_Reduce(&send, recv, 1, MPI_LONG, MPI_SUM, ROOT, MPI_COMM_WORLD);
}
