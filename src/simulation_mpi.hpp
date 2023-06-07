#include "simulation_support.h"
#include <vector>

#define ROOT 0

bool initialize_mpi(int, char**);
int get_rank();
void finalize_mpi();
void define_neutron_datatype();
long int distribute_neutrons_from_root(std::vector<neutron_struct>&, long int);
long int append_neutrons_to_root(std::vector<neutron_struct>&, std::vector<neutron_struct>&, long int neutrons_count);
void reduce_int_sum_in_root(long int, long int*);