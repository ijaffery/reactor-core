SRC=src/simulation_configuration.c src/main.cpp src/simulation_support.c src/simulation_mpi.cpp
CC=mpicc
CXXFLAGS=-O3 -fopenmp
CFLAGS=-cc=icpc $(CXXFLAGS)
LDFLAGS=-lm
TARGET=reactor

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) -Isrc $(SRC) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)