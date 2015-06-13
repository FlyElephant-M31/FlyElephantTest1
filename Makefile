# defines
CC=icc
MPICC=mpiicc
CXX=icpc
NVCC=nvcc
MPICXX=mpiicpc
NVCCFLAGS+=-arch compute_35 -code sm_35 -O3 -std=c++11 -Xptxas -fastimul
OPTFLAGS=-openmp
OPTFLAGS+=-xHost -unroll -unroll-loops -parallel
OPTFLAGS+=-opt-report=5 -opt-report-phase=openmp,par,vec
OPTFLAGS+=-inline-level=2 -inline-forceinline -ipo
ASMFLAGS=-S -fsource-asm -fverbose-asm -fcode-asm
CFLAGS= -O3 -Wall -std=gnu99
CXXFLAGS= -O3 -Wall -std=c++11
LDFLAGS= -O3 -lrt

IMPLEMENTATIONS = mst_cpu mst_gpu mst_mpi
TARGET = gen_valid_info validation gen_RMAT gen_SSCA2 $(IMPLEMENTATIONS)

# your own implementation, executable must called mst
mst: mst_cpu
	echo "#!/bin/sh" > mst
	echo KMP_AFFINITY=compact ./$< '$$*' >> mst
	chmod a+x mst

mst_mpi: mst
	cp $< $@

all: $(TARGET)

mst_mpi_%: main_mpi.mpi.o mst_%.mpi.o graph_tools.mpi.o gen_RMAT_mpi.mpi.o gen_SSCA2_mpi.mpi.o
	$(MPICXX) -lpmi $^ -o $@ $(LDFLAGS) $(OPTFLAGS)

mst_cuda_%: main.cuda.o mst_%.cuda.o graph_tools.cuda.o
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(NVCCLDFLAGS)

mst_%: main.o mst_%.o graph_tools.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(OPTFLAGS)

# reference implementation
mst_reference: main.o mst_reference.o graph_tools.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(OPTFLAGS)

gen_RMAT: gen_RMAT.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(OPTFLAGS)

gen_SSCA2: gen_SSCA2.o graph_tools.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(OPTFLAGS)

validation: validation.o graph_tools.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(OPTFLAGS)

gen_valid_info: graph_tools.o mst_reference.o gen_valid_info.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(OPTFLAGS)

%.mpi.o: %.cpp
	$(MPICXX) -DUSE_MPI $(CXXFLAGS) $(OPTFLAGS) $(ASMFLAGS) $<
	$(MPICXX) -DUSE_MPI $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $<

%.cuda.o: %.cu
	$(NVCC) $(NVCCFLAGS) -cubin -o $@.cubin $<
	nvdisasm $@.cubin > $@.ptx
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.cuda.o: %.c
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.cuda.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.mpi.o: %.c
	$(MPICC) -DUSE_MPI $(CFLAGS) $(OPTFLAGS) -o $@ -c $<

.cpp.o:
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(ASMFLAGS) $<
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $<

.c.o:
	$(CC) $(CFLAGS) $(OPTFLAGS) $(ASMFLAGS) $<
	$(CC) $(CFLAGS) $(OPTFLAGS) -o $@ -c $<

clean:
	rm -rf *.o *.cod *.s *.optrpt $(TARGET)

