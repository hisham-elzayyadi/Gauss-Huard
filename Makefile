# This is a simple standalone example. See README.txt
# Initially it is setup to use OpenBLAS.
# See magma/make.inc for alternate BLAS and LAPACK libraries,
# or use pkg-config as described below.

# Paths where MAGMA, CUDA, and OpenBLAS are installed.
# MAGMADIR can be .. to test without installing.
#MAGMADIR     ?= ..
MAGMADIR     ?= /usr/local/magma
CUDADIR      ?= /usr/local/cuda
OPENBLASDIR  ?= /usr/lib/x86_64-linux-gnu/openblas

CC            = g++
FORT          = gfortran
LD            = g++
CFLAGS        = -Wall
# needs -fopenmp if MAGMA was compiled with OpenMP
LDFLAGS       = -Wall #-fopenmp


# ----------------------------------------
# Flags and paths to MAGMA, CUDA, and LAPACK/BLAS
 MAGMA_CFLAGS     := -DADD_ \
                     -I$(MAGMADIR)/include \
                     -I$(MAGMADIR)/sparse/include \
                     -I$(CUDADIR)/include
 
 MAGMA_F90FLAGS   := -Dmagma_devptr_t="integer(kind=8)" \
                     -I$(MAGMADIR)/include
 
# # may be lib instead of lib64 on some systems
 MAGMA_LIBS       := -L$(MAGMADIR)/lib -lmagma_sparse -lmagma \
                     -L$(CUDADIR)/lib64 -lcublas -lcudart -lcusparse \
                     -L$(OPENBLASDIR)/lib -lopenblas


# ----------------------------------------
# Alternatively, using pkg-config (see README.txt):
#MAGMA_CFLAGS   := $(shell pkg-config --cflags magma)

#MAGMA_F90FLAGS := -Dmagma_devptr_t="integer(kind=8)" \
#                  $(shell pkg-config --cflags-only-I magma)

#MAGMA_LIBS     := $(shell pkg-config --libs   magma)


# ----------------------------------------
default:
	@echo "Available make targets are:"
	@echo "  make all       # compiles example_v1, example_v2, example_sparse, example_sparse_operator, and example_f"
	@echo "  make c         # compiles example_v1, example_v2, example_sparse, example_sparse_operator"
	@echo "  make fortran   # compiles example_f"
	@echo "  make clean     # deletes executables and object files"

all: c fortran

c: example_v1 example_v2 example_sparse example_sparse_operator

fortran: example_f

clean:
	-rm -f trial MAGMABGEPP example_v1 example_v2 example_sparse example_sparse_operator example_f *.o *.mod

.SUFFIXES:


# ----------------------------------------
# C example
%.o: %.cpp
	$(CC) $(CFLAGS) $(MAGMA_CFLAGS) -c -o $@ $<


BGHPP: BGHPP.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS) 
	
BGHPRBT: BGHPRBT.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS) 

BGHPRBT_LA: BGHPRBT_LA.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)
	
BGHPRBTDEL_LA: BGHPRBTDEL_LA.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)	
	
BGEPRBT: BGEPRBT.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)

BGHNP: BGHNP.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)	 
# ----------------------------------------
# Fortran example
# this uses capital .F90 to preprocess to define magma_devptr_t
%.o: %.F90
	$(FORT) $(F90FLAGS) $(MAGMA_F90FLAGS) -c -o $@ $<

example_f: example_f.o
	$(FORT) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)
