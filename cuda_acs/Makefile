
EXECUTABLE := cudaTspAcs

CU_FILES   := tsp_acs.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -std=c++11
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 -std=c++11 --gpu-architecture compute_61 -ccbin /usr/bin/gcc  -I ../cub-1.8.0 -lineinfo

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/tsp_acs.o


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
