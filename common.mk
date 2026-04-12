CUDA_ARCH = 86
ARCH = -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
OPTIONS = -O3 -use_fast_math -w -std=c++17

MGPU_DIR = ext/moderngpu/include/
CUB_DIR =
BOOST_DIR = /net/ugrads/skunapan/pvt/.local/include/
GRB_DIR = $(dir $(lastword $(MAKEFILE_LIST)))

BOOST_DEPS = /net/ugrads/skunapan/pvt/.local/lib/
GRB_DEPS = ext/moderngpu/src/mgpucontext.cu \
           ext/moderngpu/src/mgpuutil.cpp

LIBS = -L${BOOST_DEPS} \
			 -Xlinker -rpath -Xlinker ${BOOST_DEPS} \
			 -lboost_program_options \
       -lcublas \
			 -lcusparse \
			 -lcurand
