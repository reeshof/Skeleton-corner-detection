
CUDAROOT  = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2
CUDAINC   = C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.2/common/inc
PLATFORM  = PLATFORM_WINDOWS

#--------------------------------------------------------------------------------------------------------------------------------

CXX       =  cl.exe
NVCC      = $(CUDAROOT)/bin/nvcc -ccbin $(CXX)
CUDALIB   = $(CUDAROOT)/lib

CUDAGEN   = -gencode=arch=compute_30,code=\"sm_30,compute_30\"

CCFLAGS   = -I. -Iinclude -m64 -O2 -I$(CUDAROOT)/include $(CUDAINC) -D$(PLATFORM)
CXXFLAGS  = $(CCFLAGS) 
NVCC = $(CUDAROOT)/bin/nvcc -ccbin $(CXX)


OBJECTS = main.o skelft.o vis.o skelft_core.o skelcomp.o

all: skeleton

skeleton: $(OBJECTS)
	$(CXX) -o skeleton $(OBJECTS) $(CXXFLAGS) -L$(CUDALIB) -L$(CUDAROOT)/lib -lcudart

main.o: main.cpp
	$(CXX) /c $< $(CXXFLAGS)

skelft_core.o: skelft_core.cpp
	$(CXX) /c $< $(CXXFLAGS)

vis.o: vis.cpp
	$(CXX) /c $< $(CXXFLAGS) /Fovis.o

skelcomp.o: skelcomp.cpp
		$(CXX) /c $< $(CXXFLAGS)

skelft.o: skelft.cu
	$(NVCC) /c $(CCFLAGS) -use_fast_math $(CUDAGEN) --ptxas-options=-v --maxrregcount 50 skelft.cu


clean:
	del *.o skeleton
