CXX = nvcc
CXX2 = gcc
CXXFLAGS2 = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= Homework

all : $(TARGET1)
    
$(TARGET1) : Homework.cu kernel.cu Chi2.h
	$(CXX) $(CXXFLAGS3) -lm -o $(TARGET1) Homework.cu kernel.cu
clean : 
	rm -f $(TARGET1)
