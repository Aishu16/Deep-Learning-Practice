CXX = gcc
CXXFLAGS = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3
#CXXFLAGS3 =
TARGET1= hw1

all : $(TARGET1)

$(TARGET1) : hw1.c kernel.c Chi2.h
	$(CXX) $(CXXFLAGS) -o $(TARGET1) hw1.c kernel.c -lm -fopenmp
clean :
	rm -f $(TARGET1)



