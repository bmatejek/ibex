CXX = g++-4.8
CPPFLAGS = -O4 -std=c++11
LDFLAGS = -O4
OBJS = multicut.o
INC_DIR = -I/home/brian/software/graph/include -I/opt/gurobi702/linux64/include -I/opt/gurobi702/linux64/src/cpp
LIB_DIR = -L/opt/gurobi702/linux64/src/build -L/opt/gurobi702/linux64/lib
EXTERNAL_LIBS = -lgurobi_c++ -lgurobi70

multicut: $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(LIB_DIR) $(EXTERNAL_LIBS) -o ../bin/multicut

multicut.o: multicut.cpp
	$(CXX) $(CPPFLAGS) $(INC_DIR) -c multicut.cpp

clean:
	rm -f *~ *.o \#* multicut

