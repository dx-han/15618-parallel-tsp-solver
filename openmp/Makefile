APP_NAME=exactopenmp

OBJS=exactopenmp.o

CXX = g++ -m64 -std=c++11
CXXFLAGS = -I. -O3 -Wall -fopenmp -Wno-unknown-pragmas

default: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class