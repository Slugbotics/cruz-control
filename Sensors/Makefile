CXX = g++
CXXFLAGS = -Wall -Wextra -Wconversion -Wdouble-promotion -pedantic -std=c++11
LFLAGS =
HEADERS =

all: meta

meta: metavision_evt3_raw_file_decoder.o
	$(CXX) $^ $(CXXFLAGS) -o $@

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -rf meta *.o

format:
	clang-format -i -style=file *.[ch]

.PHONY: all clean format

